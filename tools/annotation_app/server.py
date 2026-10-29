"""FastAPI Server for CVC Distal-Tip Visual Annotation Tool.

Features:
- Native-resolution radiograph streaming with zoom & pan.
- Individual polyline overlay per CVC instance with stable IDs.
- Click-to-mark distal tip native coordinates (no automatic endpoint snapping).
- Explicit states: assessable, uncertain, outside_image, annotation_incorrect.
- Study-completion validation covering all inventory-listed CVCs + uncataloged CVCs.
- Versioned saves to JSON and Parquet/CSV registries.
- Reviewers work blinded from model predictions.
"""
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
from typing import Dict, List, Optional, Any, Literal
import math
import hashlib

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
STATIC_DIR = Path(__file__).resolve().parent / "static"
REVIEWS_DIR = PROJECT_ROOT / "data" / "reviews"
REVIEWS_DIR.mkdir(parents=True, exist_ok=True)

# Load configuration and inventories
with open(PROJECT_ROOT / "configs" / "pilot.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

RAW_IMAGES_DIR = Path(cfg["data"]["raw_dir"]) / "train"
INV_DF = pd.read_parquet(PROJECT_ROOT / "data" / "annotation_inventory.parquet")
STUDY_DF = pd.read_parquet(PROJECT_ROOT / "data" / "study_inventory.parquet")
REVIEWED_V3_DF = pd.read_parquet(PROJECT_ROOT / "data" / "reviewed_distal_tips_v3.parquet")

with open(PROJECT_ROOT / "data" / "splits.json", "r", encoding="utf-8") as f:
    SPLITS = json.load(f)

app = FastAPI(title="CVC Distal-Tip Visual Annotation Tool", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
QUEUE_NAMES = {"positive_expansion_200", "validation_v4_completion", "provisional_pilot_25", "carina_placement_pilot_50"}
DUAL_REVIEW_QUEUE = "carina_placement_pilot_50"


def _review_file(queue_name: str, study_id: str, reviewer_id: str) -> Path:
    if queue_name == DUAL_REVIEW_QUEUE:
        reviewer_key = hashlib.sha256(reviewer_id.strip().casefold().encode("utf-8")).hexdigest()[:20]
        return REVIEWS_DIR / queue_name / study_id / reviewer_key / "review.json"
    return REVIEWS_DIR / queue_name / f"{study_id}.json"


# Models
class CatheterReviewItem(BaseModel):
    state: Optional[Literal["assessable", "uncertain", "outside_image", "annotation_incorrect"]] = None
    marked_distal_tip: Optional[Dict[str, float]] = None  # {'x': float, 'y': float}
    notes: Optional[str] = ""
    placement_label: Optional[Literal["normal", "borderline", "malpositioned", "uncertain"]] = None


class CarinaReview(BaseModel):
    state: Optional[Literal["visible", "uncertain", "not_visible"]] = None
    coordinate: Optional[Dict[str, float]] = None
    notes: Optional[str] = ""


class UncatalogedCatheterItem(BaseModel):
    marked_distal_tip: Optional[Dict[str, float]] = None
    state: Optional[Literal["assessable", "uncertain", "outside_image", "annotation_incorrect"]] = "assessable"
    notes: Optional[str] = ""
    placement_label: Optional[Literal["normal", "borderline", "malpositioned", "uncertain"]] = None


class SaveStudyReviewRequest(BaseModel):
    study_id: str
    queue_name: str
    reviewer_id: str
    review_notes: Optional[str] = ""
    catheter_reviews: Dict[str, CatheterReviewItem]  # keyed by str(catheter_instance_id)
    carina_review: Optional[CarinaReview] = None
    uncataloged_catheters: Optional[List[UncatalogedCatheterItem]] = []
    is_study_complete: bool


# Queue loaders
def get_queue_study_ids(queue_name: str) -> List[str]:
    if queue_name == "positive_expansion_200":
        q_path = PROJECT_ROOT / "data" / "annotation_expansion_queue.json"
        if q_path.exists():
            with open(q_path, "r", encoding="utf-8") as f:
                return json.load(f).get("positive_study_ids", [])
    elif queue_name == "validation_v4_completion":
        q_csv = PROJECT_ROOT / "data" / "reviewed_distal_tips_v4_val_queue.csv"
        if q_csv.exists():
            df = pd.read_csv(q_csv)
            return sorted(df["study_id"].unique().tolist())
    elif queue_name == "provisional_pilot_25":
        # First 25 studies of positive expansion queue for consistent calibration
        q_path = PROJECT_ROOT / "data" / "annotation_expansion_queue.json"
        if q_path.exists():
            with open(q_path, "r", encoding="utf-8") as f:
                return json.load(f).get("positive_study_ids", [])[:25]
    elif queue_name == "carina_placement_pilot_50":
        q_path = PROJECT_ROOT / "data" / "curated_cvc_v1" / "carina_placement_pilot_50.json"
        if q_path.exists():
            with open(q_path, "r", encoding="utf-8") as f:
                return json.load(f).get("study_ids", [])
    return []


@app.get("/api/queues")
def list_queues():
    return [
        {
            "id": "provisional_pilot_25",
            "name": "Calibration Pilot Batch (25 Positive Studies)",
            "description": "Recommended starting milestone: review 25 studies to establish consistent rules before scaling.",
        },
        {
            "id": "positive_expansion_200",
            "name": "Full Positive Expansion Queue (200 Studies, 235 Inventory CVCs)",
            "description": "Complete review queue for 200 distinct patient studies.",
        },
        {
            "id": "validation_v4_completion",
            "name": "Validation Multi-CVC Completion Queue (25 Studies, 41 CVCs)",
            "description": "Completes remaining 16 unreviewed CVCs in validation to form V4-Val benchmark.",
        },
        {
            "id": "carina_placement_pilot_50",
            "name": "Carina + Placement Calibration (50 Training Studies)",
            "description": "Training-only, patient-separated pilot; review carina visibility/point and each CVC placement.",
        },
    ]


@app.get("/api/studies")
def list_studies(queue_name: str = Query("provisional_pilot_25"), reviewer_id: str = Query("")):
    if queue_name not in QUEUE_NAMES:
        raise HTTPException(status_code=400, detail="Unknown review queue")
    sids = get_queue_study_ids(queue_name)
    queue_review_dir = REVIEWS_DIR / queue_name
    queue_review_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for sid in sids:
        # Check if study has saved review
        rev_file = _review_file(queue_name, sid, reviewer_id) if reviewer_id else queue_review_dir / f"{sid}.json"
        is_reviewed = rev_file.exists()
        reviewer_id = ""
        reviewed_at = ""
        review_state = "pending"

        if is_reviewed:
            try:
                with open(rev_file, "r", encoding="utf-8") as f:
                    rdata = json.load(f)
                    reviewer_id = rdata.get("reviewer_id", "")
                    reviewed_at = rdata.get("review_timestamp", "")
                    review_state = "completed" if rdata.get("is_study_complete", False) else "in_progress"
            except Exception:
                pass

        # Query CVCs in study
        scvcs = INV_DF[(INV_DF["study_id"] == sid) & (INV_DF["device_type"] == "CVC")]
        patient_id = scvcs.iloc[0]["patient_id"] if len(scvcs) > 0 else "unknown"
        labels = scvcs["label"].tolist() if len(scvcs) > 0 else []

        results.append({
            "study_id": sid,
            "patient_id": patient_id,
            "cvc_count": len(scvcs),
            "labels": labels,
            "is_reviewed": is_reviewed,
            "review_state": review_state,
            "reviewer_id": reviewer_id,
            "reviewed_at": reviewed_at,
        })
    return {"queue_name": queue_name, "total_studies": len(results), "studies": results}


@app.get("/api/study/{study_id}")
def get_study_details(study_id: str, queue_name: str = Query("provisional_pilot_25"), reviewer_id: str = Query("")):
    if queue_name not in QUEUE_NAMES:
        raise HTTPException(status_code=400, detail="Unknown review queue")
    if study_id not in get_queue_study_ids(queue_name):
        raise HTTPException(status_code=404, detail="Study is not part of this review queue")
    # Study image metadata
    scvcs = INV_DF[(INV_DF["study_id"] == study_id) & (INV_DF["device_type"] == "CVC")]
    if len(scvcs) == 0:
        raise HTTPException(status_code=404, detail="Study not found in CVC inventory")

    first_row = scvcs.iloc[0]
    img_w = int(first_row["image_width"])
    img_h = int(first_row["image_height"])
    patient_id = str(first_row["patient_id"])
    dup_group = str(first_row.get("duplicate_group_id", "none"))

    # Other non-CVC lines for context if any
    other_lines = INV_DF[(INV_DF["study_id"] == study_id) & (INV_DF["device_type"] != "CVC")]

    catheters = []
    for _, r in scvcs.iterrows():
        cid = int(r["catheter_instance_id"])
        pts = r["polyline"]
        if isinstance(pts, str):
            import ast
            pts = ast.literal_eval(pts)

        catheters.append({
            "catheter_instance_id": cid,
            "instance_key": str(r["instance_key"]),
            "device_type": str(r["device_type"]),
            # Keep the placement category blinded during this new review pilot.
            "label": "CVC instance" if queue_name == DUAL_REVIEW_QUEUE else str(r["label"]),
            "polyline": pts,
            "polyline_length_px": float(r["polyline_length_px"]),
            "heuristic_tip": {"x": float(r["tip_x"]), "y": float(r["tip_y"])},
        })

    # Check existing saved review
    queue_review_dir = REVIEWS_DIR / queue_name
    rev_file = _review_file(queue_name, study_id, reviewer_id) if reviewer_id else queue_review_dir / f"{study_id}.json"
    existing_review = None
    if rev_file.exists():
        try:
            with open(rev_file, "r", encoding="utf-8") as f:
                existing_review = json.load(f)
        except Exception:
            pass

    return {
        "study_id": study_id,
        "patient_id": patient_id,
        "duplicate_group_id": dup_group,
        "image_width": img_w,
        "image_height": img_h,
        "catheters": catheters,
        "other_lines_count": len(other_lines),
        "existing_review": existing_review,
    }


@app.get("/api/image/{study_id}")
def get_study_image(study_id: str):
    if study_id not in set(INV_DF["study_id"].astype(str)):
        raise HTTPException(status_code=404, detail="Unknown study")
    img_path = RAW_IMAGES_DIR / f"{study_id}.jpg"
    if not img_path.exists():
        img_path = RAW_IMAGES_DIR / f"{study_id}.png"
    if not img_path.exists():
        raise HTTPException(status_code=404, detail=f"Image file not found for study {study_id}")
    return FileResponse(img_path, media_type="image/jpeg")


@app.post("/api/save_review")
def save_study_review(req: SaveStudyReviewRequest):
    if req.queue_name not in QUEUE_NAMES:
        raise HTTPException(status_code=400, detail="Unknown review queue")
    if req.study_id not in get_queue_study_ids(req.queue_name):
        raise HTTPException(status_code=400, detail="Study is not part of this review queue")
    if not req.reviewer_id.strip():
        raise HTTPException(status_code=422, detail="Reviewer ID is required for traceable annotation")
    cvc_rows = INV_DF[(INV_DF["study_id"] == req.study_id) & (INV_DF["device_type"] == "CVC")]
    if len(cvc_rows) == 0:
        raise HTTPException(status_code=400, detail="Study has no inventory-listed CVCs")
    dims = cvc_rows.iloc[0]
    for review in req.catheter_reviews.values():
        tip = review.marked_distal_tip
        if tip is not None and not _valid_native_point(tip, float(dims["image_width"]), float(dims["image_height"])):
            raise HTTPException(status_code=422, detail="Tip coordinate must be finite and inside the native image")
        if req.is_study_complete and (not review.state or (review.state == "assessable" and tip is None)):
            raise HTTPException(status_code=422, detail="Every CVC needs a review state; assessable tips need a coordinate")
    for item in req.uncataloged_catheters or []:
        if item.marked_distal_tip is not None and not _valid_native_point(item.marked_distal_tip, float(dims["image_width"]), float(dims["image_height"])):
            raise HTTPException(status_code=422, detail="Uncataloged CVC coordinate must be finite and inside the native image")
        if req.is_study_complete and (not item.state or (item.state == "assessable" and item.marked_distal_tip is None)):
            raise HTTPException(status_code=422, detail="Every uncataloged CVC needs a state and assessable tips need a coordinate")
    if req.carina_review and req.carina_review.coordinate is not None:
        if not _valid_native_point(req.carina_review.coordinate, float(dims["image_width"]), float(dims["image_height"])):
            raise HTTPException(status_code=422, detail="Carina coordinate must be finite and inside the native image")
    if req.queue_name == "carina_placement_pilot_50":
        expected = {str(int(x)) for x in cvc_rows["catheter_instance_id"].tolist()}
        if set(req.catheter_reviews) != expected:
            raise HTTPException(status_code=422, detail="Review must include every inventory-listed CVC")
        if req.is_study_complete and any(not item.placement_label for item in req.catheter_reviews.values()):
            raise HTTPException(status_code=422, detail="Assign a placement label to every CVC before saving a complete review")
        carina = req.carina_review
        if req.is_study_complete and (not carina or not carina.state):
            raise HTTPException(status_code=422, detail="Assign carina state: visible, uncertain, or not_visible")
        if req.is_study_complete and carina and carina.state == "visible" and not carina.coordinate:
            raise HTTPException(status_code=422, detail="Visible carina requires a coordinate")
        if carina and carina.state and carina.state != "visible" and carina.coordinate:
            raise HTTPException(status_code=422, detail="Only a visible carina may have a coordinate")
    queue_review_dir = REVIEWS_DIR / req.queue_name
    rev_file = _review_file(req.queue_name, req.study_id, req.reviewer_id)
    rev_file.parent.mkdir(parents=True, exist_ok=True)

    now_iso = datetime.now(timezone.utc).isoformat()
    record = {
        "study_id": req.study_id,
        "queue_name": req.queue_name,
        "reviewer_id": req.reviewer_id,
        "review_timestamp": now_iso,
        "review_notes": req.review_notes,
        "catheter_reviews": {k: v.model_dump() for k, v in req.catheter_reviews.items()},
        "carina_review": req.carina_review.model_dump() if req.carina_review else None,
        "uncataloged_catheters": [u.model_dump() for u in (req.uncataloged_catheters or [])],
        "is_study_complete": req.is_study_complete,
        "review_status": "provisional_engineering_review_awaiting_clinical_attestation",
    }

    # Preserve previous user reviews before replacing the current revision.
    if rev_file.exists():
        history_dir = rev_file.parent / "history"
        history_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(rev_file, history_dir / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + ".json"))
    temp_file = rev_file.with_suffix(".json.tmp")
    with open(temp_file, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)
    temp_file.replace(rev_file)

    # Update consolidated review dataframe
    consolidate_reviews(req.queue_name)

    return {"status": "success", "study_id": req.study_id, "saved_at": now_iso}


def consolidate_reviews(queue_name: str):
    queue_dir = REVIEWS_DIR / queue_name
    if not queue_dir.exists():
        return

    records = []
    carina_records = []
    for f in queue_dir.rglob("*.json"):
        if "history" in f.parts:
            continue
        try:
            with open(f, "r", encoding="utf-8") as jf:
                d = json.load(jf)
                sid = d["study_id"]
                reviewer = d.get("reviewer_id", "")
                ts = d.get("review_timestamp", "")
                carina = d.get("carina_review") or {}
                point = carina.get("coordinate") or {}
                carina_records.append({
                    "study_id": sid, "carina_state": carina.get("state"),
                    "carina_x": point.get("x"), "carina_y": point.get("y"),
                    "carina_notes": carina.get("notes", ""), "reviewer_id": reviewer,
                    "reviewed_at": ts, "review_status": d.get("review_status", "provisional_engineering_review_awaiting_clinical_attestation"),
                })
                for cid_str, crev in d.get("catheter_reviews", {}).items():
                    tip = crev.get("marked_distal_tip")
                    records.append({
                        "study_id": sid,
                        "catheter_instance_id": int(cid_str),
                        "review_state": crev.get("state"),
                        "placement_label": crev.get("placement_label"),
                        "reviewed_tip_x": tip["x"] if tip else None,
                        "reviewed_tip_y": tip["y"] if tip else None,
                        "reviewer_id": reviewer,
                        "reviewed_at": ts,
                        "notes": crev.get("notes", ""),
                        "review_status": d.get("review_status", "provisional_engineering_review_awaiting_clinical_attestation"),
                    })
        except Exception as exc:
            raise RuntimeError(f"Could not consolidate review file {f}: {exc}") from exc

    if len(records) > 0:
        df = pd.DataFrame(records)
        out_parquet = PROJECT_ROOT / "data" / f"reviews_{queue_name}.parquet"
        out_csv = PROJECT_ROOT / "data" / f"reviews_{queue_name}.csv"
        df.to_parquet(out_parquet, index=False)
        df.to_csv(out_csv, index=False)
    if carina_records:
        carina_df = pd.DataFrame(carina_records)
        carina_df.to_parquet(PROJECT_ROOT / "data" / f"carina_reviews_{queue_name}.parquet", index=False)
        carina_df.to_csv(PROJECT_ROOT / "data" / f"carina_reviews_{queue_name}.csv", index=False)


def _valid_native_point(point: Dict[str, float], width: float, height: float) -> bool:
    try:
        x, y = float(point["x"]), float(point["y"])
        return math.isfinite(x) and math.isfinite(y) and 0 <= x < width and 0 <= y < height
    except (KeyError, TypeError, ValueError):
        return False


# Mount static directory for the single-page web app
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=False)
