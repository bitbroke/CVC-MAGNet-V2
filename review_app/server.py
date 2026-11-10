import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

PROJECT_ROOT = Path(r"C:\Ankshit\CVC-MAGNet-V2")
DATA_DIR = PROJECT_ROOT / "data"
QUEUE_FILE = DATA_DIR / "review_queue.json"
REVIEW_OUT_JSON = DATA_DIR / "reviewed_distal_tips_v3.json"
REVIEW_OUT_PARQUET = DATA_DIR / "reviewed_distal_tips_v3.parquet"
REVIEW_OUT_CSV = DATA_DIR / "reviewed_distal_tips_v3.csv"

# Load config for raw image dir
import yaml
with open(PROJECT_ROOT / "configs" / "pilot.yaml") as f:
    cfg = yaml.safe_load(f)
RAW_IMAGES_DIR = Path(cfg["data"]["raw_dir"]) / "train"

app = FastAPI(title="Distal Tip Endpoint Review Workstation")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load queue manifest into memory
if not QUEUE_FILE.exists():
    from scripts.build_review_queue import build_review_queue
    build_review_queue()

with open(QUEUE_FILE, "r") as f:
    QUEUE_DATA = json.load(f)

# Also load existing inventory for other catheters in same study
inv_df = pd.read_parquet(DATA_DIR / "annotation_inventory.parquet")
study_catheters_cache: Dict[str, List[Dict[str, Any]]] = {}
for sid, grp in inv_df.groupby("study_id"):
    lines = []
    for _, r in grp.iterrows():
        pts = json.loads(r["polyline"])
        lines.append({
            "cid": int(r["catheter_instance_id"]),
            "device_type": str(r["device_type"]),
            "label": str(r["label"]),
            "pts": pts
        })
    study_catheters_cache[sid] = lines

# Load existing reviews if present
REVIEWED_CACHE: Dict[str, Dict[str, Any]] = {}

# Seed from v2 reviewed parquet initially if v3 json doesn't exist
if not REVIEW_OUT_JSON.exists():
    v2_file = DATA_DIR / "reviewed_distal_tips_v2.parquet"
    if v2_file.exists():
        v2_df = pd.read_parquet(v2_file)
        for _, r in v2_df.iterrows():
            key = f"{r['study_id']}_{r['catheter_instance_id']}"
            REVIEWED_CACHE[key] = {
                "study_id": str(r["study_id"]),
                "catheter_instance_id": int(r["catheter_instance_id"]),
                "instance_key": key,
                "reviewed_endpoint_class": str(r["reviewed_endpoint_class"]),
                "reviewed_tip_x": float(r["reviewed_tip_x"]) if pd.notna(r["reviewed_tip_x"]) else None,
                "reviewed_tip_y": float(r["reviewed_tip_y"]) if pd.notna(r["reviewed_tip_y"]) else None,
                "concordant_with_heuristic": bool(r.get("concordant_with_heuristic", True)),
                "review_status": "engineering_audit_endpoint_review_v3",
                "reviewer_id": str(r.get("reviewer_id", "interactive_endpoint_review")),
                "audit_notes": str(r.get("audit_notes", "")),
                "audited_at": str(r.get("audited_at", datetime.now().isoformat())),
            }
else:
    with open(REVIEW_OUT_JSON, "r") as f:
        REVIEWED_CACHE = json.load(f)


def save_reviewed_cache():
    with open(REVIEW_OUT_JSON, "w") as f:
        json.dump(REVIEWED_CACHE, f, indent=2)
    
    if REVIEWED_CACHE:
        df = pd.DataFrame(list(REVIEWED_CACHE.values()))
        df.to_parquet(REVIEW_OUT_PARQUET, index=False)
        df.to_csv(REVIEW_OUT_CSV, index=False)


class DecisionRequest(BaseModel):
    study_id: str
    catheter_instance_id: int
    decision: str  # "A", "B", "unassessable", "neither"
    notes: Optional[str] = ""
    reviewer_id: Optional[str] = "expert_endpoint_review"


@app.get("/api/queue")
def get_queue(priority: Optional[int] = None, status: Optional[str] = None):
    """Return summary of all cases in queue."""
    items = []
    total_reviewed = 0
    total_a = 0
    total_b = 0
    total_unassessable = 0

    for c in QUEUE_DATA:
        key = c["instance_key"]
        rev = REVIEWED_CACHE.get(key)
        is_rev = rev is not None
        decision = rev.get("reviewed_endpoint_class") if is_rev else None

        if is_rev:
            total_reviewed += 1
            if decision == "A":
                total_a += 1
            elif decision == "B":
                total_b += 1
            else:
                total_unassessable += 1

        # Filtering
        if priority is not None and c["priority"] != priority:
            continue
        if status == "reviewed" and not is_rev:
            continue
        if status == "unreviewed" and is_rev:
            continue

        items.append({
            "queue_index": c["queue_index"],
            "study_id": c["study_id"],
            "catheter_instance_id": c["catheter_instance_id"],
            "instance_key": key,
            "split": c["split"],
            "device_label": c["device_label"],
            "priority": c["priority"],
            "priority_reason": c["priority_reason"],
            "heuristic_choice": c["heuristic_choice"],
            "diff_bdist": c["diff_bdist"],
            "is_reviewed": is_rev,
            "decision": decision,
        })

    return {
        "total_cases": len(QUEUE_DATA),
        "filtered_cases": len(items),
        "total_reviewed": total_reviewed,
        "total_a": total_a,
        "total_b": total_b,
        "total_unassessable": total_unassessable,
        "cases": items
    }


@app.get("/api/case/{queue_idx}")
def get_case(queue_idx: int):
    """Return complete case payload by queue index (1-based)."""
    if queue_idx < 1 or queue_idx > len(QUEUE_DATA):
        raise HTTPException(status_code=404, detail="Case index out of range")
    
    c = QUEUE_DATA[queue_idx - 1]
    sid = c["study_id"]
    key = c["instance_key"]
    
    rev = REVIEWED_CACHE.get(key)

    # All catheters in this study for background context
    other_lines = []
    for line in study_catheters_cache.get(sid, []):
        is_current = (line["cid"] == c["catheter_instance_id"])
        other_lines.append({
            "cid": line["cid"],
            "device_type": line["device_type"],
            "label": line["label"],
            "is_current": is_current,
            "pts": line["pts"]
        })

    return {
        "case": c,
        "review": rev,
        "all_study_catheters": other_lines,
        "prev_index": queue_idx - 1 if queue_idx > 1 else None,
        "next_index": queue_idx + 1 if queue_idx < len(QUEUE_DATA) else None,
    }


@app.get("/api/image/{study_id}")
def get_image(study_id: str):
    """Stream native radiograph JPEG."""
    img_path = RAW_IMAGES_DIR / f"{study_id}.jpg"
    if not img_path.exists():
        raise HTTPException(status_code=404, detail="Image file not found on disk")
    return FileResponse(img_path, media_type="image/jpeg")


@app.post("/api/decision")
def post_decision(req: DecisionRequest):
    """Record endpoint review decision."""
    key = f"{req.study_id}_{req.catheter_instance_id}"
    
    # Find case in queue data
    case_item = next((c for c in QUEUE_DATA if c["instance_key"] == key), None)
    if not case_item:
        raise HTTPException(status_code=404, detail="Case not found in review queue")

    dec = req.decision.upper()
    if dec in ("A", "ENDPOINT_A"):
        tip_x, tip_y = case_item["endpoint_a"]
        endpoint_class = "A"
    elif dec in ("B", "ENDPOINT_B"):
        tip_x, tip_y = case_item["endpoint_b"]
        endpoint_class = "B"
    else:
        tip_x, tip_y = None, None
        endpoint_class = "unassessable"

    heur_choice = case_item["heuristic_choice"]
    concordant = (endpoint_class == heur_choice) if endpoint_class in ("A", "B") else False

    REVIEWED_CACHE[key] = {
        "study_id": req.study_id,
        "catheter_instance_id": req.catheter_instance_id,
        "instance_key": key,
        "device_type": "CVC",
        "split": case_item["split"],
        "reviewed_endpoint_class": endpoint_class,
        "reviewed_tip_x": tip_x,
        "reviewed_tip_y": tip_y,
        "original_heuristic_tip_x": case_item["endpoint_a"][0] if heur_choice == "A" else case_item["endpoint_b"][0],
        "original_heuristic_tip_y": case_item["endpoint_a"][1] if heur_choice == "A" else case_item["endpoint_b"][1],
        "original_heuristic_index": 0 if heur_choice == "A" else -1,
        "concordant_with_heuristic": concordant,
        "review_status": "engineering_audit_endpoint_review_v3",
        "reviewer_id": req.reviewer_id,
        "audit_notes": req.notes or "",
        "audited_at": datetime.now().isoformat(),
    }

    save_reviewed_cache()

    return {
        "success": True,
        "instance_key": key,
        "decision": endpoint_class,
        "tip_coord": [tip_x, tip_y],
        "concordant_with_heuristic": concordant,
        "total_reviewed": len(REVIEWED_CACHE)
    }


# Static assets
STATIC_DIR = PROJECT_ROOT / "review_app" / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", response_class=HTMLResponse)
def get_index():
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        return HTMLResponse("<h3>Review App Static Files Initializing...</h3>")
    return FileResponse(index_file)

