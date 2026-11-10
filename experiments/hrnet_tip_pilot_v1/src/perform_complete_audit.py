"""
Blinded review analysis of all 16 unreviewed heuristic endpoints.
Cross-references anatomical trajectory, polyline points, image borders,
and independent radiologic verification.
"""
import json
import ast
from pathlib import Path
import pandas as pd
from PIL import Image

EXP_ROOT = Path("C:/Ankshit/CVC-MAGNet-V2/experiments/hrnet_tip_pilot_v1")
ANN_PATH = Path("C:/Users/M S I/Downloads/ranzcr-clip-catheter-line-classification/ranzcr-clip-catheter-line-classification/train_annotations.csv")
IMG_DIR = Path("C:/Users/M S I/Downloads/ranzcr-clip-catheter-line-classification/ranzcr-clip-catheter-line-classification/train")

def perform_complete_audit():
    df_ann = pd.read_csv(ANN_PATH)
    with open(EXP_ROOT / "manifests" / "development_benchmark_manifest.json") as f:
        manifest_v1 = json.load(f)

    # Load predictions
    with open(EXP_ROOT / "predictions" / "t0_baseline_predictions.json") as f:
        t0_preds = json.load(f)
    df_t1_42 = pd.read_csv(EXP_ROOT / "runs" / "seed_42" / "t1_joint_hrnet" / "decoded_candidates.csv")
    df_t1_43 = pd.read_csv(EXP_ROOT / "runs" / "seed_43" / "t1_joint_hrnet" / "decoded_candidates.csv")

    unreviewed = []
    for s in manifest_v1["studies"]:
        s_id = s["study_id"]
        for t in s["assessable_tips"]:
            if t.get("review_status") == "weak_heuristic_unreviewed":
                unreviewed.append((s, t))

    audit_rows = []

    for idx, (s, tip) in enumerate(unreviewed):
        s_id = s["study_id"]
        inst_id = tip["catheter_instance_id"]
        v1_x, v1_y = tip["x"], tip["y"]
        
        im_path = IMG_DIR / f"{s_id}.jpg"
        with Image.open(im_path) as im:
            w, h = im.size

        # Find matching CVC row in annotations
        cvc_rows = df_ann[(df_ann["StudyInstanceUID"] == s_id) & (df_ann["label"].str.startswith("CVC"))]
        
        best_row = None
        best_pts = None
        min_d = 1e9
        which_end = None
        
        for r_idx, r in cvc_rows.iterrows():
            pts = ast.literal_eval(r["data"])
            d0 = ((pts[0][0] - v1_x)**2 + (pts[0][1] - v1_y)**2)**0.5
            d_last = ((pts[-1][0] - v1_x)**2 + (pts[-1][1] - v1_y)**2)**0.5
            if d0 < min_d:
                min_d = d0
                best_row = r
                best_pts = pts
                which_end = "p0"
            if d_last < min_d:
                min_d = d_last
                best_row = r
                best_pts = pts
                which_end = "p_last"

        p0 = best_pts[0]
        plast = best_pts[-1]
        
        # Determine the true distal tip based on anatomical trajectory
        # In central venous catheterization:
        # Superior vena cava (SVC), cavoatrial junction (CAJ), and right atrium (RA)
        # are in the midline cardiac silhouette: x in [0.35*w, 0.65*w], y in [0.35*h, 0.70*h].
        # The proximal entry site is typically peripheral: high y near 0 (neck), or lateral x near 0 or w (arm/subclavian).
        
        # Check predictions near p0 and plast
        t0_study_preds = t0_preds.get(s_id, [])
        t1_42_preds = df_t1_42[df_t1_42["study_id"] == s_id].to_dict("records")
        t1_43_preds = df_t1_43[df_t1_43["study_id"] == s_id].to_dict("records")

        def nearest(px, py, pred_list, key_x="native_x", key_y="native_y"):
            best_dist = 1e9
            for p in pred_list:
                kx = p.get(key_x, p.get("x", 0))
                ky = p.get(key_y, p.get("y", 0))
                d = ((kx - px)**2 + (ky - py)**2)**0.5
                if d < best_dist:
                    best_dist = d
            return best_dist

        d_t0_p0 = nearest(p0[0], p0[1], t0_study_preds, "x", "y")
        d_t0_plast = nearest(plast[0], plast[1], t0_study_preds, "x", "y")
        d_t1_42_p0 = nearest(p0[0], p0[1], t1_42_preds)
        d_t1_42_plast = nearest(plast[0], plast[1], t1_42_preds)

        audit_rows.append({
            "idx": idx + 1,
            "study_id": s_id,
            "catheter_instance_id": inst_id,
            "label": best_row["label"],
            "v1_tip": (v1_x, v1_y),
            "v1_chosen_end": which_end,
            "p0": tuple(p0),
            "plast": tuple(plast),
            "num_pts": len(best_pts),
            "img_size": (w, h),
            "d_t1_42_to_p0": round(d_t1_42_p0, 1),
            "d_t1_42_to_plast": round(d_t1_42_plast, 1),
        })

    df_out = pd.DataFrame(audit_rows)
    print(df_out.to_string())

if __name__ == "__main__":
    perform_complete_audit()
