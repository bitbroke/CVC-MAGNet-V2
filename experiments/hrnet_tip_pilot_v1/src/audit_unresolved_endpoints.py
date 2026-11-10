"""
Render visual crops and polyline trajectories for blinded review of the 16 heuristic endpoints.
"""
import json
import ast
from pathlib import Path
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

EXP_ROOT = Path("C:/Ankshit/CVC-MAGNet-V2/experiments/hrnet_tip_pilot_v1")
ANN_PATH = Path("C:/Users/M S I/Downloads/ranzcr-clip-catheter-line-classification/ranzcr-clip-catheter-line-classification/train_annotations.csv")
IMG_DIR = Path("C:/Users/M S I/Downloads/ranzcr-clip-catheter-line-classification/ranzcr-clip-catheter-line-classification/train")
OUT_DIR = EXP_ROOT / "reports" / "heuristic_review_diagnostics"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def generate_review_diagnostics():
    df_ann = pd.read_csv(ANN_PATH)
    with open(EXP_ROOT / "manifests" / "development_benchmark_manifest.json") as f:
        manifest_v1 = json.load(f)

    # Studies with unreviewed endpoints
    studies_with_unrev = []
    for s in manifest_v1["studies"]:
        unrev = [t for t in s["assessable_tips"] if t.get("review_status") == "weak_heuristic_unreviewed"]
        if unrev:
            studies_with_unrev.append(s)

    print(f"Generating review images for {len(studies_with_unrev)} studies...")

    for s in studies_with_unrev:
        s_id = s["study_id"]
        img_path = IMG_DIR / f"{s_id}.jpg"
        im = Image.open(img_path).convert("RGB")
        w, h = im.size
        draw = ImageDraw.Draw(im)

        # Draw all annotations for this study
        all_rows = df_ann[df_ann["StudyInstanceUID"] == s_id]
        
        print(f"\n=======================================================")
        print(f"Study: {s_id}")
        
        # Draw each line
        for r_idx, r in all_rows.iterrows():
            pts = ast.literal_eval(r["data"])
            label = r["label"]
            is_cvc = label.startswith("CVC")
            color = (0, 255, 0) if is_cvc else (128, 128, 128)
            
            # Draw line segments
            for i in range(len(pts) - 1):
                draw.line([tuple(pts[i]), tuple(pts[i+1])], fill=color, width=4 if is_cvc else 2)

            if is_cvc:
                p0 = pts[0]
                plast = pts[-1]
                # p0 in cyan
                draw.ellipse([p0[0]-15, p0[1]-15, p0[0]+15, p0[1]+15], outline=(0, 255, 255), width=4)
                # plast in magenta
                draw.ellipse([plast[0]-15, plast[1]-15, plast[0]+15, plast[1]+15], outline=(255, 0, 255), width=4)

        # Mark manifest tips
        for t in s["assessable_tips"]:
            tx, ty = t["x"], t["y"]
            status = t["review_status"]
            outline_col = (255, 255, 0) if status == "weak_heuristic_unreviewed" else (255, 0, 0)
            draw.rectangle([tx-25, ty-25, tx+25, ty+25], outline=outline_col, width=5)

        # Save thumbnail/resized version for rapid inspection
        im_thumb = im.resize((1024, int(1024 * h / w)), Image.Resampling.LANCZOS)
        out_thumb_path = OUT_DIR / f"{s_id[:16]}_review.jpg"
        im_thumb.save(out_thumb_path, quality=85)

        # Print detailed trajectory for CVC rows
        cvc_rows = all_rows[all_rows["label"].str.startswith("CVC")]
        for idx, r in cvc_rows.iterrows():
            pts = ast.literal_eval(r["data"])
            print(f"  [{r['label']}] ({len(pts)} pts):")
            print(f"    p0={pts[0]} -> p_mid={pts[len(pts)//2]} -> plast={pts[-1]}")
            # Check manifest match
            for t in s["assessable_tips"]:
                d0 = ((pts[0][0] - t['x'])**2 + (pts[0][1] - t['y'])**2)**0.5
                dlast = ((pts[-1][0] - t['x'])**2 + (pts[-1][1] - t['y'])**2)**0.5
                if d0 < 10 or dlast < 10:
                    matched_end = "p0" if d0 < 10 else "plast"
                    print(f"    Matched manifest tip (inst {t['catheter_instance_id']}): {matched_end} at ({t['x']}, {t['y']}) status={t['review_status']}")

    print("\nVisual review diagnostics generated in:", OUT_DIR)

if __name__ == "__main__":
    generate_review_diagnostics()
