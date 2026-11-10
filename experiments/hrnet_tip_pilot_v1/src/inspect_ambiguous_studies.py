"""
Print full point list and inspect anatomical trajectories of ambiguous catheters.
"""
import ast
import pandas as pd
from pathlib import Path

ANN_PATH = Path("C:/Users/M S I/Downloads/ranzcr-clip-catheter-line-classification/ranzcr-clip-catheter-line-classification/train_annotations.csv")

def inspect_details():
    df_ann = pd.read_csv(ANN_PATH)
    studies = [
        "1.2.826.0.1.3680043.8.498.12780732480359760065997067333764961045",
        "1.2.826.0.1.3680043.8.498.33128061425218955763725802634384514025",
        "1.2.826.0.1.3680043.8.498.78022247816932733427074863804392231471",
    ]
    for s_id in studies:
        print(f"\n==========================================")
        print(f"Study: {s_id}")
        rows = df_ann[(df_ann["StudyInstanceUID"] == s_id) & (df_ann["label"].str.startswith("CVC"))]
        for idx, r in rows.iterrows():
            pts = ast.literal_eval(r["data"])
            print(f"\nLabel: {r['label']} (num_pts={len(pts)})")
            for i, p in enumerate(pts):
                print(f"  [{i:2d}] x={p[0]:4d}, y={p[1]:4d}")

if __name__ == "__main__":
    inspect_details()
