import torch
import json
import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path
import sys

sys.path.insert(0, "C:/Ankshit/CVC-MAGNet-V2")
sys.path.insert(0, "C:/Ankshit/CVC-MAGNet-V2/experiments/hrnet_tip_improvement_v2")

from experiments.hrnet_tip_pilot_v1.src.models.hrnet_joint_tip import HRNetJointTipNet
from experiments.hrnet_tip_pilot_v1.src.geometry import CoordinateTransform
from src.data.transforms import LetterboxTransform
from experiments.hrnet_tip_improvement_v2.src.run_decoder_comparison import (
    evaluate_grid,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bench_df = pd.read_csv("C:/Ankshit/CVC-MAGNet-V2/experiments/hrnet_tip_improvement_v2/benchmark_v2.csv")

with open("C:/Ankshit/CVC-MAGNet-V2/experiments/hrnet_tip_pilot_v1/manifests/development_benchmark_manifest.json") as f:
    manifest = json.load(f)
studies = manifest["studies"]

images_dir = Path(r"C:\Users\M S I\Downloads\ranzcr-clip-catheter-line-classification\ranzcr-clip-catheter-line-classification\train")
transform = LetterboxTransform(target_size=(768, 768), clahe_enabled=True)
mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

def eval_ckpt(ckpt_path, name):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = HRNetJointTipNet(pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    heatmaps_dict = {}
    with torch.no_grad():
        for s in studies:
            sid = s["study_id"]
            img_path = images_dir / f"{sid}.jpg"
            pil_im = Image.open(img_path).convert("L")
            im_np = np.array(pil_im)
            orig_h, orig_w = im_np.shape
            coord_tf = CoordinateTransform.from_native_dims(orig_w, orig_h, target_size=768, stride=4)
            lb_img, meta = transform.apply_image(im_np)
            raw_t = torch.from_numpy(lb_img).unsqueeze(0).unsqueeze(0).to(device).float() / 255.0
            inp = (raw_t.repeat(1, 3, 1, 1) - mean) / std
            out = model(inp)
            tip_probs = out["tip_probs"].squeeze().cpu().numpy().astype(np.float32)
            heatmaps_dict[sid] = {
                "heatmap": tip_probs,
                "native_w": orig_w,
                "native_h": orig_h,
                "pad_left": coord_tf.padding_left,
                "pad_top": coord_tf.padding_top,
                "resized_w": coord_tf.resized_width,
                "resized_h": coord_tf.resized_height,
                "stride": 4,
            }

    rows, _ = evaluate_grid(
        heatmaps_by_study=heatmaps_dict,
        bm_v2_df=bench_df,
        seed_name=name,
        thresholds=[0.40],
        nms_radii=[20.0],
        decoding_modes=["local_subpixel_refinement"],
        cap_values=[10],
    )
    return rows[0]

if __name__ == "__main__":
    checkpoints = [
        ("update_0000", "C:/Ankshit/CVC-MAGNet-V2/experiments/hrnet_tip_pilot_v1/runs/seed_42/t1_joint_hrnet/checkpoints/update_0000.pt"),
        ("update_0250", "C:/Ankshit/CVC-MAGNet-V2/experiments/hrnet_tip_pilot_v1/runs/seed_42/t1_joint_hrnet/checkpoints/update_0250.pt"),
        ("update_0500", "C:/Ankshit/CVC-MAGNet-V2/experiments/hrnet_tip_pilot_v1/runs/seed_42/t1_joint_hrnet/checkpoints/update_0500.pt"),
        ("update_1000", "C:/Ankshit/CVC-MAGNet-V2/experiments/hrnet_tip_pilot_v1/runs/seed_42/t1_joint_hrnet/checkpoints/final_model.pt"),
    ]
    for name, p in checkpoints:
        m = eval_ckpt(p, name)
        print(f"=== {name} ===", flush=True)
        print(f"Success@5:  {m['success_at_5px']*100:.2f}%", flush=True)
        print(f"Success@10: {m['success_at_10px']*100:.2f}%", flush=True)
        print(f"Success@15: {m['success_at_15px']*100:.2f}% ({m['matches_15px']}/41)", flush=True)
        print(f"Success@25: {m['success_at_25px']*100:.2f}%", flush=True)
        print(f"Mean FP/img: {m['mean_false_tips_per_image']:.3f} | Neg FP/img: {m['mean_false_tips_negative_image']:.3f}", flush=True)
        print(f"Single-CVC Matches: {m['single_cvc_matches']}/11 | Multi-CVC Matches: {m['multi_cvc_matches']}/30", flush=True)
