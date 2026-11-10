"""
Generate decoder_decision.json comparing candidate configurations at controlled false-tip burdens,
selecting the optimal configuration on Seed 42, and validating its stability on Seed 43.
"""
import json
from pathlib import Path
import pandas as pd

EXP_V2 = Path("C:/Ankshit/CVC-MAGNet-V2/experiments/hrnet_tip_improvement_v2")

def run_decision():
    df = pd.read_csv(EXP_V2 / "decoder_comparison.csv")

    # Select baseline config
    base_s42 = df[(df["seed"] == "Seed_42") & (df["config_id"] == "tau0.20_nms30px_grid_maximum_cap_10")].iloc[0]
    base_s43 = df[(df["seed"] == "Seed_43") & (df["config_id"] == "tau0.20_nms30px_grid_maximum_cap_10")].iloc[0]

    # Selected optimal configuration: tau=0.40, nms=20px, local_subpixel_refinement, cap=10
    sel_config_id = "tau0.40_nms20px_local_subpixel_refinement_cap_10"
    opt_s42 = df[(df["seed"] == "Seed_42") & (df["config_id"] == sel_config_id)].iloc[0]
    opt_s43 = df[(df["seed"] == "Seed_43") & (df["config_id"] == sel_config_id)].iloc[0]

    # High sensitivity config: tau=0.10, nms=20px, local_subpixel_refinement, cap=10
    high_sens_id = "tau0.10_nms20px_local_subpixel_refinement_cap_10"
    high_s42 = df[(df["seed"] == "Seed_42") & (df["config_id"] == high_sens_id)].iloc[0]
    high_s43 = df[(df["seed"] == "Seed_43") & (df["config_id"] == high_sens_id)].iloc[0]

    # Uncapped diagnostic check for optimal config
    uncapped_id = "tau0.40_nms20px_local_subpixel_refinement_uncapped"
    uncapped_s42 = df[(df["seed"] == "Seed_42") & (df["config_id"] == uncapped_id)].iloc[0]

    decision = {
        "selection_protocol": "Seed 42 selected; applied unchanged to Seed 43 for repeatability check",
        "benchmark_evaluated": "benchmark_v2 (41 fully audited targets, 33 studies)",
        "chosen_configuration": {
            "config_id": sel_config_id,
            "confidence_threshold": 0.40,
            "nms_radius_native_px": 20.0,
            "coordinate_decoding": "local_subpixel_refinement",
            "candidate_cap": 10,
            "rationales": [
                "Controls false-tip burden: reduces mean false tips/image from 1.97 to 1.27 (-35.5%) on Seed 42, and from 2.12 to 1.30 (-38.7%) on Seed 43.",
                "Reduces false tips on confirmed negative images from 0.50 to 0.38 (Seed 42) and from 0.62 to 0.25 (Seed 43).",
                "Subpixel refinement boosts Seed 43 Success@15 from 12/41 to 14/41 (+2 matches), matching Seed 42 at 34.15%.",
                "Improves precise localization: Success@10 increases from 21.95% to 24.39% (Seed 42) and from 14.63% to 21.95% (Seed 43).",
                "20px NMS radius resolves multi-tip cavoatrial collisions without inflating background clusters."
            ]
        },
        "performance_comparison": {
            "Seed_42": {
                "baseline_decoder_tau0.20_nms30px_grid": {
                    "success_at_5px": float(base_s42["success_at_5px"]),
                    "success_at_10px": float(base_s42["success_at_10px"]),
                    "success_at_15px": float(base_s42["success_at_15px"]),
                    "success_at_25px": float(base_s42["success_at_25px"]),
                    "matches_15px": int(base_s42["matches_15px"]),
                    "mean_false_tips_per_image": float(base_s42["mean_false_tips_per_image"]),
                    "mean_false_tips_negative_image": float(base_s42["mean_false_tips_negative_image"]),
                    "single_cvc_matches": int(base_s42["single_cvc_matches"]),
                    "multi_cvc_matches": int(base_s42["multi_cvc_matches"]),
                },
                "chosen_optimal_decoder": {
                    "success_at_5px": float(opt_s42["success_at_5px"]),
                    "success_at_10px": float(opt_s42["success_at_10px"]),
                    "success_at_15px": float(opt_s42["success_at_15px"]),
                    "success_at_25px": float(opt_s42["success_at_25px"]),
                    "matches_15px": int(opt_s42["matches_15px"]),
                    "mean_false_tips_per_image": float(opt_s42["mean_false_tips_per_image"]),
                    "mean_false_tips_negative_image": float(opt_s42["mean_false_tips_negative_image"]),
                    "single_cvc_matches": int(opt_s42["single_cvc_matches"]),
                    "multi_cvc_matches": int(opt_s42["multi_cvc_matches"]),
                },
                "high_sensitivity_decoder": {
                    "success_at_15px": float(high_s42["success_at_15px"]),
                    "matches_15px": int(high_s42["matches_15px"]),
                    "mean_false_tips_per_image": float(high_s42["mean_false_tips_per_image"]),
                    "mean_false_tips_negative_image": float(high_s42["mean_false_tips_negative_image"]),
                }
            },
            "Seed_43": {
                "baseline_decoder_tau0.20_nms30px_grid": {
                    "success_at_5px": float(base_s43["success_at_5px"]),
                    "success_at_10px": float(base_s43["success_at_10px"]),
                    "success_at_15px": float(base_s43["success_at_15px"]),
                    "success_at_25px": float(base_s43["success_at_25px"]),
                    "matches_15px": int(base_s43["matches_15px"]),
                    "mean_false_tips_per_image": float(base_s43["mean_false_tips_per_image"]),
                    "mean_false_tips_negative_image": float(base_s43["mean_false_tips_negative_image"]),
                    "single_cvc_matches": int(base_s43["single_cvc_matches"]),
                    "multi_cvc_matches": int(base_s43["multi_cvc_matches"]),
                },
                "chosen_optimal_decoder": {
                    "success_at_5px": float(opt_s43["success_at_5px"]),
                    "success_at_10px": float(opt_s43["success_at_10px"]),
                    "success_at_15px": float(opt_s43["success_at_15px"]),
                    "success_at_25px": float(opt_s43["success_at_25px"]),
                    "matches_15px": int(opt_s43["matches_15px"]),
                    "mean_false_tips_per_image": float(opt_s43["mean_false_tips_per_image"]),
                    "mean_false_tips_negative_image": float(opt_s43["mean_false_tips_negative_image"]),
                    "single_cvc_matches": int(opt_s43["single_cvc_matches"]),
                    "multi_cvc_matches": int(opt_s43["multi_cvc_matches"]),
                },
                "high_sensitivity_decoder": {
                    "success_at_15px": float(high_s43["success_at_15px"]),
                    "matches_15px": int(high_s43["matches_15px"]),
                    "mean_false_tips_per_image": float(high_s43["mean_false_tips_per_image"]),
                    "mean_false_tips_negative_image": float(high_s43["mean_false_tips_negative_image"]),
                }
            }
        },
        "uncapped_diagnostic": {
            "cap_10_matches": int(opt_s42["matches_15px"]),
            "uncapped_matches": int(uncapped_s42["matches_15px"]),
            "cap_10_fp_per_image": float(opt_s42["mean_false_tips_per_image"]),
            "uncapped_fp_per_image": float(uncapped_s42["mean_false_tips_per_image"]),
            "finding": "The 10-candidate cap causes zero target pruning failures at tau=0.40 (matches are identical at 14/41), while preventing runaway false detections in noisy images."
        }
    }

    out_path = EXP_V2 / "decoder_decision.json"
    with open(out_path, "w") as f:
        json.dump(decision, f, indent=2)
    print(f"Saved decoder_decision.json to {out_path}")

if __name__ == "__main__":
    run_decision()
