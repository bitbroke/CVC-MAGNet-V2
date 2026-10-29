"""Complete two-stage MAG-Net inference pipeline with native-coordinate reconstruction.

Pipeline Path:
Radiograph → Coarse Segmentation → Predicted Crop Selection → Fine Segmentation → Distal Tip Candidates → Native Coordinates

Guarantees:
- Validation crop selection strictly uses coarse predictions, never ground-truth annotations.
- Explicit failure states recorded when no crop or catheter is detected.
- Multiple candidates preserved where co-occurring catheters or crossing branches exist.
- Complete inversion back to native radiograph pixels for Success@15 evaluation.
"""
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch

from src.data.transforms import FineCropTransform, LetterboxTransform, TransformMetadata
from src.infer.endpoints import TipCandidate, extract_distal_tips, vendor_findpoint
from src.models.coarse import CoarseSegNet
from src.models.fine import FineSegNet


class MAGNetInferencePipeline:
    """Full inference pipeline executing coarse detection, predicted cropping, and fine refinement."""

    def __init__(
        self,
        coarse_model: CoarseSegNet,
        fine_model: Optional[FineSegNet] = None,
        device: Union[str, torch.device] = "cuda",
        coarse_threshold: float = 0.3,
        fine_threshold: float = 0.5,
        target_size: Tuple[int, int] = (512, 512),
        crop_padding_px: int = 32,
    ):
        self.device = torch.device(device) if isinstance(device, str) else device
        self.coarse_model = coarse_model.to(self.device).eval()
        self.fine_model = fine_model.to(self.device).eval() if fine_model is not None else None
        self.coarse_threshold = coarse_threshold
        self.fine_threshold = fine_threshold
        self.target_size = target_size
        self.crop_padding_px = crop_padding_px

        self.letterbox = LetterboxTransform(target_size=target_size, clahe_enabled=True)
        self.crop_transform = FineCropTransform(target_size=target_size, padding_px=crop_padding_px)

    @torch.no_grad()
    def predict_image(
        self,
        image_input: Union[str, Path, np.ndarray],
        study_id: str = "UNKNOWN",
    ) -> Dict[str, Any]:
        """Run complete inference path on a single chest radiograph.
        
        Args:
            image_input: File path or (H, W) numpy array of native radiograph.
            study_id: Identifier for record tracking.
            
        Returns:
            result_dict: Complete evaluable prediction output.
        """
        t0 = time.time()

        # 1. Load native radiograph
        if isinstance(image_input, (str, Path)):
            img_path = Path(image_input)
            native_img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if native_img is None:
                return {
                    "study_id": study_id,
                    "status": "failed_image_read",
                    "candidates": [],
                    "error": f"Could not read {img_path}",
                }
        else:
            native_img = image_input

        orig_h, orig_w = native_img.shape[:2]

        # 2. Stage 1: Coarse Global Segmentation
        lb_img, coarse_meta = self.letterbox.apply_image(native_img)
        x_coarse = (
            torch.from_numpy(lb_img)
            .unsqueeze(0)
            .unsqueeze(0)
            .float()
            .to(self.device)
            / 255.0
        )

        coarse_logits = self.coarse_model(x_coarse)
        coarse_probs = torch.sigmoid(coarse_logits).squeeze().cpu().numpy()
        coarse_max_p = float(coarse_probs.max())

        # 3. Stage 2: Predicted Crop Selection (Strictly based on coarse predictions)
        choice = np.where(coarse_probs > self.coarse_threshold)
        
        if len(choice[0]) == 0 or coarse_max_p < self.coarse_threshold:
            # Explicit failure state: No catheter region found by coarse stage
            total_time_ms = (time.time() - t0) * 1000.0
            return {
                "study_id": study_id,
                "status": "no_coarse_prediction",
                "native_width": orig_w,
                "native_height": orig_h,
                "coarse_max_prob": coarse_max_p,
                "crop_box_native": None,
                "candidates": [],
                "primary_tip": None,
                "vendor_tip": None,
                "inference_time_ms": total_time_ms,
            }

        # Bounding box in letterbox space
        lb_y0, lb_y1 = int(choice[0].min()), int(choice[0].max())
        lb_x0, lb_x1 = int(choice[1].min()), int(choice[1].max())

        # Map bounding box back to native coordinates
        nat_x0, nat_y0 = coarse_meta.model_to_native(float(lb_x0), float(lb_y0))
        nat_x1, nat_y1 = coarse_meta.model_to_native(float(lb_x1), float(lb_y1))
        
        bbox_native = (
            max(0, int(np.floor(min(nat_x0, nat_x1)))),
            max(0, int(np.floor(min(nat_y0, nat_y1)))),
            min(orig_w, int(np.ceil(max(nat_x0, nat_x1)))),
            min(orig_h, int(np.ceil(max(nat_y0, nat_y1)))),
        )

        # 4. Stage 3: Fine Segmentation (or fallback to coarse if no fine model)
        # 4. Stage 3: Fine Segmentation (or fallback to coarse if no fine model)
        if self.fine_model is not None:
            # Unpad letterboxed coarse prediction and resize valid content to native radiograph dimensions
            pad_top = coarse_meta.pad_top
            pad_left = coarse_meta.pad_left
            new_h = coarse_meta.new_h
            new_w = coarse_meta.new_w

            # Extract valid unpadded content from coarse letterbox map
            valid_coarse_prob = coarse_probs[pad_top : pad_top + new_h, pad_left : pad_left + new_w]

            # Resize valid unpadded coarse probability map directly to native image dimensions (orig_w, orig_h)
            full_coarse_prob = cv2.resize(valid_coarse_prob, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

            # Co-extract 2-channel crop from native image and full coarse mask
            img_crop, coarse_crop, _, fine_meta = self.crop_transform.extract_crop(
                native_img, full_coarse_prob, bbox_native
            )

            # Stack into (1, 2, 512, 512) tensor
            x_fine_0 = torch.from_numpy(img_crop).float() / 255.0
            x_fine_1 = torch.from_numpy(coarse_crop).float()
            x_fine = torch.stack([x_fine_0, x_fine_1], dim=0).unsqueeze(0).to(self.device)

            fine_logits, fine_att = self.fine_model(x_fine)
            eval_probs = torch.sigmoid(fine_logits).squeeze().cpu().numpy()
            eval_meta = fine_meta
            stage_source = "fine_stage"
        else:
            # Coarse-only evaluation mode
            eval_probs = coarse_probs
            eval_meta = coarse_meta
            stage_source = "coarse_stage"

        # 5. Stage 4: Distal Tip Extraction
        raw_candidates, diag_info = extract_distal_tips(
            eval_probs, threshold=self.fine_threshold, min_component_area=40
        )

        # 6. Stage 5: Project Candidates back to Native Radiograph Pixels
        # Preserve out-of-image coordinates as invalid instead of silently relocating them onto image borders
        native_candidates: List[Dict[str, Any]] = []
        for cand in raw_candidates:
            nat_x, nat_y = eval_meta.model_to_native(cand.x, cand.y)
            is_in_bounds = bool((0.0 <= nat_x < orig_w) and (0.0 <= nat_y < orig_h))

            native_candidates.append({
                "native_x": round(float(nat_x), 2),
                "native_y": round(float(nat_y), 2),
                "confidence": round(cand.confidence, 4),
                "border_distance": round(cand.border_distance, 1),
                "is_primary": cand.is_primary,
                "component_id": cand.component_id,
                "source_method": cand.source_method,
                "stage": stage_source,
                "in_image": is_in_bounds,
                "is_valid": is_in_bounds,
            })

        # Comparative audit: vendor findpoint heuristic
        vx, vy, v_status = vendor_findpoint(eval_probs)
        vendor_native = None
        if vx is not None and vy is not None:
            vn_x, vn_y = eval_meta.model_to_native(vx, vy)
            v_in_bounds = bool((0.0 <= vn_x < orig_w) and (0.0 <= vn_y < orig_h))
            vendor_native = {
                "native_x": round(float(vn_x), 2),
                "native_y": round(float(vn_y), 2),
                "in_image": v_in_bounds,
                "is_valid": v_in_bounds,
                "status": v_status,
            }

        total_time_ms = (time.time() - t0) * 1000.0

        return {
            "study_id": study_id,
            "status": "success" if len(native_candidates) > 0 else "no_tips_found",
            "native_width": orig_w,
            "native_height": orig_h,
            "coarse_max_prob": coarse_max_p,
            "crop_box_native": bbox_native,
            "candidates": native_candidates,
            "primary_tip": native_candidates[0] if len(native_candidates) > 0 else None,
            "vendor_tip": vendor_native,
            "inference_time_ms": round(total_time_ms, 2),
            "stage_source": stage_source,
            "diag_info": diag_info,
        }
