"""Aspect-ratio-preserving geometric transforms, CLAHE contrast enhancement, and coordinate tracking.

Ensures every model input is accompanied by complete affine inversion metadata
so predictions can be accurately projected back to native radiograph pixel space.
"""
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union
import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
import numpy as np
import torch


@dataclass
class TransformMetadata:
    """Records the exact geometric mapping between native and model coordinate spaces."""
    scale: float
    pad_x: float
    pad_y: float
    orig_w: int
    orig_h: int
    target_w: int
    target_h: int
    crop_x0: float = 0.0
    crop_y0: float = 0.0
    new_w: Optional[int] = None
    new_h: Optional[int] = None
    pad_left: Optional[int] = None
    pad_top: Optional[int] = None

    def __post_init__(self):
        if self.new_w is None:
            self.new_w = int(round(self.orig_w * self.scale))
        if self.new_h is None:
            self.new_h = int(round(self.orig_h * self.scale))
        if self.pad_left is None:
            self.pad_left = int(np.floor(self.pad_x))
        if self.pad_top is None:
            self.pad_top = int(np.floor(self.pad_y))

    def to_dict(self) -> Dict[str, Union[float, int]]:
        return {
            "scale": self.scale,
            "pad_x": self.pad_x,
            "pad_y": self.pad_y,
            "orig_w": self.orig_w,
            "orig_h": self.orig_h,
            "target_w": self.target_w,
            "target_h": self.target_h,
            "crop_x0": self.crop_x0,
            "crop_y0": self.crop_y0,
            "new_w": self.new_w,
            "new_h": self.new_h,
            "pad_left": self.pad_left,
            "pad_top": self.pad_top,
        }

    def native_to_model(self, x: float, y: float) -> Tuple[float, float]:
        """Map a native image pixel coordinate to model input space using OpenCV pixel-center convention."""
        rel_x = float(x) - self.crop_x0
        rel_y = float(y) - self.crop_y0

        scale_x = float(self.new_w) / float(self.orig_w)
        scale_y = float(self.new_h) / float(self.orig_h)

        model_x = (rel_x + 0.5) * scale_x - 0.5 + float(self.pad_left)
        model_y = (rel_y + 0.5) * scale_y - 0.5 + float(self.pad_top)
        return float(model_x), float(model_y)

    def model_to_native(self, x_model: float, y_model: float) -> Tuple[float, float]:
        """Project a model prediction coordinate back to native radiograph pixel space."""
        scale_x = float(self.new_w) / float(self.orig_w)
        scale_y = float(self.new_h) / float(self.orig_h)

        rel_x = (float(x_model) - float(self.pad_left) + 0.5) / scale_x - 0.5
        rel_y = (float(y_model) - float(self.pad_top) + 0.5) / scale_y - 0.5

        native_x = rel_x + self.crop_x0
        native_y = rel_y + self.crop_y0
        return float(native_x), float(native_y)


class LetterboxTransform:
    """Isotropic letterbox resize with symmetric padding, preserving aspect ratio."""

    def __init__(
        self,
        target_size: Tuple[int, int] = (512, 512),
        fill_value: int = 0,
        clahe_enabled: bool = True,
        clahe_clip_limit: float = 2.0,
        clahe_tile_grid: Tuple[int, int] = (8, 8),
    ):
        self.target_h, self.target_w = target_size
        self.fill_value = fill_value
        self.clahe_enabled = clahe_enabled
        if clahe_enabled:
            self.clahe = cv2.createCLAHE(
                clipLimit=clahe_clip_limit, tileGridSize=clahe_tile_grid
            )
        else:
            self.clahe = None

    def compute_params(self, orig_h: int, orig_w: int) -> TransformMetadata:
        scale = min(self.target_w / orig_w, self.target_h / orig_h)
        new_w = int(round(orig_w * scale))
        new_h = int(round(orig_h * scale))
        pad_x = (self.target_w - new_w) / 2.0
        pad_y = (self.target_h - new_h) / 2.0
        pad_left = int(np.floor(pad_x))
        pad_top = int(np.floor(pad_y))
        return TransformMetadata(
            scale=scale,
            pad_x=pad_x,
            pad_y=pad_y,
            orig_w=orig_w,
            orig_h=orig_h,
            target_w=self.target_w,
            target_h=self.target_h,
            new_w=new_w,
            new_h=new_h,
            pad_left=pad_left,
            pad_top=pad_top,
        )

    def apply_image(
        self, image: np.ndarray, meta: Optional[TransformMetadata] = None
    ) -> Tuple[np.ndarray, TransformMetadata]:
        """Apply CLAHE and isotropic letterbox resizing to an image.
        
        Args:
            image: (H, W) or (H, W, C) uint8 or float32 image.
        """
        orig_h, orig_w = image.shape[:2]
        if meta is None:
            meta = self.compute_params(orig_h, orig_w)

        # CLAHE preprocessing on grayscale uint8
        if self.clahe_enabled and image.dtype == np.uint8:
            if image.ndim == 2:
                processed = self.clahe.apply(image)
            else:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                processed = self.clahe.apply(gray)
        else:
            processed = image

        new_w = int(round(orig_w * meta.scale))
        new_h = int(round(orig_h * meta.scale))
        resized = cv2.resize(processed, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Pad symmetrically
        pad_left = int(np.floor(meta.pad_x))
        pad_right = int(self.target_w - new_w - pad_left)
        pad_top = int(np.floor(meta.pad_y))
        pad_bottom = int(self.target_h - new_h - pad_top)

        canvas = np.full(
            (self.target_h, self.target_w), self.fill_value, dtype=processed.dtype
        )
        canvas[pad_top : pad_top + new_h, pad_left : pad_left + new_w] = resized
        return canvas, meta

    def apply_mask(
        self, mask: np.ndarray, meta: TransformMetadata, interpolation: int = cv2.INTER_NEAREST
    ) -> np.ndarray:
        """Apply the identical geometric transform to a mask."""
        orig_h, orig_w = mask.shape[:2]
        new_w = int(round(orig_w * meta.scale))
        new_h = int(round(orig_h * meta.scale))

        resized = cv2.resize(mask, (new_w, new_h), interpolation=interpolation)
        pad_left = int(np.floor(meta.pad_x))
        pad_top = int(np.floor(meta.pad_y))

        canvas = np.zeros((self.target_h, self.target_w), dtype=mask.dtype)
        canvas[pad_top : pad_top + new_h, pad_left : pad_left + new_w] = resized
        return canvas


class FineCropTransform:
    """Extracts an ROI bounding box crop with padding and isotropic letterboxing."""

    def __init__(
        self,
        target_size: Tuple[int, int] = (512, 512),
        padding_px: int = 32,
    ):
        self.target_h, self.target_w = target_size
        self.padding_px = padding_px

    def extract_crop(
        self,
        image: np.ndarray,
        coarse_mask: np.ndarray,
        bbox: Tuple[int, int, int, int],  # [x_min, y_min, x_max, y_max] in native coordinates
        gt_mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], TransformMetadata]:
        """Extract padded crop from image, coarse mask, and optional ground truth mask.
        
        Args:
            image: (H, W) radiograph.
            coarse_mask: (H, W) coarse segmentation map.
            bbox: [x0, y0, x1, y1] in native pixels.
        """
        orig_h, orig_w = image.shape[:2]
        x0, y0, x1, y1 = bbox
        
        # Add padding and clamp
        cx0 = max(0, int(np.floor(x0 - self.padding_px)))
        cy0 = max(0, int(np.floor(y0 - self.padding_px)))
        cx1 = min(orig_w, int(np.ceil(x1 + self.padding_px)))
        cy1 = min(orig_h, int(np.ceil(y1 + self.padding_px)))

        crop_w = max(1, cx1 - cx0)
        crop_h = max(1, cy1 - cy0)

        img_crop = image[cy0:cy1, cx0:cx1]
        coarse_crop = coarse_mask[cy0:cy1, cx0:cx1]

        scale = min(self.target_w / crop_w, self.target_h / crop_h)
        new_w = int(round(crop_w * scale))
        new_h = int(round(crop_h * scale))
        pad_x = (self.target_w - new_w) / 2.0
        pad_y = (self.target_h - new_h) / 2.0
        pad_left = int(np.floor(pad_x))
        pad_top = int(np.floor(pad_y))

        meta = TransformMetadata(
            scale=scale,
            pad_x=pad_x,
            pad_y=pad_y,
            orig_w=crop_w,
            orig_h=crop_h,
            target_w=self.target_w,
            target_h=self.target_h,
            crop_x0=float(cx0),
            crop_y0=float(cy0),
            new_w=new_w,
            new_h=new_h,
            pad_left=pad_left,
            pad_top=pad_top,
        )

        # Letterbox image crop
        img_resized = cv2.resize(img_crop, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        img_canvas = np.zeros((self.target_h, self.target_w), dtype=image.dtype)
        img_canvas[pad_top : pad_top + new_h, pad_left : pad_left + new_w] = img_resized

        # Letterbox coarse mask crop
        coarse_resized = cv2.resize(coarse_crop, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        coarse_canvas = np.zeros((self.target_h, self.target_w), dtype=coarse_mask.dtype)
        coarse_canvas[pad_top : pad_top + new_h, pad_left : pad_left + new_w] = coarse_resized

        gt_canvas = None
        if gt_mask is not None:
            gt_crop = gt_mask[cy0:cy1, cx0:cx1]
            gt_resized = cv2.resize(gt_crop, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            gt_canvas = np.zeros((self.target_h, self.target_w), dtype=gt_mask.dtype)
            gt_canvas[pad_top : pad_top + new_h, pad_left : pad_left + new_w] = gt_resized

        return img_canvas, coarse_canvas, gt_canvas, meta
