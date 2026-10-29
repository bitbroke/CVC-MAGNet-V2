"""Inference pipeline and endpoint extraction for CVC-MAGNet-V2."""
from src.infer.endpoints import TipCandidate, extract_distal_tips, vendor_findpoint
from src.infer.pipeline import MAGNetInferencePipeline

__all__ = [
    "TipCandidate",
    "extract_distal_tips",
    "vendor_findpoint",
    "MAGNetInferencePipeline",
]
