"""FullSpatialNet: Pure 2D Spatial EfficientNet-B0 Placement Classifier.

Inputs: [B, 3, 512, 512]
- Ch0: Full CLAHE Radiograph
- Ch1: T2 Continuous Segmentation Ribbon
- Ch2: Anatomical Coordinate Map (Carina Gaussian + CAJ Landing Strip + Tip Gaussian)

Direct 2D Conv Backbone -> GAP -> Linear(1280, 3) -> Logits.
No attached tabular MLP branch (prevents multimodal gradient starvation).
"""

import timm
import torch
import torch.nn as nn


class FullSpatialNet(nn.Module):
    """Pure 2D EfficientNet-B0 classifier for spatial CVC placement."""

    def __init__(
        self,
        backbone_name: str = "tf_efficientnet_b0.ns_jft_in1k",
        pretrained: bool = True,
        num_classes: int = 3,
        dropout_rate: float = 0.25,
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.num_classes = num_classes

        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            in_chans=3,
            num_classes=0,  # return pooled feature vector [B, 1280]
            drop_rate=dropout_rate,
        )
        in_features = self.backbone.num_features

        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: [B, 3, 512, 512] tensor

        Returns:
            logits: [B, 3] unnormalized logits for [Normal, Borderline, Abnormal]
        """
        feats = self.backbone(x)
        logits = self.classifier(feats)
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Sigmoid probability prediction."""
        logits = self.forward(x)
        return torch.sigmoid(logits)

