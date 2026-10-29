"""Image-only CVC Placement Baseline Network using EfficientNet-B0.

Outputs independent logits for:
1. CVC - Normal
2. CVC - Borderline
3. CVC - Abnormal

Uses multi-label binary cross-entropy (BCEWithLogitsLoss) formulation.
"""

from typing import Dict, Tuple

import timm
import torch
import torch.nn as nn


class CVCPlacementBaselineNet(nn.Module):
    """EfficientNet-B0 multi-label image-level CVC placement classifier."""

    def __init__(
        self,
        backbone_name: str = "tf_efficientnet_b0.ns_jft_in1k",
        pretrained: bool = True,
        num_classes: int = 3,
        dropout_rate: float = 0.25,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,  # pooled features
            drop_rate=dropout_rate,
        )
        in_features = self.backbone.num_features  # 1280 for EfficientNet-B0

        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor [B, 3, H, W]

        Returns:
            logits: Unnormalized logits [B, 3] for [Normal, Borderline, Abnormal]
        """
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits

    def predict_probabilities(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with sigmoid activation."""
        logits = self.forward(x)
        return torch.sigmoid(logits)


if __name__ == "__main__":
    net = CVCPlacementBaselineNet(pretrained=False)
    dummy_x = torch.randn(2, 3, 512, 512)
    out = net(dummy_x)
    probs = net.predict_probabilities(dummy_x)
    print("Logits shape:", out.shape)
    print("Probs shape:", probs.shape)
    assert out.shape == (2, 3)
    assert probs.shape == (2, 3)
    print("CVCPlacementBaselineNet smoke test PASSED.")
