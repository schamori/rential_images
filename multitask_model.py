"""
Multi-task wrapper around ConvNextV2ForImageClassification.

Architecture (hard parameter sharing):
    Shared ConvNeXt V2 encoder  (pretrained, fine-tuned)
        ├── Classification head  → 5 classes   (main task)
        ├── Age regression head  → 1 neuron    (auxiliary)
        └── Centre prediction    → 1 neuron    (auxiliary)

Usage:
    model = MultiTaskConvNeXt(cfg)
    cls_logits, age_pred, centre_pred = model(images)
"""
import torch
import torch.nn as nn
from transformers import ConvNextV2ForImageClassification


class MultiTaskConvNeXt(nn.Module):
    """Hard-sharing MTL model: shared ConvNeXt V2 encoder + task-specific heads."""

    def __init__(self, cfg):
        super().__init__()

        # shared encoder ────────────────────────────────────────────────
        # pretrained ConvNeXt V2
        base = ConvNextV2ForImageClassification.from_pretrained(
            cfg["model_id"],
            num_labels=cfg["num_classes"],
            ignore_mismatched_sizes=True,
        )
        self.backbone = base.convnextv2          # the feature extractor
        hidden_dim = base.config.hidden_sizes[-1] # 768 for convnextv2-tiny

        # task-specific heads (short — single linear layer each) ────────
        # main task: 5-class maculopathy classification
        self.cls_head = nn.Linear(hidden_dim, cfg["num_classes"])

        # aux task 1: age regression (continuous, normalised to [0, 1])
        self.age_head = nn.Linear(hidden_dim, 1)

        # aux task 2: data centre prediction (binary)
        self.centre_head = nn.Linear(hidden_dim, 1)

        # global average pooling
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, pixel_values):
        # shared encoder forward pass
        features = self.backbone(pixel_values).last_hidden_state   # (B, C, H, W)
        pooled = self.pool(features).flatten(1)                    # (B, hidden_dim)

        # task-specific heads
        cls_logits  = self.cls_head(pooled)        # (B, 5)
        age_pred    = self.age_head(pooled)         # (B, 1)
        centre_pred = self.centre_head(pooled)      # (B, 1)

        return cls_logits, age_pred.squeeze(-1), centre_pred.squeeze(-1)