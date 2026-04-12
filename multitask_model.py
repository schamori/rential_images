"""
Multi-task wrapper around ConvNextV2ForImageClassification.

Architecture (hard parameter sharing):
    Shared ConvNeXt V2 encoder  (pretrained, fine-tuned)
        ├── Classification head  → 5 classes   (main task)
        ├── Age regression head  → 1 neuron    (auxiliary)
        └── Centre prediction    → 1 neuron    (auxiliary)
        + optional gradient reversal (DANN mode)

Usage:
    model = MultiTaskConvNeXt(cfg)
    cls_logits, age_pred, centre_pred = model(images)

    # DANN mode — call set_grl_lambda() each epoch to ramp up reversal
    model = MultiTaskConvNeXt(cfg)          # with cfg["dann"] = true
    model.set_grl_lambda(0.5)              # controls reversal strength
    cls_logits, age_pred, centre_pred = model(images)
"""
import torch
import torch.nn as nn
from transformers import ConvNextV2ForImageClassification

# gradient reversal layer ────────────────────────────────────────────────────
class GradientReversalFn(Function):
    """Reverses gradients in the backward pass, scaled by lambda."""
 
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.clone()
 
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lam * grad_output, None
 
 
class GradientReversalLayer(nn.Module):
    """Wraps GradientReversalFn as a module with a settable lambda."""
 
    def __init__(self):
        super().__init__()
        self.lam = 0.0          # start at 0 — ramp up during training
 
    def set_lambda(self, lam):
        self.lam = lam
 
    def forward(self, x):
        return GradientReversalFn.apply(x, self.lam)

# multi-task model ────────────────────────────────────────────────────
class MultiTaskConvNeXt(nn.Module):
    """Hard-sharing MTL model: shared ConvNeXt V2 encoder + task-specific heads.
    When cfg["dann"] is True, a gradient reversal layer is inserted before the centre head so the backbone is trained to SUPPRESS centre information.
    """

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

        # DANN: gradient reversal before centre head
        self.use_dann = cfg.get("dann", False)
        self.grl = GradientReversalLayer() if self.use_dann else None

        # global average pooling
        self.pool = nn.AdaptiveAvgPool2d(1)

    def set_grl_lambda(self, lam):
        """Set the gradient reversal strength (call each epoch)."""
        if self.grl is not None:
            self.grl.set_lambda(lam)

    def forward(self, pixel_values):
        # shared encoder forward pass
        features = self.backbone(pixel_values).last_hidden_state   # (B, C, H, W)
        pooled = self.pool(features).flatten(1)                    # (B, hidden_dim)

        # task-specific heads
        cls_logits  = self.cls_head(pooled)        # (B, 5)
        age_pred    = self.age_head(pooled)         # (B, 1)
        
        # DANN: reverse gradients flowing to backbone through centre head
        if self.use_dann and self.grl is not None:
            centre_pred = self.centre_head(self.grl(pooled)).squeeze(-1)
        else:
            centre_pred = self.centre_head(pooled).squeeze(-1)
 
        return cls_logits, age_pred, centre_pred