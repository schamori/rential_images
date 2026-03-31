import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal Loss: down-weights easy examples, focuses on hard/minority ones."""
    def __init__(self, gamma=2.0, weight=None, label_smoothing=0.0):
        super().__init__()
        self.gamma = gamma
        self.weight = weight  # per-class weights (same as in WCE)
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, weight=self.weight, reduction="none", label_smoothing=self.label_smoothing)
        pt = torch.exp(-ce)                        # probability of correct class
        return ((1 - pt) ** self.gamma * ce).mean()


def get_loss(cfg, class_weights, device):
    w = class_weights.to(device) if cfg["use_class_weights"] else None
    smoothing = cfg.get("label_smoothing", 0.0)

    if cfg["loss"] == "focal":
        return FocalLoss(gamma=cfg.get("focal_gamma", 2.0), weight=w, label_smoothing=smoothing)
    else:
        # "ce" or "weighted_ce"
        return nn.CrossEntropyLoss(weight=w, label_smoothing=smoothing)
