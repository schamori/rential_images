"""
Loss functions for MMAC classification.
 
Includes:
    - CrossEntropyLoss (standard and weighted)
    - FocalLoss (with optional class weights)
    - MultiTaskLoss (classification + age regression + centre prediction)
"""

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

class MultiTaskLoss(nn.Module):
    """
    Combined loss for multi-task learning.
    - L_cls:    cross-entropy (or focal) for 5-class maculopathy grade
    - L_age:    smooth L1 on normalised age (masked for missing values)
    - L_centre: binary cross-entropy for data centre prediction
 
    Args:
        cls_criterion:  the classification loss (CE or Focal, from get_loss)
        lambda_age:     weight for the age regression loss
        lambda_centre:  weight for the centre prediction loss
    """
    def __init__(self, cls_criterion, lambda_age=0.5, lambda_centre=0.3):
        super().__init__()
        self.cls_criterion = cls_criterion
        self.lambda_age = lambda_age
        self.lambda_centre = lambda_centre
 
    def forward(self, cls_logits, age_pred, centre_pred,
                grade_targets, age_targets, age_valid, centre_targets):
        """
        Args:
            cls_logits:      (B, 5) raw classification logits
            age_pred:        (B,)   predicted normalised age
            centre_pred:     (B,)   predicted centre logit (pre-sigmoid)
            grade_targets:   (B,)   integer class labels 0-4
            age_targets:     (B,)   normalised age (0 if missing)
            age_valid:       (B,)   1.0 if age present, 0.0 if missing
            centre_targets:  (B,)   binary centre label (0 or 1)
 
        Returns:
            total_loss, dict of individual losses for logging
        """
        # classification loss (main task)
        loss_cls = self.cls_criterion(cls_logits, grade_targets)
 
        # age regression loss (auxiliary, masked for missing)
        if age_valid.sum() > 0:
            # only compute loss for samples with valid age
            age_loss_raw = F.smooth_l1_loss(age_pred, age_targets, reduction="none")
            loss_age = (age_loss_raw * age_valid).sum() / age_valid.sum()
        else:
            loss_age = torch.tensor(0.0, device=cls_logits.device)
 
        # centre prediction loss (auxiliary)
        loss_centre = F.binary_cross_entropy_with_logits(
            centre_pred, centre_targets, reduction="mean"
        )
 
        # total weighted loss
        total = loss_cls + self.lambda_age * loss_age + self.lambda_centre * loss_centre
 
        return total, {
            "cls": loss_cls.item(),
            "age": loss_age.item(),
            "centre": loss_centre.item(),
            "total": total.item(),
        }


def get_loss(cfg, class_weights, device):
    w = class_weights.to(device) if cfg["use_class_weights"] else None
    smoothing = cfg.get("label_smoothing", 0.0)

    if cfg["loss"] == "focal":
        return FocalLoss(gamma=cfg.get("focal_gamma", 2.0), weight=w, label_smoothing=smoothing)
    else:
        # "ce" or "weighted_ce"
        return nn.CrossEntropyLoss(weight=w, label_smoothing=smoothing)
