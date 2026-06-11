# -*- coding: utf-8 -*-
import torch as tc
import torch.nn as nn
import torch.nn.functional as F
from config import GAMMA, ALPHA, CONTRASTIVE_TEMP

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets, mask=None):
        bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        pt = tc.exp(-bce)
        loss = self.alpha * (1 - pt) ** self.gamma * bce
        if mask is not None:
            loss = loss * mask.float()
            return loss.sum() / (mask.float().sum() + 1e-6)
        return loss.mean() if self.reduction == "mean" else loss.sum()


class FocalLossWithPosWeight(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25, pos_weight=1.0, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.pos_weight = pos_weight
        self.reduction = reduction

    def forward(self, inputs, targets, mask=None):
        bce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets, reduction="none", 
            pos_weight=tc.tensor(self.pos_weight, device=inputs.device)
        )
        pt = tc.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        if mask is not None:
            focal_loss = focal_loss * mask.float()
            return focal_loss.sum() / (mask.float().sum() + 1e-6)
        return focal_loss.mean() if self.reduction == "mean" else focal_loss.sum()


class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=CONTRASTIVE_TEMP):
        super().__init__()
        self.temperature = temperature

    def forward(self, feat1, feat2):
        B = feat1.size(0)
        feat1 = F.normalize(feat1, dim=-1)
        feat2 = F.normalize(feat2, dim=-1)
        logits = tc.matmul(feat1, feat2.T) / self.temperature
        labels = tc.arange(B, device=feat1.device)
        loss = F.cross_entropy(logits, labels)
        return loss


class UncertaintyWeightedLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred_mean, y_pred_log_var, y_true, mask=None):
        loss = 0.5 * tc.exp(-y_pred_log_var) * (y_pred_mean - y_true)**2 + 0.5 * y_pred_log_var
        if mask is not None:
            loss = loss * mask.float()
            return loss.sum() / (mask.float().sum() + 1e-6)
        return loss.mean()