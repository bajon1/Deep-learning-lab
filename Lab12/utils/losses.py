import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha     = alpha
        self.gamma     = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        bce     = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs   = torch.sigmoid(logits)
        pt      = torch.where(targets == 1, probs, 1 - probs)
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        loss    = alpha_t * (1 - pt).pow(self.gamma) * bce
        if self.reduction == "mean": return loss.mean()
        elif self.reduction == "sum": return loss.sum()
        return loss



class LabelSmoothingBinaryCrossEntropy(nn.Module):
    def __init__(self, eps: float = 0.1, reduction: str = "mean"):
        super().__init__()
        self.eps       = eps
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        targets          = targets.float()
        smoothed_targets = targets * (1.0 - self.eps) + (1.0 - targets) * self.eps
        return F.binary_cross_entropy_with_logits(logits, smoothed_targets,
                                                   reduction=self.reduction)