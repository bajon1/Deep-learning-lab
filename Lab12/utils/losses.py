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



class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, smooth=1e-6, from_logits=True):
        super().__init__()
        self.alpha = alpha
        self.beta  = beta
        self.smooth = smooth
        self.from_logits = from_logits

    def forward(self, logits, targets):
        if self.from_logits:
            probs = torch.sigmoid(logits.squeeze(1) if logits.dim() == 2 else logits)
        else:
            probs = logits.squeeze(1)
        targets = targets.float()
        tp = (probs * targets).sum()
        fp = (probs * (1.0 - targets)).sum()
        fn = ((1.0 - probs) * targets).sum()
        tversky_index = (tp + self.smooth) / (
            tp + self.alpha * fp + self.beta * fn + self.smooth
        )
        return 1.0 - tversky_index