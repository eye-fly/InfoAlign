import torch.nn as nn
import torch.nn.functional as F

__all__ = ["ClassificationHead"]


class ClassificationHead(nn.Module):
    """Mean-pool encoder tokens → MLP → multi-task binary classification.

    Used to evaluate encoder representation quality on ChEMBL2K's 41 bioactivity tasks.
    """
    def __init__(self, d, num_tasks, hidden=256, drop_ratio=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, hidden),
            nn.ReLU(),
            nn.Dropout(drop_ratio),
            nn.Linear(hidden, num_tasks),
        )

    def forward(self, z, pad_mask=None):
        # z: (B, L, d) → masked mean pool → (B, d) → (B, num_tasks)
        if pad_mask is not None:
            # pad_mask: (B, L) bool, True = real token, False = padding
            mask = pad_mask.unsqueeze(-1).float()  # (B, L, 1)
            pooled = (z * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            pooled = z.mean(dim=1)
        return self.net(pooled)

    def loss(self, z, targets, pad_mask=None):
        logits = self.forward(z, pad_mask=pad_mask)
        is_labeled = targets == targets  # mask NaN
        return F.binary_cross_entropy_with_logits(
            logits[is_labeled],
            targets[is_labeled],
        )
