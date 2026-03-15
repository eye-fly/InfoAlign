import torch.nn as nn
import torch.nn.functional as F

__all__ = ["ClassificationHead"]


class ClassificationHead(nn.Module):
    """Mean-pool encoder tokens → MLP → multi-task binary classification.

    Used to evaluate encoder representation quality on ChEMBL2K's 41 bioactivity tasks.
    """
    def __init__(self, d, num_tasks, hidden=256, drop_ratio=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, hidden),
            nn.ReLU(),
            nn.Dropout(drop_ratio),
            nn.Linear(hidden, num_tasks),
        )

    def forward(self, z):
        # z: (B, L, d) → mean pool → (B, d) → (B, num_tasks)
        return self.net(z.mean(dim=1))

    def loss(self, z, targets):
        logits = self.forward(z)
        is_labeled = targets == targets  # mask NaN
        return F.binary_cross_entropy_with_logits(
            logits[is_labeled],
            targets[is_labeled],
        )
