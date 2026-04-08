import torch.nn as nn
import torch.nn.functional as F

__all__ = ["ClassificationHead"]


class ClassificationHead(nn.Module):
    """Mean-pool encoder tokens → MLP → multi-task binary classification.

    Used to evaluate encoder representation quality on ChEMBL2K's 41 bioactivity tasks.

    head_type:
        'small' — original: d→256→num_tasks  (dropout 0.5)
        'wide'  — wider + less dropout: d→512→num_tasks  (dropout 0.3)
        'deep'  — wider + deeper: d→512→256→num_tasks  (dropout 0.5)
    """
    def __init__(self, d, num_tasks, head_type="small", drop_ratio=None):
        super().__init__()
        if head_type == "small":
            dr = drop_ratio if drop_ratio is not None else 0.5
            self.net = nn.Sequential(
                nn.Linear(d, 256),
                nn.ReLU(),
                nn.Dropout(dr),
                nn.Linear(256, num_tasks),
            )
        elif head_type == "wide":
            dr = drop_ratio if drop_ratio is not None else 0.5
            self.net = nn.Sequential(
                nn.Linear(d, 512),
                nn.ReLU(),
                nn.Dropout(dr),
                nn.Linear(512, num_tasks),
            )
        elif head_type == "deep":
            dr = drop_ratio if drop_ratio is not None else 0.5
            self.net = nn.Sequential(
                nn.Linear(d, 512),
                nn.GELU(),
                nn.Dropout(dr),
                nn.LayerNorm(512),
                nn.Linear(512, 256),
                nn.GELU(),
                nn.Dropout(dr),
                nn.LayerNorm(256),
                nn.Linear(256, num_tasks),
            )
        else:
            raise ValueError(f"Unknown head_type: {head_type}")

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
