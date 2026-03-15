import torch.nn as nn
import torch.nn.functional as F

__all__ = ["FingerprintDecoder"]


class FingerprintDecoder(nn.Module):
    """Mean-pool encoder tokens → MLP → reconstruct 2048-bit Morgan fingerprint."""
    def __init__(self, d, hidden=512, fp_dim=2048):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, hidden),
            nn.ReLU(),
            nn.Linear(hidden, fp_dim),
        )

    def forward(self, z):
        # z: (B, L, d) → mean pool over sequence → (B, d) → (B, fp_dim)
        return self.net(z.mean(dim=1))

    def loss(self, z, fingerprint):
        logits = self.forward(z)
        return F.binary_cross_entropy_with_logits(logits, fingerprint)
