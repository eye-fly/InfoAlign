import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["FingerprintDecoder", "GEDecoder", "SMILESDecoder"]


class FingerprintDecoder(nn.Module):
    """Mean-pool encoder tokens → MLP → reconstruct 2048-bit Morgan fingerprint."""
    def __init__(self, d, hidden=512, fp_dim=2048):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, hidden),
            nn.ReLU(),
            nn.Linear(hidden, fp_dim),
        )

    def forward(self, z, pad_mask=None):
        # z: (B, L, d) → masked mean pool over sequence → (B, d) → (B, fp_dim)
        if pad_mask is not None:
            mask = pad_mask.unsqueeze(-1).float()  # (B, L, 1)
            pooled = (z * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            pooled = z.mean(dim=1)
        return self.net(pooled)

    def loss(self, z, fingerprint, pad_mask=None):
        logits = self.forward(z, pad_mask=pad_mask)
        return F.binary_cross_entropy_with_logits(logits, fingerprint)


class SMILESDecoder(nn.Module):
    """Linear head predicting original token ID at masked positions (BERT-style).

    Receives encoder output z (B, L, d), a bool mask of valid masked positions (B, L),
    and original token ids (B, L). Computes cross-entropy only at masked positions.
    Per-token supervision — prevents codebook collapse unlike global decoders.
    """
    def __init__(self, d, vocab_size):
        super().__init__()
        self.head = nn.Linear(d, vocab_size)

    def loss(self, z, mask, original_tokens):
        # z: (B, L, d), mask: (B, L) bool, original_tokens: (B, L) long
        logits = self.head(z[mask])          # (N_masked, vocab_size)
        targets = original_tokens[mask]      # (N_masked,)
        return F.cross_entropy(logits, targets)


class GEDecoder(nn.Module):
    """Mean-pool encoder tokens → MLP → reconstruct 978-dim L1000 gene expression profile."""
    def __init__(self, d, hidden=512, ge_dim=978):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, hidden),
            nn.ReLU(),
            nn.Linear(hidden, ge_dim),
        )

    def forward(self, z):
        # z: (B, L, d) → mean pool → (B, d) → (B, ge_dim)
        return self.net(z.mean(dim=1))

    def loss(self, z, ge_targets):
        valid = ~torch.isnan(ge_targets).any(dim=1)  # (B,) — only compounds with GE data
        if not valid.any():
            return torch.tensor(0.0, device=z.device)
        pred = self.forward(z[valid])
        return F.mse_loss(pred, ge_targets[valid])
