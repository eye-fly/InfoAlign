import torch.nn as nn
import torch.nn.functional as F
import random

__all__ = ["FingerprintDecoder", "CellInpaintingDecoder", "GeneExpressionDecoder"]

from scipy._lib.array_api_compat import torch


class FingerprintDecoder(nn.Module):
    """Mean-pool encoder tokens → MLP → reconstruct 2048-bit Morgan fingerprint."""
    def __init__(self, d, hidden=512, fp_dim=2048):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, hidden),
            nn.ReLU(),
            nn.Linear(hidden, fp_dim),
        )
        self.modality = 'fingerprints'

    def forward(self, z):
        # z: (B, L, d) → mean pool over sequence → (B, d) → (B, fp_dim)
        return self.net(z.mean(dim=1))

    def loss(self, z, fingerprint):
        logits = self.forward(z)
        return F.binary_cross_entropy_with_logits(logits, fingerprint)


# TODO
class SMILESDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.modality = 'SMILES'


class CellInpaintingDecoder(nn.Module):
    """Mean-pool encoder tokens → MLP → reconstruct Cell inpainting profile"""
    def __init__(self, d, hidden=512, cp_dim=5792):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, hidden),
            nn.ReLU(),
            nn.Linear(hidden, cp_dim),
        )
        self.modality = 'cell_inpainting'

    def forward(self, z):
        # z: (B, L, d) → mean pool over sequence → (B, d) → (B, cp_dim)
        return self.net(z.mean(dim=1))

    def loss(self, z, profile):
        logits = self.forward(z)
        return F.binary_cross_entropy_with_logits(logits, profile)


# TODO
class GeneExpressionDecoder(nn.Module):
    """Mean-pool encoder tokens → MLP → reconstruct Gene Expression"""
    def __init__(self, d, hidden=512, cp_dim=5792):
        super().__init__()
        self.modality = 'gene_expression'


class MultipleDecoders(nn.Module):
    def __init__(self, decoders: list[nn.Module], factors: list[float]=None):
        super().__init__()
        self.decoders = decoders
        self.factors = factors if factors is not None else [1] * len(decoders)

    def forward(self, z):
        decoder = random.choice(self.decoders)
        return decoder.forward(z)

    def loss(self, z, data):
        idx = random.randint(1, len(self.decoders)) - 1
        decoder = self.decoders[idx]
        factor = self.factors[idx]
        return factor * decoder.loss(z, data[decoder.modality])
