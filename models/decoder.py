import torch.nn as nn
import torch.nn.functional as F
import random

__all__ = ["FingerprintDecoder", "SMILESDecoder", "CellInpaintingDecoder", "GeneExpressionDecoder"]

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
    """Mean-pool encoder tokens → MLP → reconstruct SMILES."""
    def __init__(self, d, vocab_size, hidden_dim=512, n_layers=2):
        super().__init__()
        self.vocab_size = vocab_size

        self.z_to_h = nn.Linear(d, hidden_dim * n_layers)
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, n_layers, batch_first=True)

        self.fc_out = nn.Linear(hidden_dim, vocab_size)
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        self.modality = 'SMILES'

    def forward(self, z, target_tokens):
        z_global = z.mean(dim=1)
        h0 = self.z_to_h(z_global).view(self.n_layers, -1, self.hidden_dim)
        embedded = self.embedding(target_tokens)
        output, _ = self.gru(embedded, h0)
        logits = self.fc_out(output)
        return logits

    def loss(self, z, target_tokens):
        logits = self.forward(z, target_tokens[:, :-1])
        targets = target_tokens[:, 1:]
        return F.cross_entropy(logits.reshape(-1, self.vocab_size), targets.reshape(-1))


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
    def __init__(self, d, hidden=512, ge_dim=18211):
        super().__init__()
        self.modality = 'gene_expression'


class MultipleDecoders(nn.Module):
    def __init__(self, decoders: list[nn.Module], factors: list[float]=None):
        super().__init__()
        self.decoders = decoders
        self.factors = factors if factors is not None else [1] * len(decoders)
        self.modality = 'multiple'

    def forward(self, z):
        decoder = random.choice(self.decoders)
        return decoder.forward(z)

    def loss(self, z, data):
        idx = random.randint(1, len(self.decoders)) - 1
        decoder = self.decoders[idx]
        factor = self.factors[idx]
        return factor * decoder.loss(z, data[decoder.modality])
