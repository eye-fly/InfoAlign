# import numpy as np
# import pandas as pd
import math
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend
from torch import linalg as LA
import copy
import itertools

__all__ = ["Encoder"]

#########################################
########### TRANSFORMER #################
#########################################

# JUST FOR REFERENCE, TODO change embedding layer properly
class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size, embed_dim, max_len):
        super(EmbeddingLayer, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_len, embed_dim)

    def forward(self, x):
        # x: (batch_size, seq_len)
        seq_len = x.size(1)
        positions = (
            torch.arange(seq_len, dtype=torch.long, device=x.device)
            .unsqueeze(0)
            .expand_as(x)
        )
        token_embeddings = self.token_embedding(x)
        position_embeddings = self.position_embedding(positions)
        embeddings = token_embeddings + position_embeddings
        return embeddings

class AttentionLayer(nn.Module):
    def __init__(
        self,
        dmodel,
        heads,
    ):
        super(AttentionLayer, self).__init__()

        self.ln = nn.LayerNorm(dmodel)

        self.heads = heads

        self.input_projection = nn.Linear(dmodel, 3 * dmodel, bias=False)

        self.output_projection = nn.Linear(dmodel, dmodel, bias=False)

    def forward(self, x, attention_mask):
        x = self.ln(x)

        projected = self.input_projection(x)

        batch, seq_len = x.shape[:-1]
        q_chunk, k_chunk, v_chunk = torch.chunk(projected, chunks=3, dim=-1)
        query = q_chunk.view(batch, seq_len, self.heads, -1).transpose(1, 2)
        key = k_chunk.view(batch, seq_len, self.heads, -1).transpose(1, 2)
        value = v_chunk.view(batch, seq_len, self.heads, -1).transpose(1, 2)

        with torch.nn.attention.sdpa_kernel(
            [
                SDPBackend.FLASH_ATTENTION,
                SDPBackend.EFFICIENT_ATTENTION,
                SDPBackend.MATH,
            ]
        ):
            attention_output = F.scaled_dot_product_attention(
                query=query,
                key=key,
                value=value,
                attn_mask=attention_mask,
                is_causal=False, # This is not LLM, but fingerprints, we do not look only at previous tokens, 
                # but whole fingerprint vector.
            )

        output = self.output_projection(attention_output.transpose(1, 2).flatten(-2))
        return output

class SwiGLU(nn.Module):
    """
    Arguments:
    1) dmodel: Dimension of model embedding.
    2) beta:   Parameter of function Swish_beta. Default=1.
    3) scale:  Scale of hidden dimension in regard to dmodel. In ff_layer it was 4, but
               in paper they propose to change it into 2/3 of that value, so approximately 2.66666. 
               The reason is that we want to keep similar number of parameters.
    4) bias:   Whether we want add learnable bias in linear projections or not. In paper both options
               were considered, but eventually they have ommited bias. However bias should not harm
               training effectiveness, so we keep it (same as in ff_layer). 
    """
    def __init__(
        self,
        dmodel,
        beta=1.,
        scale=2.66666,
        bias=True
    ):
        super(SwiGLU, self).__init__()

        self.beta = beta
        hidden_dim = int(scale * dmodel)
        self.ln = nn.LayerNorm(dmodel)
        self.swish_projection1 = nn.Linear(
            dmodel,
            hidden_dim,
            bias=bias,
        )
        self.swish_projection2 = nn.Linear(
            dmodel,
            hidden_dim,
            bias=bias,
        )
        self.output_projection = nn.Linear(hidden_dim, dmodel, bias=bias)
    def swish(self, x):
        y = torch.sigmoid(self.beta * x)
        return x * y

    def forward(self, x):
        x = self.ln(x)
        x1 = self.swish(self.swish_projection1(x))
        x2 = self.swish_projection2(x)
        x = x1 * x2
        return self.output_projection(x)

class Block(nn.Module):

    def __init__(
        self,
        dmodel,
        heads,
    ):
        super().__init__()
        self.attention_layer = AttentionLayer(dmodel, heads)
        # self.feed_forward_layer = FeedForward(dmodel) # DELETED
        self.swiglu_layer = SwiGLU(dmodel) # ADDED

    def forward(self, x, attention_mask):
        out_attention = self.attention_layer(x, attention_mask)
        x = x + out_attention

        out_swiglu = self.swiglu_layer(x) # SMALL CHANGE
        x = x + out_swiglu
        return x


class Transformer(nn.Module):
    def __init__(self, dx, d_model, num_heads, num_layers, vocab_size=None, pad_token_id=0, max_len=512):
        super().__init__()
        if vocab_size is not None:
            # SMILES / token sequence mode: input is (B, L) LongTensor
            self.embedding  = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
            self.input_proj = None
        else:
            # Float feature mode: input is (B, L, dx) FloatTensor
            self.embedding  = None
            self.input_proj = nn.Linear(dx, d_model)

        self.position_embedding = nn.Embedding(max_len, d_model)

        self.blocks = nn.ModuleList(
            [Block(d_model, num_heads) for _ in range(num_layers)]
        )

    def forward(self, input_ids, attention_mask=None, return_all_layers=False):
        if self.embedding is not None:
            output = self.embedding(input_ids)   # (B, L) -> (B, L, d_model)
        else:
            output = self.input_proj(input_ids)  # (B, L, dx) -> (B, L, d_model)

        positions = torch.arange(output.size(1), device=output.device).unsqueeze(0)
        output = output + self.position_embedding(positions)

        all_layers = []
        for block in self.blocks:
            output = block(output, attention_mask)
            if return_all_layers:
                all_layers.append(output)

        if return_all_layers:
            return output, all_layers
        return output

#######################################
############## CODEWORDS ##############
#######################################

def applyMask(x, mask_prob=0.5):
    # In case of audio, it probably would be better if we mask whole spans of audio frames
    # because of similarity between adjacent frames. In our case this is rather not the problem.
    B, L, D = x.shape
    mask = torch.rand(B, L) < mask_prob
    xMask = x.clone()
    xMask[mask,:] = 0
    return xMask, mask

def applyTokenMask(x, mask_prob, mask_token_id):
    # x: (B, L) LongTensor of token ids
    # Returns masked token ids and a bool mask of which positions were masked.
    B, L = x.shape
    mask = torch.rand(B, L, device=x.device) < mask_prob
    xMask = x.clone()
    xMask[mask] = mask_token_id
    return xMask, mask

def applyMaskWithSpans(x, mask_ratio=0.5, mask_span=10):
    # We should test this to std applyMask. This version masks mask_ratio of frames,
    # different in each sample in batch, whole spans. It is slower than applyMask
    # because of not using pytorch enough, but just python.
    B, L, D = x.shape
    mask = torch.zeros(B, L, dtype=torch.bool)

    total_to_mask = int(L * mask_ratio)

    for b in range(B):
        masked = 0
        while masked < total_to_mask:
            start = random.randint(0, L - 1)
            span = random.randint(1, mask_span)
            end = min(L, start + span)

            for i in range(start, end):
                if not mask[b, i]:
                    mask[b, i] = True
                    masked += 1
                    if masked >= total_to_mask:
                        break

    x_masked = x.clone()
    x_masked[mask] = 0.0  # lepiej: learned mask embedding

    return x_masked, mask

################################################################

class Codebook(nn.Module):
    def __init__(self, V : int, d : int, gamma_codebook=0.99):
        super().__init__()
        self.register_buffer("s", torch.randn(V, d))
        self.register_buffer("n", torch.ones(V, 1))
        self.register_buffer("e", self.s / self.n)

        self.gamma_codebook = gamma_codebook
        self.V = V
        self.d = d

    def update(self, y, z):
        # z: (B, L, d) — teacher representations
        # y: (B, L)    — codebook assignments
        assert len(z.shape) == 3
        assert z.shape[2] == self.d
        for v in range(self.V):
            z_v = z[y == v]
            self.s[v] = self.gamma_codebook*self.s[v] + (1-self.gamma_codebook)*z_v.sum(dim=0)
            self.n[v] = self.gamma_codebook * self.n[v] + (1 - self.gamma_codebook) * (y == v).sum()
        self.e = self.s / (self.n + 1e-8)

        # Dead codeword restart: re-initialise any codeword with near-zero usage
        # from a random live teacher embedding so it can compete again.
        dead = (self.n < 1.0).squeeze(-1)  # (V,) bool
        if dead.any():
            live = z.reshape(-1, self.d).detach()
            perm = torch.randperm(live.shape[0], device=live.device)[:dead.sum()]
            self.e[dead] = live[perm]
            self.s[dead] = self.e[dead] * self.n[dead]
    
    def findMostSimilarInCodebook(self, z):
        # z: (B,L,d)
        # e: (V,d)
        B,L,d = z.shape
        z_flat = z.reshape(B * L, d)                 # [BL, d]
        dist = torch.cdist(z_flat, self.e, p=2.0)    # [BL, V]
        y = dist.argmin(dim=1).view(B, L)            # [B, L]
        return y

class PseudoLabelPredictionHead(nn.Module):
    def __init__(self, d, V):
        super().__init__()
        self.ll = nn.Linear(d, V, bias=True)
    def forward(self, z):
        return self.ll(z) # only logits, later softmax
        # out = self.ll(z)
        # return F.softmax(out, -1) # dim = -1, it represents V values

class Encoder(nn.Module):
    def __init__(self, V, dx, d, num_heads, num_layers, K_layers,
                 gamma_teacher=0.996, gamma_codebook=0.99, mask_prob=0.15,
                 vocab_size=None, mask_token_id=None, pad_token_id=0):
        super().__init__()
        assert gamma_teacher <= 1. and gamma_teacher >= 0.
        assert gamma_codebook <= 1. and gamma_codebook >= 0.
        self.V = V
        self.d = d
        self.K_layers = K_layers
        self.gamma_teacher = gamma_teacher
        self.gamma_codebook = gamma_codebook
        self.mask_prob = mask_prob
        self.vocab_size    = vocab_size
        self.mask_token_id = mask_token_id
        self.pad_token_id  = pad_token_id
        self.encoder = Transformer(dx, d, num_heads, num_layers, vocab_size=vocab_size, pad_token_id=pad_token_id)
        self.teacher = copy.deepcopy(self.encoder)
        for p in self.teacher.parameters():
            p.requires_grad=False
        self.heads = nn.ModuleDict({
            str(k) : PseudoLabelPredictionHead(self.d, self.V)
            for k in self.K_layers
        })
        self.codebooks = nn.ModuleDict({
            str(k) : Codebook(self.V, self.d, gamma_codebook=gamma_codebook)
            for k in self.K_layers
        })
    
    @torch.no_grad()
    def codebook_stats(self):
        """Returns active codeword count and assignment entropy per K layer."""
        stats = {}
        for k in self.K_layers:
            n = self.codebooks[str(k)].n.squeeze(-1)  # (V,)
            active = (n > 1.0).sum().item()
            p = n / (n.sum() + 1e-8)
            entropy = -(p * (p + 1e-8).log()).sum().item()
            stats[k] = {"active": active, "entropy": round(entropy, 3)}
        return stats

    def student_params(self):
        return itertools.chain(
            self.encoder.parameters(),
            *[h.parameters() for h in self.heads.values()]
        )
    
    @torch.no_grad()
    def update_teacher(self):
        for pt, ps in zip(self.teacher.parameters(), self.encoder.parameters()):
            pt.data.mul_(self.gamma_teacher).add_(ps.data, alpha=1-self.gamma_teacher)

    def forward(self, x, return_loss=False, update_codebooks=False, return_masked_info=False):
        if not return_loss:
            return self.encoder(x)
            
        # x can be:
        #   LongTensor  (B, L)     — SMILES token ids
        #   FloatTensor (B, L, dx) — continuous features (fingerprint chunks etc.)
        if x.dtype == torch.long:
            B, L = x.shape
            xMask, mask = applyTokenMask(x, self.mask_prob, self.mask_token_id)
            # Exclude padding positions from the loss
            valid = mask & (x != self.pad_token_id)
        else:
            B, L, dx = x.shape
            xMask, mask = applyMask(x, mask_prob=self.mask_prob)
            valid = mask

        z = self.encoder(xMask)                                    # (B, L, d)
        with torch.no_grad():
            _, z_teacher_layers = self.teacher(x, return_all_layers=True)

        loss = 0
        for k in self.K_layers:
            z_layer = z_teacher_layers[k]
            y = self.codebooks[str(k)].findMostSimilarInCodebook(z_layer)  # (B, L)
            if update_codebooks:
                self.codebooks[str(k)].update(y, z_layer.detach())
            logits   = self.heads[str(k)](z)
            log_probs = F.log_softmax(logits, dim=-1)
            log_phi  = log_probs.gather(dim=2, index=y.unsqueeze(-1)).squeeze(-1)
            num_valid = valid.sum().clamp(min=1)
            loss -= log_phi[valid].sum() / num_valid

        if return_masked_info:
            return loss, z, valid, x  # z at all positions, mask of valid masked positions, original tokens
        return loss


# PSEUDOCODE for training:
# def train_encoder(dataloader, encoder, epochs):
#     optimizer = torch.optim.SGD(encoder.student_params(), lr=0.001, momentum=0.9)
#     for epoch in range(epochs):
#         for i, data in enumerate(dataloader):
#             inputs, labels = data
#             # xMask, mask = applyMask(x, mask_prob=0.5)
#             assert len(inputs.shape) == 3 # B,L,dx
#             optimizer.zero_grad()
#             loss = encoder.loss(inputs, update_codebooks=True)
#             loss.backward()
#             optimizer.step()
#             encoder.update_teacher()
#             # TODO: add on wandb all loss values.


# encoder = Encoder(V=100, d=1000, K_layers=[0,1,2])






