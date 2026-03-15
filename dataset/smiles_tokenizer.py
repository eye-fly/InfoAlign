import re
import torch

# Handles multi-char atoms (Br, Cl), bracket atoms ([NH+] etc.), and all bond/ring symbols.
PATTERN = r'(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])'

PAD_TOKEN  = '<pad>'   # always id=0
MASK_TOKEN = '<mask>'  # always id=1


def tokenize(smiles: str):
    return re.findall(PATTERN, smiles)


def build_vocab(smiles_list):
    tokens = set()
    for s in smiles_list:
        tokens.update(tokenize(s))
    vocab = {PAD_TOKEN: 0, MASK_TOKEN: 1}
    for i, tok in enumerate(sorted(tokens)):
        vocab[tok] = i + 2
    return vocab


def encode(smiles: str, vocab: dict, max_len: int):
    toks = tokenize(smiles)[:max_len]
    ids = [vocab.get(t, 0) for t in toks]
    n_real = len(ids)
    ids += [0] * (max_len - n_real)
    return torch.tensor(ids, dtype=torch.long), n_real
