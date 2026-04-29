"""
Pretraining dataset: ~2M SMILES from antoinebcx/smiles-molecules-chembl.

Downloads via HuggingFace `datasets`, tokenises SMILES with the same
character-level tokeniser used for ChEMBL2K, and caches the result.
Returns (token_ids,) per sample — labels are not needed for self-supervised
encoder pretraining.
"""

import os
import os.path as osp
import json
import torch
from torch.utils.data import Dataset

from .smiles_tokenizer import build_vocab, encode, PAD_TOKEN, MASK_TOKEN
from .smiles_enumerator import enumerate_smiles


class PretrainSMILESDataset(Dataset):
    """HuggingFace ChEMBL SMILES dataset for self-supervised pretraining."""

    HF_REPO = "antoinebcx/smiles-molecules-chembl"

    def __init__(self, root="raw_data", cache_name="chembl_pretrain",
                 max_len=128, n_augmentations=0):
        super().__init__()
        self.root = root
        self.cache_dir = osp.join(root, cache_name)
        self.max_len = max_len

        os.makedirs(self.cache_dir, exist_ok=True)

        vocab_path = osp.join(self.cache_dir, "vocab.json")
        aug_tag = f"_aug{n_augmentations}" if n_augmentations > 0 else ""
        tokens_path = osp.join(self.cache_dir, f"tokens_L{max_len}{aug_tag}.pt")

        if osp.exists(tokens_path) and osp.exists(vocab_path):
            print("Loading cached pretrain tokens ...")
            self.data = torch.load(tokens_path, weights_only=False)
            with open(vocab_path) as f:
                self.vocab = json.load(f)
        else:
            print(f"Downloading {self.HF_REPO} via HuggingFace datasets ...")
            from datasets import load_dataset, concatenate_datasets
            ds = load_dataset(self.HF_REPO)

            # Use all splits for pretraining (self-supervised, no label leakage)
            all_splits = [ds[s] for s in ds.keys()]
            full = concatenate_datasets(all_splits)
            smiles_list = full["smiles"]
            print(f"Total SMILES: {len(smiles_list):,}")

            # Build vocab from the full large dataset
            print("Building vocabulary ...")
            self.vocab = build_vocab(smiles_list)
            with open(vocab_path, "w") as f:
                json.dump(self.vocab, f)
            print(f"Vocab size: {len(self.vocab)}")

            # Tokenise all SMILES (+ augmented enumerations)
            print("Tokenising SMILES ...")
            token_list = []
            for smi in smiles_list:
                ids, _ = encode(smi, self.vocab, max_len)
                token_list.append(ids)

                if n_augmentations > 0:
                    aug_smiles = enumerate_smiles(smi, n_augmentations)
                    for aug_smi in aug_smiles:
                        aug_ids, _ = encode(aug_smi, self.vocab, max_len)
                        token_list.append(aug_ids)
                    # Pad with canonical if fewer unique enumerations found
                    for _ in range(n_augmentations - len(aug_smiles)):
                        token_list.append(ids)

            self.data = torch.stack(token_list)

            torch.save(self.data, tokens_path)
            n_orig = len(smiles_list)
            factor = 1 + n_augmentations
            print(f"Cached {len(self.data):,} tokenised SMILES "
                  f"({n_orig:,} molecules × {factor} variants)")

        self.vocab_size = len(self.vocab)
        self.pad_token_id = self.vocab[PAD_TOKEN]
        self.mask_token_id = self.vocab[MASK_TOKEN]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return (self.data[idx],)

    @staticmethod
    def vocab_path(root="raw_data", cache_name="chembl_pretrain"):
        """Return path to the saved vocabulary file."""
        return osp.join(root, cache_name, "vocab.json")
