import os
import os.path as osp
import json
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from triton.backends.nvidia.compiler import get_features

from .data_utils import scaffold_split


class PredictionMoleculeDataset(object):
    def __init__(self, name="chembl2k", root="raw_data", transform="fingerprint", vocab=None):
        self.cp_features = None

        assert transform in [
            "fingerprint",
            "smiles",
        ], "Invalid transform type"

        self.name = name
        self.folder = osp.join(root, name)
        self.transform = transform
        self.raw_data = os.path.join(self.folder, "raw", "assays.csv.gz")
        self.task_type = 'finetune'

        self.eval_metric = "roc_auc"
        if name == "chembl2k":
            self.num_tasks = 41
            self.start_column = 4
        elif name == "broad6k":
            self.num_tasks = 32
            self.start_column = 2
        elif "moltoxcast" in self.name:
            self.num_tasks = 617
            self.start_column = 2
        elif name == "biogenadme":
            self.num_tasks = 6
            self.start_column = 4
            self.eval_metric = "avg_mae"
        else:
            meta_path = osp.join(self.folder, "raw", "meta.json")
            if os.path.exists(meta_path):
                with open(meta_path, "r") as f:
                    meta = json.load(f)
                self.num_tasks = meta["num_tasks"]
                self.start_column = meta["start_column"]
            else:
                raise ValueError("Invalid dataset name")

        super(PredictionMoleculeDataset, self).__init__()
        if transform == "smiles":
            self.prepare_smiles_tokenized(vocab=vocab)
        elif transform == "fingerprint":
            self.prepare_fingerprints()

    def get_idx_split(self, to_list=False):
        path = osp.join(self.folder, "split", "scaffold")
        if os.path.isfile(os.path.join(path, "split_dict.pt")):
            split_dict = torch.load(os.path.join(path, "split_dict.pt"), weights_only=False)
        else:
            data_df = pd.read_csv(self.raw_data)
            train_idx, valid_idx, test_idx = scaffold_split(data_df)
            train_idx = torch.tensor(train_idx, dtype=torch.long)
            valid_idx = torch.tensor(valid_idx, dtype=torch.long)
            test_idx = torch.tensor(test_idx, dtype=torch.long)

            os.makedirs(path, exist_ok=True)
            torch.save(
                {"train": train_idx, "valid": valid_idx, "test": test_idx},
                os.path.join(path, "split_dict.pt"),
            )
            split_dict = {"train": train_idx, "valid": valid_idx, "test": test_idx}

        if to_list:
            split_dict = {k: v.tolist() for k, v in split_dict.items()}
        return split_dict

    def resample_train_idx(self, train_idx, ratio, seed=0):
        if ratio >= 1.0:
            return train_idx

        if isinstance(train_idx, torch.Tensor):
            indices = train_idx.tolist()
        else:
            indices = train_idx

        try:
            if isinstance(self.labels, torch.Tensor):
                y_subset = self.labels[indices]
            else:
                y_subset = torch.stack([self.labels[i] for i in indices])

            stratify = None
            if self.eval_metric == 'avg_mae':
                stratify = None
            else:
                if y_subset.dim() > 1:
                    stratify = y_subset.sum(dim=1).long().numpy()
                else:
                    stratify = y_subset.long().numpy()

            if stratify is not None:
                unique, counts = np.unique(stratify, return_counts=True)
                if (counts < 2).any():
                    print(f"Warning: Stratification failed due to insufficient class samples (min count: {counts.min()}). Falling back to random split.")
                    stratify = None

        except Exception as e:
            print(f"Warning: Failed to prepare labels for stratification: {e}. Falling back to random split.")
            stratify = None

        try:
            subset, _ = train_test_split(indices, train_size=ratio, random_state=seed, shuffle=True, stratify=stratify)
        except Exception as e:
            print(f"Warning: Stratification failed during split: {e}. Falling back to random split.")
            subset, _ = train_test_split(indices, train_size=ratio, random_state=seed, shuffle=True, stratify=None)

        return torch.tensor(subset, dtype=torch.long)

    def prepare_smiles_tokenized(self, max_len=128, vocab=None):
        assert os.path.exists(self.raw_data), f"{self.raw_data} does not exist"

        processed_dir = osp.join(self.folder, "processed")
        os.makedirs(processed_dir, exist_ok=True)

        # If an external vocab is provided, always re-tokenise with it
        # (the cached file was tokenised with the dataset's own vocab).
        suffix = "_extv" if vocab is not None else ""
        cache_path = osp.join(processed_dir, f"processed_smiles_L{max_len}{suffix}.pt")

        if osp.exists(cache_path):
            x_list, y_list, vocab = torch.load(cache_path, weights_only=False)
        else:
            from .smiles_tokenizer import build_vocab as _build_vocab, encode
            print("Tokenizing SMILES...")
            data_df = pd.read_csv(self.raw_data)
            smiles_list = data_df["smiles"].tolist()
            if vocab is None:
                vocab = _build_vocab(smiles_list)

            x_list, y_list = [], []
            for _, row in data_df.iterrows():
                ids, _ = encode(row["smiles"], vocab, max_len)
                x_list.append(ids)
                y = torch.tensor([float(row.iloc[col]) for col in range(self.start_column, len(row))], dtype=torch.float32)
                y_list.append(y)

            x_list = torch.stack(x_list)
            y_list = torch.stack(y_list)
            torch.save((x_list, y_list, vocab), cache_path)

        self.data   = x_list
        self.labels = y_list
        self.vocab         = vocab
        self.vocab_size    = len(vocab)
        self.pad_token_id  = vocab['<pad>']
        self.mask_token_id = vocab['<mask>']
        self.max_smiles_len = max_len

        # Load fingerprints as decoder targets (same molecule order as SMILES).
        fp_cache = osp.join(processed_dir, "processed_fp.pt")
        if osp.exists(fp_cache):
            fps, _ = torch.load(fp_cache, weights_only=False)
        else:
            from rdkit import Chem
            from rdkit.Chem import AllChem
            data_df = pd.read_csv(self.raw_data)
            fps = torch.stack([
                torch.tensor(list(AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(row["smiles"]), 2)), dtype=torch.float32)
                for _, row in data_df.iterrows()
            ])
        self.fingerprints = fps

        # Load gene expression features — NaN rows for compounds with no GE data.
        data_df = pd.read_csv(self.raw_data)
        self.ge_features = self._load_ge_features(data_df)

        # Load cell profile features - NaN rows for compounds with no CP data.
        self.cp_features = self._load_cp_features(data_df)

    def prepare_smiles(self):
        assert os.path.exists(
            self.raw_data
        ), f" {self.raw_data} assays.csv.gz does not exist"
        data_df = pd.read_csv(self.raw_data)

        x_list = []
        y_list = []
        for idx, row in data_df.iterrows():
            smiles = row["smiles"]
            x_list.append(smiles)
            y = []
            for col in range(self.start_column, len(row)):
                y.append(float(row.iloc[col]))
            y = torch.tensor(y, dtype=torch.float32)
            y_list.append(y)

        self.data = x_list
        self.labels = y_list

    def prepare_fingerprints(self):
        assert os.path.exists(
            self.raw_data
        ), f" {self.raw_data} assays.csv.gz does not exist"
        data_df = pd.read_csv(self.raw_data)

        processed_dir = osp.join(self.folder, "processed")
        os.makedirs(processed_dir, exist_ok=True)

        if not osp.exists(osp.join(processed_dir, "processed_fp.pt")):
            print("Processing fingerprints...")
            from rdkit import Chem
            from rdkit.Chem import AllChem

            x_list = []
            y_list = []
            for idx, row in data_df.iterrows():
                smiles = row["smiles"]
                mol = Chem.MolFromSmiles(smiles)
                x = torch.tensor(
                    list(AllChem.GetMorganFingerprintAsBitVect(mol, 2)),
                    dtype=torch.float32,
                )
                x_list.append(x)
                y = []
                for col in range(self.start_column, len(row)):
                    y.append(float(row.iloc[col]))
                y = torch.tensor(y, dtype=torch.float32)
                y_list.append(y)

            x_list = torch.stack(x_list, dim=0)
            y_list = torch.stack(y_list, dim=0)
            torch.save((x_list, y_list), osp.join(processed_dir, "processed_fp.pt"))
        else:
            x_list, y_list = torch.load(osp.join(processed_dir, "processed_fp.pt"), weights_only=False)

        self.data = x_list
        self.labels = y_list

    def _load_ge_features(self, data_df, ge_dim=978):
        """Return (N, ge_dim) float32 tensor; NaN rows for compounds without GE data."""
        raw_dir = osp.join(self.folder, "raw")
        ge_csv  = osp.join(raw_dir, "GE.csv.gz")
        ge_npz  = osp.join(raw_dir, "GE_feature.npz")
        if not (osp.exists(ge_csv) and osp.exists(ge_npz)):
            return None

        ge_index = pd.read_csv(ge_csv)
        ge_matrix = np.load(ge_npz)["data"].astype(np.float32)  # (631, 978)

        # inchikey → list of row indices in ge_matrix (positional, multiple cell lines possible)
        key_to_rows = {}
        for i, row in ge_index.iterrows():
            key_to_rows.setdefault(row["inchikey"], []).append(i)

        N = len(data_df)
        ge_out = np.full((N, ge_dim), np.nan, dtype=np.float32)
        for i, (_, row) in enumerate(data_df.iterrows()):
            key = row.get("inchikey", None)
            if key is not None and key in key_to_rows:
                rows = key_to_rows[key]
                ge_out[i] = ge_matrix[rows].mean(axis=0)

        return torch.tensor(ge_out)

    def _load_cp_features(self, data_df, dim=978):
        """Return (N, cp_dim) float32 tensor; NaN rows for compounds without CP data."""
        # TODO Fix dimensions and file name
        raw_dir = osp.join(self.folder, "raw")
        csv = osp.join(raw_dir, "CP-JUMP.csv.gz")
        npz = osp.join(raw_dir, "CP_feature.npz")
        if not (osp.exists(csv) and osp.exists(npz)):
            return None

        index = pd.read_csv(csv)
        matrix = np.load(npz)["data"].astype(np.float32)  # (631, 978)

        # inchikey → list of row indices in matrix (positional, multiple cell lines possible)
        key_to_rows = {}
        for i, row in index.iterrows():
            key_to_rows.setdefault(row["inchikey"], []).append(i)

        N = len(data_df)
        out = np.full((N, dim), np.nan, dtype=np.float32)
        for i, (_, row) in enumerate(data_df.iterrows()):
            key = row.get("inchikey", None)
            if key is not None and key in key_to_rows:
                rows = key_to_rows[key]
                out[i] = matrix[rows].mean(axis=0)

        return torch.tensor(out)

    def __getitem__(self, idx):
        batch = dict()

        batch['data'] = self.data[idx]
        batch['labels'] = self.labels[idx]
        if hasattr(self, "fingerprints") and self.fingerprints is not None:
            batch['fingerprints'] = self.fingerprints[idx]
        if hasattr(self, "ge_features") and self.ge_features is not None:
            batch['ge_features'] = self.ge_features[idx]
        if hasattr(self, "cp_features") and self.cp_features is not None:
            batch['cp_features'] = self.cp_features[idx]
        return batch

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, len(self))
