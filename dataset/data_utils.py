from copy import deepcopy
from collections import defaultdict

import numpy as np
from joblib import Parallel, delayed
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


def get_scaffold(mol):
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    return Chem.MolToSmiles(scaffold)


def parallel_scaffold_computation(molecule, molecule_id):
    scaffold = get_scaffold(molecule)
    return scaffold, molecule, molecule_id


def cluster_molecules_by_scaffold(
    molecules, all_data_id, n_jobs=-1, remove_single=True, flatten_id=True
):
    paired_results = Parallel(n_jobs=n_jobs)(
        delayed(parallel_scaffold_computation)(mol, molecule_id)
        for mol, molecule_id in zip(molecules, all_data_id)
    )

    batch = defaultdict(list)
    batched_data_id = defaultdict(list)

    for scaffold, mol, molecule_id in paired_results:
        batch[scaffold].append(mol)
        batched_data_id[scaffold].append(molecule_id)

    if remove_single:
        batch = {scaffold: mols for scaffold, mols in batch.items() if len(mols) > 1}
        batched_data_id = {
            scaffold: ids for scaffold, ids in batched_data_id.items() if len(ids) > 1
        }

    scaffolds = list(batch.keys())
    batch = list(batch.values())
    batched_data_id = list(batched_data_id.values())
    if flatten_id:
        batched_data_id = [idd for batch in batched_data_id for idd in batch]
        batched_data_id = np.array(batched_data_id)

    return scaffolds, batch, batched_data_id


def scaffold_split(train_df, train_ratio=0.6, valid_ratio=0.15, test_ratio=0.25):
    train_smiles_list = train_df["smiles"]

    indinces = list(range(len(train_smiles_list)))
    train_mol_list = [Chem.MolFromSmiles(smiles) for smiles in train_smiles_list]
    scaffold_names, _, batched_id = cluster_molecules_by_scaffold(
        train_mol_list, indinces, remove_single=False, flatten_id=False
    )

    train_cutoff = int(train_ratio * len(train_df))
    valid_cutoff = int(valid_ratio * len(train_df)) + train_cutoff
    train_inds, valid_inds, test_inds = [], [], []
    inds_all = deepcopy(batched_id)
    np.random.seed(3)
    np.random.shuffle(inds_all)
    idx_count = 0
    for inds_list in inds_all:
        for ind in inds_list:
            if idx_count < train_cutoff:
                train_inds.append(ind)
            elif idx_count < valid_cutoff:
                valid_inds.append(ind)
            else:
                test_inds.append(ind)
            idx_count += 1

    return train_inds, valid_inds, test_inds
