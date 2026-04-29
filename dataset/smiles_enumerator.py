"""
SMILES enumeration for data augmentation.

Reference: Bjerrum, E.J. "SMILES Enumeration as Data Augmentation for
Neural Network Modeling of Molecules" (2017), arXiv:1703.07076
https://github.com/EBjerrum/SMILES-enumeration
"""

import numpy as np
from rdkit import Chem


def randomize_smiles(smiles: str, isomeric: bool = True) -> str:
    """Return a randomised SMILES string for the same molecule."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles
    atom_order = list(range(mol.GetNumAtoms()))
    np.random.shuffle(atom_order)
    mol = Chem.RenumberAtoms(mol, atom_order)
    return Chem.MolToSmiles(mol, canonical=False, isomericSmiles=isomeric)


def enumerate_smiles(smiles: str, n_augmentations: int, isomeric: bool = True):
    """Generate up to *n_augmentations* unique non-canonical SMILES.

    Returns a list of unique SMILES strings (excluding the canonical form).
    If fewer than *n_augmentations* unique forms can be found after
    ``n_augmentations * 10`` attempts, returns however many were found.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() < 2:
        return []

    canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=isomeric)
    seen = {canonical}
    results = []
    max_attempts = n_augmentations * 10
    for _ in range(max_attempts):
        if len(results) >= n_augmentations:
            break
        atom_order = list(range(mol.GetNumAtoms()))
        np.random.shuffle(atom_order)
        new_mol = Chem.RenumberAtoms(mol, atom_order)
        smi = Chem.MolToSmiles(new_mol, canonical=False, isomericSmiles=isomeric)
        if smi not in seen:
            seen.add(smi)
            results.append(smi)
    return results
