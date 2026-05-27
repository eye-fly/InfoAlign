"""
Script to reformat the mayaanlab data available at https://maayanlab.net/SEP-L1000
to the format used by InfoAlign and our project.
"""
import os

import numpy as np
import pandas as pd
from rdkit import Chem

FOLDER_IN = 'raw_data/mayaanlab/original/'
SMILES_IN = 'meta_SMILES.csv'
GE_IN = 'LINCS_Gene_Experssion_signatures_CD.csv'
CP_IN = 'MLPCN_morplological_profiles.csv'

FOLDER_OUT = 'raw_data/mayaanlab/raw/'
SMILES_OUT = 'assays.csv.gz'
GE_NPZ_OUT = 'GE_feature.npz'
GE_CSV_OUT = 'GE.csv.gz'
CP_NPZ_OUT = 'CP-JUMP_feature.npz'
CP_CSV_OUT = 'CP-JUMP.csv.gz'


def main():
    """
    Expected format:
    Three files:
    - CP data with inchikey index as npz file
    - GE data with inchikey index as npz file
    - Smiles data with inchikey index as csv.gz
    Files are joined by inchikey.
    MayaanLab format:
    - Morphological profiles indexed by pert_id as csv
    - Gene expression data indexed by pert_id as csv
    - Smiles indexed by pert_id as csv
    This will reformat to proper files
    :return:
    """
    os.chdir('..')
    smiles = pd.read_csv(FOLDER_IN + SMILES_IN)
    gene_expression = pd.read_csv(FOLDER_IN + GE_IN)
    cell_profile = pd.read_csv(FOLDER_IN + CP_IN)

    print("=== Cleaning SMILES Data ===")
    initial_smiles_len = len(smiles)
    smiles_column = 'SMILES'

    # Remove completely null/NaN rows
    smiles = smiles.dropna(subset=[smiles_column])
    nan_removed = initial_smiles_len - len(smiles)
    print(f"-> Removed {nan_removed} NaN/Null rows.")

    # Ensure every entry is strictly a string
    prev_len = len(smiles)
    smiles = smiles[smiles[smiles_column].apply(lambda x: isinstance(x, str))]
    non_str_removed = prev_len - len(smiles)
    print(f"-> Removed {non_str_removed} non-string/numerical entries.")

    # Filter out chemically invalid SMILES that break RDKit
    def is_valid_smiles(smiles_string):
        if not smiles_string.strip():
            return False
        mol = Chem.MolFromSmiles(smiles_string)
        return mol is not None

    prev_len = len(smiles)
    smiles = smiles[smiles[smiles_column].apply(is_valid_smiles)]
    invalid_rdkit_removed = prev_len - len(smiles)
    print(f"-> Removed {invalid_rdkit_removed} chemically invalid RDKit SMILES.")
    print(f"Total SMILES rows remaining: {len(smiles)}\n")

    gene_expression = gene_expression.dropna()
    cell_profile = cell_profile.dropna()

    if not os.path.exists(FOLDER_OUT):
        os.makedirs(FOLDER_OUT)

    smiles.to_csv(FOLDER_OUT + SMILES_OUT, index=False, compression="gzip")
    gene_expression.to_csv(FOLDER_OUT + GE_CSV_OUT, index=False, compression="gzip")
    cell_profile.to_csv(FOLDER_OUT + CP_CSV_OUT, index=False, compression="gzip")


    gene_expression_array = gene_expression.drop(columns=["pert_id"]).values
    cell_profile_array = cell_profile.drop(columns=['pert_id']).values

    np.savez_compressed(FOLDER_OUT + GE_NPZ_OUT, data=gene_expression_array)
    np.savez_compressed(FOLDER_OUT + CP_NPZ_OUT, data=cell_profile_array)



    print("Conversion complete!")
    pass


if __name__ == '__main__':
    main()
