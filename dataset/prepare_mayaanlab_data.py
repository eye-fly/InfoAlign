"""
Script to reformat the mayaanlab data available at https://maayanlab.net/SEP-L1000
to the format used by InfoAlign and our project.
"""
import os

import numpy as np
import pandas as pd

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

    smiles = smiles.dropna(subset=["SMILES"])
    smiles = smiles[smiles["SMILES"].apply(lambda x: isinstance(x, str))]

    gene_expression = gene_expression.dropna()
    cell_profile = cell_profile.dropna()

    if not os.path.exists(FOLDER_OUT):
        os.makedirs(FOLDER_OUT)

    smiles.to_csv(FOLDER_OUT + SMILES_OUT, index=False, compression="gzip")
    gene_expression.to_csv(FOLDER_OUT + GE_CSV_OUT, index=False, compression="gzip")
    cell_profile.to_csv(FOLDER_OUT + CP_CSV_OUT, index=False, compression="gzip")


    gene_expression_array = gene_expression.drop(columns=["pert_id"]).values
    cell_profile_array = cell_profile.drop(columns=['pert_id']).values

    np.savez_compressed(FOLDER_OUT + GE_NPZ_OUT, my_array=gene_expression_array)
    np.savez_compressed(FOLDER_OUT + CP_NPZ_OUT, my_array=cell_profile_array)



    print("Conversion complete!")
    pass


if __name__ == '__main__':
    main()
