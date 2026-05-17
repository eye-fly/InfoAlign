"""
Script to reformat the mayaanlab data available at https://maayanlab.net/SEP-L1000
to the format used by InfoAlign and our project.
"""

MAYAANLAB_DATA_PATH = 'raw_data/mayaanlab'


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
    pass


if __name__ == '__main__':
    main()
