import pandas as pd

# ge = pd.read_csv("LINCS_Gene_Experssion_signatures_CD.csv.gz")
# cp = pd.read_csv("MLPCN_morplological_profiles.csv.gz")
smiles = pd.read_csv("meta_SMILES.csv")
# smiles = smiles[["pert_id", "SMILES"]]

df = smiles
# połączenie tylko wspólnych cząsteczek
# df = ge.merge(cp, on="pert_id", how="inner")
# df = df.merge(smiles, on="pert_id", how="inner")

# usunięcie wszystkich wierszy zawierających NaN
# cols = list(smiles.columns) + [c for c in df.columns if c not in smiles.columns]
# df = df[cols]

df = df.dropna()

df.to_csv("assays.csv.gz", index=False, compression="gzip")