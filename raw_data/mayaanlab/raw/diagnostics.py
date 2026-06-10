import pandas as pd

# To see columns:
# ge = pd.read_csv("LINCS_Gene_Experssion_signatures_CD.csv.gz", nrows=5)
# cp = pd.read_csv("MLPCN_morplological_profiles.csv.gz", nrows=5)
# smiles = pd.read_csv("meta_SMILES.csv")

# print("GE:", ge.columns.tolist())
# print("CP:", cp.columns.tolist())
# print("SMILES:", smiles.columns.tolist())


df = pd.read_csv("meta_SMILES.csv")

print(df.dtypes[df.dtypes == "object"])

for c in df.columns:
    if df[c].dtype == object:
        print(c)