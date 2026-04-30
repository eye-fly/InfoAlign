from huggingface_hub import hf_hub_download, HfApi
import os

from .prediction_molecule import PredictionMoleculeDataset


def download_finetune_data(data_name, root_path):
    local_dir_raw = os.path.join(root_path, data_name, "raw")
    if os.path.exists(os.path.join(local_dir_raw, "assays.csv.gz")):
        print(f"Finetuning data for {data_name} already exists. Skipping download.")
        return

    print(f"Downloading finetuning data for {data_name}...")
    repo_id = "liuganghuggingface/InfoAlign-Data"

    os.makedirs(local_dir_raw, exist_ok=True)

    try:
        api = HfApi()
        all_files = api.list_repo_files(repo_id, repo_type="dataset")

        finetune_data_repo_prefix = f"finetune_raw/{data_name}/raw/"
        finetune_files = [f for f in all_files if f.startswith(finetune_data_repo_prefix)]

        for file_path_in_repo in finetune_files:
            filename = os.path.basename(file_path_in_repo)

            hf_hub_download(repo_id=repo_id,
                            filename=file_path_in_repo,
                            repo_type="dataset",
                            local_dir=local_dir_raw,
                            local_dir_use_symlinks=False)

            nested_path_to_file = os.path.join(local_dir_raw, finetune_data_repo_prefix, filename)
            target_file_path = os.path.join(local_dir_raw, filename)

            if os.path.exists(nested_path_to_file):
                os.rename(nested_path_to_file, target_file_path)
                try:
                    os.removedirs(os.path.join(local_dir_raw, finetune_data_repo_prefix))
                except OSError:
                    pass
            elif not os.path.exists(target_file_path):
                print(f"Warning: Downloaded file '{filename}' for '{data_name}' not found at expected location '{target_file_path}' or nested path '{nested_path_to_file}'.")

        print(f"Successfully downloaded {len(finetune_files)} files to {local_dir_raw}")
    except Exception as e:
        print(f"Error downloading dataset for {data_name}: {str(e)}")
        print("Please check your internet connection and ensure you have the necessary permissions.")
        print("If the issue persists, you may need to log in using `huggingface-cli login`")


def get_data(dataset, n_aug, load_path, transform="fingerprint"):
    assert transform in [
        "fingerprint",
        "smiles",
    ]


    if dataset.startswith("finetune"):
        data_name = dataset.split("-")[1]
        download_finetune_data(data_name, load_path)
    else:
        data_name = dataset

    return PredictionMoleculeDataset(
        name=data_name, root=load_path, transform=transform,
        n_augmentations=n_aug
    )

