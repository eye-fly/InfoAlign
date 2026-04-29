import yaml
import argparse


def load_arguments_from_yaml(filename):
    with open(filename, "r") as file:
        config = yaml.safe_load(file)
    return config


def save_arguments_to_yaml(args, filename):
    with open(filename, "w") as f:
        yaml.dump(vars(args), f)


def get_args():
    parser = argparse.ArgumentParser(
        description="Transformer backbone for molecular property prediction"
    )
    parser.add_argument(
        "--gpu-id", type=int, default=0, help="which gpu to use if any (default: 0)"
    )
    parser.add_argument(
        "--num-workers", type=int, default=0, help="number of workers for data loader"
    )
    parser.add_argument(
        "--no-print", action="store_true", default=False, help="don't use progress bar"
    )

    parser.add_argument("--dataset", default="finetune-chembl2k", type=str, help="dataset name")

    parser.add_argument(
        "--drop-ratio", type=float, default=0.5, help="dropout ratio (default: 0.5)"
    )
    parser.add_argument(
        "--emb-dim",
        type=int,
        default=2048,
        help="input dimensionality — Morgan fingerprint size (default: 2048)",
    )
    parser.add_argument(
        "--subset-ratio",
        type=float,
        default=1.0,
        help="ratio of training data to use (default: 1.0)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5120,
        help="input batch size for training (default: 5120)",
    )
    parser.add_argument(
        "--lr",
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Learning rate (default: 1e-3)",
    )
    parser.add_argument("--wdecay", default=1e-5, type=float, help="weight decay")
    parser.add_argument(
        "--epochs", type=int, default=300, help="number of epochs to train"
    )
    parser.add_argument(
        "--initw-name",
        type=str,
        default="default",
        help="method to initialize the model parameters",
    )
    parser.add_argument(
        "--patience", type=int, default=50, help="patience for early stop"
    )
    parser.add_argument(
        "--n-augmentations", type=int, default=0,
        help="number of SMILES enumeration augmentations per molecule (default: 0 = none)",
    )

    args = parser.parse_args()

    return args
