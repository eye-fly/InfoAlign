## Setup

```bash
conda create --name infoalign python=3.11.7
conda activate infoalign
pip install torch
pip install -r requirements.txt
```

## Run

Test command to check for glitches on finetune-chembl2k.
```bash
python train.py --dataset finetune-chembl2k --with-ge-decoder --with-cp-decoder --cpu-training --pretrain-epochs 1 --finetune-epochs 1
```

Test command for pretraining with mayaanlab data.
```bash
python train.py --pretrain-dataset mayaanlab --dataset finetune-chembl2k --with-ge-decoder --with-cp-decoder --cpu-training --pretrain-epochs 1 --finetune-epochs 1
```

Command for pretraining with mayaanlab data.
```bash
python train.py --pretrain-dataset mayaanlab --dataset finetune-chembl2k --with-ge-decoder --with-cp-decoder --pretrain-epochs 1 --finetune-epochs 1
```



The ChEMBL dataset is downloaded automatically from HuggingFace.
Morgan fingerprint encoding (2048-bit, radius=2) is triggered by `get_data()` — see `dataset/prediction_molecule.py:prepare_fingerprints()`.

The Gene Expression dataset is . It is downloaded from .

The Cell Profile dataset is . It is downloaded from .

### Key args

| Arg | Default | Description |
|-----|---------|-------------|
| `--dataset` | `finetune-chembl2k` | Dataset name |
| `--epochs` | `300` | Max training epochs |
| `--batch-size` | `5120` | Batch size |
| `--lr` | `1e-3` | Learning rate |
| `--patience` | `50` | Early stopping patience |
| `--drop-ratio` | `0.5` | Dropout ratio |
| `--gpu-id` | `0` | GPU device ID |
| `--no-print` | off | Disable progress bar |

## Project structure

```
main.py                        # Entry point (MLP training + eval)
models/mlp.py                  # Simple MLP model
dataset/create_datasets.py     # HuggingFace download + dataset factory
dataset/prediction_molecule.py # Fingerprint/SMILES dataset class
dataset/data_utils.py          # Scaffold splitting
utils/misc.py                  # Validation, metrics, prediction saving
utils/train_funcs.py           # Training loop
configures/arguments.py        # CLI arguments
configures/finetune.yaml       # Default config
```
