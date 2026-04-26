## TODO:
- Test existing training
- Find all files to change to add decoder for cell profiles
- Add decoder for cell profiles
- Find dataset for cell painting
- Find dataset for gene expression.
- Debug training locally
- Train on the resources given to us

Files to change:
- train_funcs
- 

TODO in future:
- Rewrite train_one_epoch to allow for arbitrary collection of decoders instead of named arguments

## Setup

```bash
conda create --name infoalign python=3.11.7
conda activate infoalign
pip install torch
pip install -r requirements.txt
```

## Run

Example training on finetune-chembl2k.
```bash
python train.py --dataset finetune-chembl2k --pretrain-epochs 60 --finetune-epochs 100
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
