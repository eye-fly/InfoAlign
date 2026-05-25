"""
Encoder pretraining + classification head evaluation on ChEMBL2K.

Usage:
    # Pretrain encoder then evaluate frozen representations
    python train_encoder.py --dataset finetune-chembl2k --pretrain-epochs 60 --finetune-epochs 100

    # Skip pretraining (random encoder baseline)
    python train_encoder.py --dataset finetune-chembl2k --pretrain-epochs 0 --finetune-epochs 100

    # Pretrain + end-to-end finetune (unfreeze encoder)
    python train_encoder.py --dataset finetune-chembl2k --pretrain-epochs 60 --finetune-epochs 100 --no-freeze
"""
from datetime import datetime
import sys
import warnings

import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)

import os
import argparse

import torch
from torch.utils.data import DataLoader, Subset

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from configures.arguments import get_args
from dataset.create_datasets import get_data
from dataset.pretrain_smiles import PretrainSMILESDataset
from models.encoder import Encoder
from models.decoder import FingerprintDecoder, GEDecoder, SMILESDecoder, CPDecoder
from models.classification_head import ClassificationHead
from utils.train_funcs import unwrap, pretrain, joint_train, finetune
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')


def load_smiles_pretrain_dataset(args, cli, local_rank):
    # finetune_smiles = pd.read_csv("raw_data/chembl2k/raw/assays.csv.gz")["smiles"].tolist()
    pretrain_ds = PretrainSMILESDataset(root="./raw_data", n_augmentations=cli.n_augmentations)

    chembl_cache = "./raw_data/chembl2k/processed/processed_smiles_L128.pt"
    if os.path.exists(chembl_cache):
        try:
            _, _, cached_vocab = torch.load(chembl_cache, weights_only=False)
            # Only invalidate the old local cache if it doesn't match the pretrain vocab length
            if len(cached_vocab) != len(pretrain_ds.vocab):
                if local_rank <= 0:
                    print("Vocab size changed — invalidating old local ChEMBL2K cache to align with pretrained vocab!")
                os.remove(chembl_cache)
        except Exception:
            pass
    args._vocab_override = pretrain_ds.vocab
    return pretrain_ds


def prepare_encoder(cli, dataset, device, local_rank):
    encoder = Encoder(
        V=512, dx=None, d=256, num_heads=8, num_layers=6, K_layers=[1, 3, 5],
        gamma_teacher=0.95, gamma_codebook=0.99, mask_prob=0.15,
        vocab_size=dataset.vocab_size, mask_token_id=dataset.mask_token_id, pad_token_id=dataset.pad_token_id,
    ).to(device)

    if dist.is_initialized():
        encoder = DDP(encoder, device_ids=[local_rank], output_device=local_rank)

    return encoder


def prepare_decoders(cli, dataset, device, local_rank):
    fp_decoder = FingerprintDecoder(d=256).to(device) if cli.with_fp_decoder else None
    ge_decoder = GEDecoder(d=256, out_dim=dataset.ge_features.shape[1]).to(device) if cli.with_ge_decoder else None
    cp_decoder = CPDecoder(d=256, out_dim=dataset.cp_features.shape[1]).to(device) if cli.with_cp_decoder else None
    smiles_decoder = SMILESDecoder(d=256, vocab_size=dataset.vocab_size).to(device) if cli.with_smiles_decoder else None
    head = ClassificationHead(d=256, head_type=cli.head_type, num_tasks=dataset.num_tasks).to(device)

    if dist.is_initialized():
        if fp_decoder is not None: fp_decoder = DDP(fp_decoder, device_ids=[local_rank], output_device=local_rank)
        if ge_decoder is not None: ge_decoder = DDP(ge_decoder, device_ids=[local_rank], output_device=local_rank)
        if cp_decoder is not None: cp_decoder = DDP(cp_decoder, device_ids=[local_rank], output_device=local_rank)
        if smiles_decoder is not None: smiles_decoder = DDP(smiles_decoder, device_ids=[local_rank],
                                                            output_device=local_rank)
        head = DDP(head, device_ids=[local_rank], output_device=local_rank)

    return {
        'fp_decoder': fp_decoder,
        'ge_decoder': ge_decoder,
        'cp_decoder': cp_decoder,
        'smiles_decoder': smiles_decoder,
        'head': head
    }


def prepare_finetune_dataloaders(args, local_rank):
    if local_rank > 0 and dist.is_initialized():
        dist.barrier()
    dataset = get_data(args.dataset,  args.n_augmentations, "./raw_data", transform="smiles")
    if local_rank == 0 and dist.is_initialized():
        dist.barrier()

    split   = dataset.get_idx_split()

    args.num_trained = len(split["train"])
    args.task_type   = "classification"

    train_sub = Subset(dataset, split["train"])
    valid_sub = Subset(dataset, split["valid"])
    test_sub  = Subset(dataset, split["test"])

    world_size = dist.get_world_size() if dist.is_initialized() else 1
    per_gpu_batch = max(1, args.batch_size // world_size)

    train_sampler = DistributedSampler(train_sub, shuffle=True) if dist.is_initialized() else None
    valid_sampler = DistributedSampler(valid_sub, shuffle=False) if dist.is_initialized() else None
    test_sampler  = DistributedSampler(test_sub, shuffle=False) if dist.is_initialized() else None

    train_loader = DataLoader(train_sub, batch_size=per_gpu_batch, shuffle=(train_sampler is None),
                              sampler=train_sampler, num_workers=args.num_workers)
    valid_loader = DataLoader(valid_sub, batch_size=per_gpu_batch, shuffle=False, sampler=valid_sampler,
                              num_workers=args.num_workers)
    test_loader = DataLoader(test_sub, batch_size=per_gpu_batch, shuffle=False, sampler=test_sampler,
                             num_workers=args.num_workers)
    return train_loader, valid_loader, test_loader


def get_datasets(args, cli, local_rank=None):
    smiles_pretrain_dataset = None
    pretrain_dataset = None

    if local_rank > 0 and dist.is_initialized():
        dist.barrier()

    # Get smiles_pretrain_dataset
    need_pretrain_vocab = cli.pretrain_on_pretrain_raw or (cli.load_pretrained is not None)
    if need_pretrain_vocab:
        smiles_pretrain_dataset = load_smiles_pretrain_dataset(args, cli, local_rank)
    vocab = None

    # Get pretrain dataset
    if args.pretrain_dataset is not None:
        if smiles_pretrain_dataset is not None:
            vocab = smiles_pretrain_dataset.vocab
        pretrain_dataset = get_data(args.pretrain_dataset, args.n_augmentations, vocab, "./raw_data",
                                    transform="smiles")

    # Get finetune dataset
    if args.pretrain_dataset is not None:
        vocab = pretrain_dataset.vocab
    if smiles_pretrain_dataset is not None:
        vocab = smiles_pretrain_dataset.vocab
    dataset = get_data(args.dataset, args.n_augmentations, vocab, "./raw_data", transform="smiles")

    if local_rank == 0 and dist.is_initialized():
        dist.barrier()

    return smiles_pretrain_dataset, pretrain_dataset, dataset


def get_loaders(args, cli, smiles_pretrain_dataset, pretrain_dataset, dataset):
    smiles_pretrain_loader = None

    split = dataset.get_idx_split()
    args.num_trained = len(split["train"])
    args.task_type = "classification"
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    per_gpu_batch = max(1, cli.batch_size // world_size)

    train_sub = Subset(dataset, split["train"])
    valid_sub = Subset(dataset, split["valid"])
    test_sub = Subset(dataset, split["test"])

    train_sampler = DistributedSampler(train_sub, shuffle=True) if dist.is_initialized() else None
    valid_sampler = DistributedSampler(valid_sub, shuffle=False) if dist.is_initialized() else None
    test_sampler = DistributedSampler(test_sub, shuffle=False) if dist.is_initialized() else None

    train_loader = DataLoader(train_sub, batch_size=per_gpu_batch, shuffle=(train_sampler is None),
                              sampler=train_sampler, num_workers=cli.num_workers)
    valid_loader = DataLoader(valid_sub, batch_size=per_gpu_batch, shuffle=False, sampler=valid_sampler,
                              num_workers=cli.num_workers)
    test_loader = DataLoader(test_sub, batch_size=per_gpu_batch, shuffle=False, sampler=test_sampler,
                             num_workers=cli.num_workers)

    if pretrain_dataset is None:
        pretrain_loader = train_loader
    else:
        pretrain_sampler = DistributedSampler(pretrain_dataset, shuffle=True) if dist.is_initialized() else None
        pretrain_loader = DataLoader(pretrain_dataset, batch_size=per_gpu_batch, shuffle=(pretrain_sampler is None),
                                     sampler=pretrain_sampler, num_workers=args.num_workers)

    if smiles_pretrain_dataset is not None:
        smiles_pretrain_sampler = DistributedSampler(smiles_pretrain_dataset,
                                                     shuffle=True) if dist.is_initialized() else None
        smiles_pretrain_loader = DataLoader(smiles_pretrain_dataset, batch_size=per_gpu_batch,
                                            shuffle=(smiles_pretrain_sampler is None), sampler=smiles_pretrain_sampler,
                                            num_workers=cli.num_workers)

    return smiles_pretrain_loader, pretrain_loader, train_loader, test_loader, valid_loader


def main():
    parser = argparse.ArgumentParser(description="InfoAlign: Multimodal Pretraining & Finetuning Pipeline")
    parser.add_argument("--dataset",         default="finetune-chembl2k", help="Name of the finetuning dataset (e.g., finetune-chembl2k, broad6k)")
    parser.add_argument("--pretrain-epochs", type=int,   default=60, help="Number of epochs to run the unsupervised pretraining phase")
    parser.add_argument("--finetune-epochs", type=int,   default=100, help="Number of epochs to run the supervised finetuning (classification head) phase")
    parser.add_argument("--batch-size",      type=int,   default=256, help="Global batch size across all GPUs (will be divided by world_size in DDP)")
    parser.add_argument("--lr",              type=float, default=1e-3, help="Peak learning rate for the Adam optimizer (cosine annealed)")
    parser.add_argument("--wdecay",          type=float, default=1e-5, help="Weight decay for regularization")
    parser.add_argument("--cpu-training",    action='store_true', help="Training on cpu")
    parser.add_argument("--gpu-id",          type=int,   default=0, help="ID of the GPU to use when not running under torchrun/DDP")
    parser.add_argument("--num-workers",     type=int,   default=0, help="Number of dataloader workers (keep 0 if hitting IPC memory issues)")
    parser.add_argument("--no-freeze",       action="store_true", help="Unfreeze the encoder during finetuning (end-to-end training). Default is frozen.")
    parser.add_argument("--with-fp-decoder",    action="store_true", help="Enable the Fingerprint decoder module during encoder training")
    parser.add_argument("--with-ge-decoder",     action="store_true", help="Enable the LINCS L1000 Gene Expression regression decoder")
    parser.add_argument("--with-cp-decoder",     action="store_true", help="Enable the CP Jump cell profile regression decoder")
    parser.add_argument("--with-smiles-decoder", action="store_true", help="Enable BERT-style Masked Language Modeling (SMILES reconstruction)")
    parser.add_argument("--joint",               action="store_true", help="Skipping staged pretraining: train Encoder, Decoders, and Classification Head all at once from scratch")
    parser.add_argument("--pretrain-on-pretrain-raw", action="store_true", help="Use the massive HuggingFace 1.5M ChEMBL dataset for purely unsupervised pretraining before finetuning")
    parser.add_argument("--save-pretrained",  default=None, help="Filepath (e.g., 'model.pt') to snapshot the encoder weights after the pretraining phase concludes")
    parser.add_argument("--load-pretrained",  default=None, help="Filepath to load pre-existing encoder weights and skip pretraining entirely (triggers cache vocabulary invalidation)")
    parser.add_argument("--no-print",        action="store_true", help="Suppress tqdm progress bars and standard output (useful for non-interactive cluster jobs)")
    parser.add_argument("--subset-ratio",    type=float, default=1.0, help="Fraction of the dataset to use for training (useful for quick debugging, e.g., 0.1)")
    parser.add_argument("--mask-prob-start", type=float, default=0.15, help="Starting probability for token masking curriculum (if SMILES decoder used)")
    parser.add_argument("--mask-prob-end",   type=float, default=0.15, help="Ending probability for token masking curriculum (linearly interpolated over epochs)")
    parser.add_argument("--head-type",       type=str,   default="small", choices=["small", "wide", "deep"], help="Architecture volume of the classification MLPs built on top of the encoder")
    parser.add_argument("--n-augmentations", type=int,   default=0, help="Number of SMILES enumerations per molecule")
    parser.add_argument("--pretrain-dataset", default=None, help="Dataset name to be used as a separate pretraining dataset (if none specified then finetune-dataset is used")
    cli = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if local_rank != -1:
        dist.init_process_group(backend="nccl" if dist.is_nccl_available() else "gloo")
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        cli.gpu_id = local_rank
    else:
        device = torch.device(f"cuda:{cli.gpu_id}" if torch.cuda.is_available() and not cli.cpu_training else "cpu")

    # Reuse get_args for dataset-level settings (eval metric, num_tasks etc.)
    args = get_args.__wrapped__() if hasattr(get_args, "__wrapped__") else argparse.Namespace(
        dataset=cli.dataset, batch_size=cli.batch_size, lr=cli.lr, wdecay=cli.wdecay,
        gpu_id=cli.gpu_id, num_workers=cli.num_workers, no_print=True, subset_ratio=cli.subset_ratio,
    )
    args.dataset     = cli.dataset
    args.pretrain_dataset = cli.pretrain_dataset
    args.batch_size  = cli.batch_size
    args.lr          = cli.lr
    args.wdecay      = cli.wdecay
    args.num_workers = cli.num_workers
    args.no_print    = cli.no_print or (local_rank > 0)
    args.subset_ratio = cli.subset_ratio
    args.mask_prob_start = cli.mask_prob_start
    args.mask_prob_end   = cli.mask_prob_end
    args.head_type   = cli.head_type
    args.n_augmentations = cli.n_augmentations
    args.device      = device
    args.gpu_id      = cli.gpu_id
    args.num_workers = cli.num_workers

    torch.manual_seed(0)

    # Datasets for different stages of training
    smiles_pretrain_dataset, pretrain_dataset, dataset = get_datasets(args, cli, local_rank)

    # Loaders
    smiles_pretrain_loader, pretrain_loader, train_loader, test_loader, valid_loader = (
        get_loaders(args, cli, smiles_pretrain_dataset, pretrain_dataset, dataset))

    ### GET ENCODER AND DECODERS
    encoder = prepare_encoder(cli, dataset, device, local_rank)
    decoders = prepare_decoders(cli, dataset, device, local_rank)
    decoders_str = ", ".join(name for name, obj in decoders.items() if obj is not None)
    fp_decoder = decoders['fp_decoder']
    ge_decoder = decoders['ge_decoder']
    cp_decoder = decoders['cp_decoder']
    smiles_decoder = decoders['smiles_decoder']
    head = decoders['head']

    print(f"Device: {device}")
    split = dataset.get_idx_split()
    print(f"Train/valid/test: {len(split['train'])}/{len(split['valid'])}/{len(split['test'])}")
    print(f"Pretrain epochs: {cli.pretrain_epochs}  |  Finetune epochs: {cli.finetune_epochs}")
    mode = "joint" if cli.joint else ("frozen" if not cli.no_freeze else "e2e")
    print(f"Decoders: {decoders_str}  |  Mode: {mode}  |  Head: {cli.head_type}")

    if cli.load_pretrained:
        unwrap(encoder).load_state_dict(torch.load(cli.load_pretrained, map_location=device))
        print(f"Loaded pretrained encoder from {cli.load_pretrained}")

    # FIRST PRETRAINING ON LARGE DATASET OF SMILES
    if cli.pretrain_on_pretrain_raw:
        pretrain(encoder, fp_decoder, ge_decoder, cp_decoder, smiles_decoder, smiles_pretrain_loader, args,
                 cli.pretrain_epochs)
        if cli.save_pretrained:
            os.makedirs(os.path.dirname(cli.save_pretrained) or ".", exist_ok=True)
            if local_rank <= 0:
                torch.save(unwrap(encoder).state_dict(), cli.save_pretrained)
            print(f"Saved pretrained encoder to {cli.save_pretrained}")

    print(fp_decoder)
    if cli.joint:
        best_valid, best_test = joint_train(
            encoder,
            train_loader, valid_loader, test_loader,
            args, cli.finetune_epochs,
            head, ge_decoder, cp_decoder, smiles_decoder
        )
    else:
        # SECOND PRETRAINING ON ALL MODALITIES
        if cli.pretrain_epochs > 0:
            pretrain(
                encoder,
                pretrain_loader,
                args, cli.pretrain_epochs,
                fp_decoder, ge_decoder, cp_decoder, smiles_decoder)

        best_valid, best_test = finetune(
            encoder, not cli.no_freeze,
            train_loader, valid_loader, test_loader,
            args, cli.finetune_epochs,
            head
        )

    if local_rank <= 0:
        os.makedirs("results", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        tag = (f"{timestamp}_"
               f"{'joint' if cli.joint else f'pre{cli.pretrain_epochs}'}_"
               f"ft{cli.finetune_epochs}_"
               f"{'frozen' if not cli.no_freeze else 'e2e'}_"
               f"{'ge' if ge_decoder else 'noge'}_"
               f"{'cp' if cp_decoder else 'nocp'}_"
               f"{cli.head_type}")
        with open(f"results/{tag}.txt", "w") as f:
            f.write(" ".join(sys.argv))
            f.write("\n")
            f.write(f"valid={best_valid:.4f}  test={best_test:.4f}\n")
        print(f"\nSaved to results/{tag}.txt")


if __name__ == "__main__":
    main()
