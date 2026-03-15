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

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import math
import os
import argparse

import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import roc_auc_score

from configures.arguments import get_args
from dataset.create_datasets import get_data
from models.encoder import Encoder
from models.decoder import FingerprintDecoder
from models.classification_head import ClassificationHead
from utils.train_funcs import train_one_epoch_only_encoder


def get_cosine_schedule(optimizer, total_steps):
    def lr(step):
        p = step / max(1, total_steps)
        return max(0, math.cos(math.pi * 7 / 16 * p))
    return LambdaLR(optimizer, lr)


def roc_auc_eval(encoder, head, loader, device):
    encoder.eval()
    head.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch in loader:
            data = batch[0].to(device)
            targets = batch[-1].to(device, dtype=torch.float32)
            logits = head(encoder(data))
            preds.append(torch.sigmoid(logits).cpu())
            trues.append(targets.cpu())
    preds = torch.cat(preds).numpy()
    trues = torch.cat(trues).numpy()
    scores = []
    for i in range(trues.shape[1]):
        mask = ~np.isnan(trues[:, i])
        if trues[mask, i].sum() > 0 and (1 - trues[mask, i]).sum() > 0:
            scores.append(roc_auc_score(trues[mask, i], preds[mask, i]))
    return float(np.mean(scores))


def pretrain(encoder, decoder, train_loader, args, epochs, device):
    params = list(encoder.student_params())
    if decoder is not None:
        params += list(decoder.parameters())
    optimizer = optim.Adam(params, lr=args.lr, weight_decay=args.wdecay)
    scheduler = get_cosine_schedule(optimizer, epochs * args.steps)
    train_loaders = {"train_iter": iter(train_loader), "train_loader": train_loader}

    print(f"\n{'='*50}")
    print(f"Pretraining encoder for {epochs} epochs")
    print(f"{'='*50}")
    for epoch in range(epochs):
        train_loaders, loss = train_one_epoch_only_encoder(
            args, encoder, train_loaders, optimizer, scheduler, epoch, decoder=decoder
        )
        stats = encoder.codebook_stats()
        ent = np.mean([s["entropy"] for s in stats.values()])
        print(f"  epoch {epoch+1:>3}/{epochs}  loss={loss:.4f}  codebook_entropy={ent:.3f}")


def finetune(encoder, head, train_loader, valid_loader, test_loader, args, epochs, freeze_encoder, device):
    if freeze_encoder:
        for p in encoder.parameters():
            p.requires_grad = False
        params = head.parameters()
    else:
        for p in encoder.parameters():
            p.requires_grad = True
        params = list(head.parameters()) + list(encoder.parameters())

    optimizer = optim.Adam(params, lr=args.lr, weight_decay=args.wdecay)
    scheduler = get_cosine_schedule(optimizer, epochs * args.steps)

    mode = "frozen encoder" if freeze_encoder else "end-to-end"
    print(f"\n{'='*50}")
    print(f"Finetuning classification head ({mode}) for {epochs} epochs")
    print(f"{'='*50}")

    best_valid, best_test, best_epoch = 0.0, 0.0, 0
    for epoch in range(epochs):
        encoder.train() if not freeze_encoder else encoder.eval()
        head.train()
        for batch in train_loader:
            data = batch[0].to(device)
            targets = batch[-1].to(device, dtype=torch.float32)
            optimizer.zero_grad()
            head.loss(encoder(data), targets).backward()
            optimizer.step()
            scheduler.step()

        valid_auc = roc_auc_eval(encoder, head, valid_loader, device)
        if valid_auc > best_valid:
            best_valid = valid_auc
            best_test = roc_auc_eval(encoder, head, test_loader, device)
            best_epoch = epoch + 1

        if (epoch + 1) % 10 == 0:
            print(f"  epoch {epoch+1:>3}/{epochs}  valid={valid_auc:.4f}  best_valid={best_valid:.4f}  best_test={best_test:.4f}")

    print(f"\n  Best epoch {best_epoch}: valid={best_valid:.4f}  test={best_test:.4f}")
    return best_valid, best_test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",         default="finetune-chembl2k")
    parser.add_argument("--pretrain-epochs", type=int,   default=60)
    parser.add_argument("--finetune-epochs", type=int,   default=100)
    parser.add_argument("--batch-size",      type=int,   default=256)
    parser.add_argument("--lr",              type=float, default=1e-3)
    parser.add_argument("--wdecay",          type=float, default=1e-5)
    parser.add_argument("--gpu-id",          type=int,   default=0)
    parser.add_argument("--num-workers",     type=int,   default=0)
    parser.add_argument("--no-freeze",       action="store_true", help="finetune encoder end-to-end")
    parser.add_argument("--with-decoder",    action="store_true", help="use fingerprint decoder during pretraining")
    parser.add_argument("--no-print",        action="store_true")
    parser.add_argument("--subset-ratio",    type=float, default=1.0)
    cli = parser.parse_args()

    device = torch.device(f"cuda:{cli.gpu_id}" if torch.cuda.is_available() else "cpu")

    # Reuse get_args for dataset-level settings (eval metric, num_tasks etc.)
    args = get_args.__wrapped__() if hasattr(get_args, "__wrapped__") else argparse.Namespace(
        dataset=cli.dataset, batch_size=cli.batch_size, lr=cli.lr, wdecay=cli.wdecay,
        gpu_id=cli.gpu_id, num_workers=cli.num_workers, no_print=True, subset_ratio=cli.subset_ratio,
    )
    args.dataset     = cli.dataset
    args.batch_size  = cli.batch_size
    args.lr          = cli.lr
    args.wdecay      = cli.wdecay
    args.num_workers = cli.num_workers
    args.no_print    = True
    args.subset_ratio = cli.subset_ratio
    args.device      = device
    args.gpu_id      = cli.gpu_id

    torch.manual_seed(0)
    dataset = get_data(args, "./raw_data", transform="smiles")
    split   = dataset.get_idx_split()

    args.num_trained = len(split["train"])
    args.task_type   = "classification"
    args.steps       = args.num_trained // args.batch_size + 1

    train_loader = DataLoader(Subset(dataset, split["train"]), batch_size=cli.batch_size, shuffle=True,  num_workers=cli.num_workers)
    valid_loader = DataLoader(Subset(dataset, split["valid"]), batch_size=cli.batch_size, shuffle=False, num_workers=cli.num_workers)
    test_loader  = DataLoader(Subset(dataset, split["test"]),  batch_size=cli.batch_size, shuffle=False, num_workers=cli.num_workers)

    encoder = Encoder(
        V=512, dx=None, d=256, num_heads=8, num_layers=6, K_layers=[1, 3, 5],
        gamma_teacher=0.95, gamma_codebook=0.99, mask_prob=0.15,
        vocab_size=dataset.vocab_size, mask_token_id=dataset.mask_token_id, pad_token_id=dataset.pad_token_id,
    ).to(device)

    decoder = FingerprintDecoder(d=256).to(device) if cli.with_decoder else None
    head    = ClassificationHead(d=256, num_tasks=dataset.num_tasks).to(device)

    print(f"Device: {device}")
    print(f"Train/valid/test: {len(split['train'])}/{len(split['valid'])}/{len(split['test'])}")
    print(f"Pretrain epochs: {cli.pretrain_epochs}  |  Finetune epochs: {cli.finetune_epochs}")
    print(f"Decoder: {'FingerprintDecoder' if decoder else 'none'}  |  Freeze encoder: {not cli.no_freeze}")

    if cli.pretrain_epochs > 0:
        pretrain(encoder, decoder, train_loader, args, cli.pretrain_epochs, device)

    best_valid, best_test = finetune(
        encoder, head, train_loader, valid_loader, test_loader,
        args, cli.finetune_epochs, freeze_encoder=not cli.no_freeze, device=device,
    )

    os.makedirs("results", exist_ok=True)
    tag = f"pre{cli.pretrain_epochs}_ft{cli.finetune_epochs}_{'frozen' if not cli.no_freeze else 'e2e'}_{'dec' if decoder else 'nodec'}"
    with open(f"results/{tag}.txt", "w") as f:
        f.write(f"valid={best_valid:.4f}  test={best_test:.4f}\n")
    print(f"\nSaved to results/{tag}.txt")


if __name__ == "__main__":
    main()
