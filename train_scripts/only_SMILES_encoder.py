"""
Encoder pretraining + classification head evaluation on ChEMBL2K.
Decoder on SMILES training.

Usage:
    # Pretrain encoder then evaluate frozen representations
    python only_SMILES_encoder.py --dataset finetune-chembl2k --pretrain-epochs 60 --finetune-epochs 100

    # Skip pretraining (random encoder baseline)
    python only_SMILES_encoder.py --dataset finetune-chembl2k --pretrain-epochs 0 --finetune-epochs 100

    # Pretrain + end-to-end finetune (unfreeze encoder)
    python only_SMILES_encoder.py --dataset finetune-chembl2k --pretrain-epochs 60 --finetune-epochs 100 --no-freeze
"""

import warnings


warnings.filterwarnings("ignore", category=UserWarning)

import os

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import roc_auc_score

from dataset.create_datasets import get_data
from models.encoder import Encoder
from models.decoder import SMILESDecoder
from models.classification_head import ClassificationHead
from utils.train_funcs import train_one_epoch, get_cosine_schedule_with_warmup, parse_arguments


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


def pretrain(encoder, decoder, train_loader, args):
    params = list(encoder.student_params())
    if decoder is not None:
        params += list(decoder.parameters())
    optimizer = optim.Adam(params, lr=args.lr, weight_decay=args.wdecay)
    scheduler = get_cosine_schedule_with_warmup(optimizer, 0, args.epochs * args.steps)
    train_loaders = {"train_iter": iter(train_loader), "train_loader": train_loader}

    print(f"\n{'='*50}")
    print(f"Pretraining encoder for {args.epochs} epochs")
    print(f"{'='*50}")
    for epoch in range(args.epochs):
        train_loaders, loss = train_one_epoch(
            args, encoder, train_loaders, optimizer, scheduler, epoch, decoder=decoder
        )
        stats = encoder.codebook_stats()
        ent = np.mean([s["entropy"] for s in stats.values()])
        print(f"  epoch {epoch+1:>3}/{args.epochs}  loss={loss:.4f}  codebook_entropy={ent:.3f}")


def finetune(encoder, head, train_loader, valid_loader, test_loader, args):
    if args.freeze:
        for p in encoder.parameters():
            p.requires_grad = False
        params = head.parameters()
    else:
        for p in encoder.parameters():
            p.requires_grad = True
        params = list(head.parameters()) + list(encoder.parameters())

    optimizer = optim.Adam(params, lr=args.lr, weight_decay=args.wdecay)
    scheduler = get_cosine_schedule_with_warmup(optimizer, 0, args.epochs * args.steps)

    mode = "frozen encoder" if args.freeze else "end-to-end"
    print(f"\n{'='*50}")
    print(f"Finetuning classification head ({mode}) for {args.epochs} epochs")
    print(f"{'='*50}")

    best_valid, best_test, best_epoch = 0.0, 0.0, 0
    for epoch in range(args.epochs):
        encoder.train() if not args.freeze else encoder.eval()
        head.train()
        for batch in train_loader:
            data = batch[0].to(args.device)
            targets = batch[-1].to(args.device, dtype=torch.float32)
            optimizer.zero_grad()
            head.loss(encoder(data), targets).backward()
            optimizer.step()
            scheduler.step()

        valid_auc = roc_auc_eval(encoder, head, valid_loader, args.device)
        if valid_auc > best_valid:
            best_valid = valid_auc
            best_test = roc_auc_eval(encoder, head, test_loader, args.device)
            best_epoch = epoch + 1

        if (epoch + 1) % 10 == 0:
            print(f"  epoch {epoch+1:>3}/{args.epochs}  valid={valid_auc:.4f}  best_valid={best_valid:.4f}  best_test={best_test:.4f}")

    print(f"\n  Best epoch {best_epoch}: valid={best_valid:.4f}  test={best_test:.4f}")
    return best_valid, best_test


def main():
    args = parse_arguments()

    torch.manual_seed(0)
    dataset = get_data(args, "../raw_data", transform="smiles")
    split   = dataset.get_idx_split()

    args.num_trained = len(split["train"])
    args.task_type   = "classification"
    args.steps       = args.num_trained // args.batch_size + 1

    train_loader = DataLoader(Subset(dataset, split["train"]), batch_size=args.batch_size, shuffle=True,  num_workers=args.num_workers)
    valid_loader = DataLoader(Subset(dataset, split["valid"]), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader  = DataLoader(Subset(dataset, split["test"]),  batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    encoder = Encoder(
        V=512, dx=None, d=256, num_heads=8, num_layers=6, K_layers=[1, 3, 5],
        gamma_teacher=0.95, gamma_codebook=0.99, mask_prob=0.15,
        vocab_size=dataset.vocab_size, mask_token_id=dataset.mask_token_id, pad_token_id=dataset.pad_token_id,
    ).to(args.device)

    decoder = SMILESDecoder(d=256, vocab_size=dataset.vocab_size, ).to(args.device) if args.with_decoder else None
    head    = ClassificationHead(d=256, num_tasks=dataset.num_tasks).to(args.device)

    print(f"Device: {args.device}")
    print(f"Train/valid/test: {len(split['train'])}/{len(split['valid'])}/{len(split['test'])}")
    print(f"Pretrain epochs: {args.pretrain_epochs}  |  Finetune epochs: {args.finetune_epochs}")
    print(f"Decoder: {'SMILESDecoder' if decoder else 'none'}  |  Freeze encoder: {args.freeze}")

    if args.pretrain_epochs > 0:
        pretrain(encoder, decoder, train_loader, args)

    best_valid, best_test = finetune(
        encoder, head, train_loader, valid_loader, test_loader,
        args, args.finetune_epochs, freeze_encoder=args.freeze, device=args.device,
    )

    os.makedirs("../results", exist_ok=True)
    tag = f"pre{args.pretrain_epochs}_ft{args.finetune_epochs}_{'frozen' if args.freeze else 'e2e'}_{'dec' if decoder else 'nodec'}"
    with open(f"results/{tag}.txt", "w") as f:
        f.write(f"valid={best_valid:.4f}  test={best_test:.4f}\n")
    print(f"\nSaved to results/{tag}.txt")


if __name__ == "__main__":
    main()
