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
from dataset.pretrain_smiles import PretrainSMILESDataset
from models.encoder import Encoder
from models.decoder import FingerprintDecoder, GEDecoder, SMILESDecoder
from models.classification_head import ClassificationHead
from utils.train_funcs import train_one_epoch_only_encoder
from utils.misc import AverageMeter


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
    preds = torch.cat(preds)
    trues = torch.cat(trues)
    import torch.distributed as dist
    if dist.is_initialized():
        gathered_preds = [torch.zeros_like(preds) for _ in range(dist.get_world_size())]
        gathered_trues = [torch.zeros_like(trues) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_preds, preds)
        dist.all_gather(gathered_trues, trues)
        preds = torch.cat(gathered_preds)
        trues = torch.cat(gathered_trues)

    preds = preds.numpy()
    trues = trues.numpy()
    scores = []
    for i in range(trues.shape[1]):
        mask = ~np.isnan(trues[:, i])
        if trues[mask, i].sum() > 0 and (1 - trues[mask, i]).sum() > 0:
            scores.append(roc_auc_score(trues[mask, i], preds[mask, i]))
    return float(np.mean(scores))


def unwrap(model):
    return model.module if hasattr(model, "module") else model


def pretrain(encoder, decoder, ge_decoder, train_loader, args, epochs, device):
    if hasattr(train_loader.sampler, "set_epoch"):
        train_loader.sampler.set_epoch(0)
    params = list(unwrap(encoder).student_params())
    if decoder is not None:
        params += list(unwrap(decoder).parameters())
    if ge_decoder is not None:
        params += list(unwrap(ge_decoder).parameters())
    optimizer = optim.Adam(params, lr=args.lr, weight_decay=args.wdecay)
    scheduler = get_cosine_schedule(optimizer, epochs * args.steps)
    train_loaders = {"train_iter": iter(train_loader), "train_loader": train_loader}

    print(f"\n{'='*50}")
    print(f"Pretraining encoder for {epochs} epochs")
    print(f"{'='*50}")
    for epoch in range(epochs):
        if hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)

        progress = epoch / max(1, epochs - 1)
        current_mask = args.mask_prob_start + (args.mask_prob_end - args.mask_prob_start) * progress
        unwrap(encoder).mask_prob = current_mask

        train_loaders, loss, components = train_one_epoch_only_encoder(
            args, encoder, train_loaders, optimizer, scheduler, epoch,
            decoder=decoder, ge_decoder=ge_decoder,
        )
        stats = unwrap(encoder).codebook_stats()
        ent   = np.mean([s["entropy"] for s in stats.values()])
        act   = int(np.mean([s["active"]  for s in stats.values()]))
        comp_str = "  ".join(f"{k}={v:.3f}" for k, v in components.items())
        print(f"  epoch {epoch+1:>3}/{epochs}  total={loss:.3f}  [{comp_str}]  entropy={ent:.3f}  active={act}/512")


def joint_train(encoder, ge_decoder, smiles_decoder, head, train_loader, valid_loader, test_loader, args, epochs, device,
                ge_lambda=10.0, smiles_lambda=0.25, cls_lambda=1.0):
    params = list(unwrap(encoder).student_params()) + list(unwrap(head).parameters())
    if ge_decoder is not None:
        params += list(unwrap(ge_decoder).parameters())
    if smiles_decoder is not None:
        params += list(unwrap(smiles_decoder).parameters())
    optimizer = optim.Adam(params, lr=args.lr, weight_decay=args.wdecay)
    scheduler = get_cosine_schedule(optimizer, epochs * args.steps)

    print(f"\n{'='*50}")
    print(f"Joint training (encoder + GE decoder + head) for {epochs} epochs")
    print(f"{'='*50}")

    best_valid, best_test, best_epoch = 0.0, 0.0, 0
    for epoch in range(epochs):
        if hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)
            
        progress = epoch / max(1, epochs - 1)
        current_mask = args.mask_prob_start + (args.mask_prob_end - args.mask_prob_start) * progress
        unwrap(encoder).mask_prob = current_mask

        encoder.train()
        head.train()
        enc_m = AverageMeter(); cls_m = AverageMeter()
        ge_m  = AverageMeter(); smi_m = AverageMeter()

        for batch in train_loader:
            data, fingerprints, ge, targets = batch
            data    = data.to(device)
            ge      = ge.to(device, dtype=torch.float32)
            targets = targets.to(device, dtype=torch.float32)

            optimizer.zero_grad()
            use_smiles = smiles_decoder is not None and data.dtype == torch.long
            enc_loss, z_masked, mask, x_orig = encoder(data, return_loss=True, update_codebooks=True, return_masked_info=True)
            z = encoder(data)
            cls_loss = head(z, targets=targets)
            loss = enc_loss + cls_lambda * cls_loss
            enc_m.update(enc_loss.item()); cls_m.update(cls_loss.item())

            if ge_decoder is not None:
                ge_l = ge_decoder(z, ge_targets=ge)
                loss = loss + ge_lambda * ge_l
                ge_m.update(ge_l.item())
            if use_smiles:
                smi_l = smiles_decoder(z_masked, mask=mask, original_tokens=x_orig)
                loss = loss + smiles_lambda * smi_l
                smi_m.update(smi_l.item())

            loss.backward()
            optimizer.step()
            scheduler.step()
            unwrap(encoder).update_teacher()

        valid_auc = roc_auc_eval(encoder, head, valid_loader, device)
        if valid_auc > best_valid:
            best_valid = valid_auc
            best_test = roc_auc_eval(encoder, head, test_loader, device)
            best_epoch = epoch + 1

        stats = unwrap(encoder).codebook_stats()
        ent = np.mean([s["entropy"] for s in stats.values()])
        act = int(np.mean([s["active"] for s in stats.values()]))
        comp = f"enc={enc_m.avg:.3f}  cls={cls_m.avg:.3f}"
        if ge_m.count > 0:   comp += f"  ge={ge_m.avg:.4f}"
        if smi_m.count > 0:  comp += f"  smi={smi_m.avg:.3f}"
        print(f"  epoch {epoch+1:>3}/{epochs}  [{comp}]  entropy={ent:.3f}  active={act}/512  valid={valid_auc:.4f}  best={best_valid:.4f}")

    print(f"\n  Best epoch {best_epoch}: valid={best_valid:.4f}  test={best_test:.4f}")
    return best_valid, best_test


def finetune(encoder, head, train_loader, valid_loader, test_loader, args, epochs, freeze_encoder, device):
    if freeze_encoder:
        for p in encoder.parameters():
            p.requires_grad = False
        params = list(head.parameters())
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
        if hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)
        encoder.train() if not freeze_encoder else encoder.eval()
        head.train()
        for batch in train_loader:
            data = batch[0].to(device)
            targets = batch[-1].to(device, dtype=torch.float32)
            optimizer.zero_grad()
            head(encoder(data), targets=targets).backward()
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
    parser.add_argument("--with-ge-decoder",     action="store_true", help="use gene expression decoder")
    parser.add_argument("--with-smiles-decoder", action="store_true", help="use BERT-style masked token prediction decoder")
    parser.add_argument("--joint",               action="store_true", help="joint training: encoder + decoders + head simultaneously")
    parser.add_argument("--pretrain-on-pretrain-raw", action="store_true", help="pretrain encoder on pretrain_raw/ (156k) before finetuning on ChEMBL2K")
    parser.add_argument("--save-pretrained",  default=None, help="save encoder weights after pretraining to this path")
    parser.add_argument("--load-pretrained",  default=None, help="load encoder weights from this path instead of pretraining")
    parser.add_argument("--no-print",        action="store_true")
    parser.add_argument("--subset-ratio",    type=float, default=1.0)
    parser.add_argument("--mask-prob-start", type=float, default=0.15)
    parser.add_argument("--mask-prob-end",   type=float, default=0.15)
    parser.add_argument("--head-type",       type=str,   default="small", choices=["small", "wide", "deep"], help="Type of classification head: small, wide, deep")
    cli = parser.parse_args()

    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data.distributed import DistributedSampler

    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if local_rank != -1:
        dist.init_process_group(backend="nccl" if dist.is_nccl_available() else "gloo")
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        cli.gpu_id = local_rank
    else:
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
    args.no_print    = cli.no_print or (local_rank > 0)
    args.subset_ratio = cli.subset_ratio
    args.mask_prob_start = cli.mask_prob_start
    args.mask_prob_end   = cli.mask_prob_end
    args.head_type   = cli.head_type
    args.device      = device
    args.gpu_id      = cli.gpu_id

    torch.manual_seed(0)

    # Ensure only rank 0 downloads/processes dataset and invalidates the cache first
    if local_rank > 0 and dist.is_initialized():
        dist.barrier()

    # Load finetune dataset — build combined vocab if using pretrain_raw/ (either pretraining or loading checkpoint)
    need_pretrain_vocab = cli.pretrain_on_pretrain_raw or (cli.load_pretrained is not None)
    if need_pretrain_vocab:
        import pandas as pd
        finetune_smiles = pd.read_csv("raw_data/chembl2k/raw/assays.csv.gz")["smiles"].tolist()
        pretrain_ds = PretrainSMILESDataset(root="./raw_data")
        # Re-tokenize ChEMBL2K with the combined vocab (invalidate old cache)
        chembl_cache = "./raw_data/chembl2k/processed/processed_smiles_L128.pt"
        if os.path.exists(chembl_cache):
            try:
                _, _, cached_vocab = torch.load(chembl_cache, weights_only=False)
                if len(cached_vocab) != len(pretrain_ds.vocab):
                    print("Vocab size changed — invalidating ChEMBL2K cache...")
                    os.remove(chembl_cache)
            except Exception:
                pass
        args._vocab_override = pretrain_ds.vocab

    dataset = get_data(args, "./raw_data", transform="smiles")

    if local_rank <= 0 and dist.is_initialized():
        dist.barrier()

    # Inject combined vocab into dataset if using pretrain_raw/ vocab
    if need_pretrain_vocab:
        dataset.vocab        = pretrain_ds.vocab
        dataset.vocab_size   = pretrain_ds.vocab_size
        dataset.pad_token_id = pretrain_ds.pad_token_id
        dataset.mask_token_id = pretrain_ds.mask_token_id

    split   = dataset.get_idx_split()

    args.num_trained = len(split["train"])
    args.task_type   = "classification"

    train_sub = Subset(dataset, split["train"])
    valid_sub = Subset(dataset, split["valid"])
    test_sub  = Subset(dataset, split["test"])

    world_size = dist.get_world_size() if dist.is_initialized() else 1
    per_gpu_batch = max(1, cli.batch_size // world_size)

    train_sampler = DistributedSampler(train_sub, shuffle=True) if dist.is_initialized() else None
    valid_sampler = DistributedSampler(valid_sub, shuffle=False) if dist.is_initialized() else None
    test_sampler  = DistributedSampler(test_sub, shuffle=False) if dist.is_initialized() else None

    train_loader = DataLoader(train_sub, batch_size=per_gpu_batch, shuffle=(train_sampler is None), sampler=train_sampler, num_workers=cli.num_workers)
    valid_loader = DataLoader(valid_sub, batch_size=per_gpu_batch, shuffle=False, sampler=valid_sampler, num_workers=cli.num_workers)
    test_loader  = DataLoader(test_sub,  batch_size=per_gpu_batch, shuffle=False, sampler=test_sampler, num_workers=cli.num_workers)

    args.steps = len(train_loader)

    vocab_size = pretrain_ds.vocab_size if need_pretrain_vocab else dataset.vocab_size
    pad_id     = pretrain_ds.pad_token_id if need_pretrain_vocab else dataset.pad_token_id
    mask_id    = pretrain_ds.mask_token_id if need_pretrain_vocab else dataset.mask_token_id

    encoder = Encoder(
        V=512, dx=None, d=256, num_heads=8, num_layers=6, K_layers=[1, 3, 5],
        gamma_teacher=0.95, gamma_codebook=0.99, mask_prob=0.15,
        vocab_size=vocab_size, mask_token_id=mask_id, pad_token_id=pad_id,
    ).to(device)

    decoder        = FingerprintDecoder(d=256).to(device) if cli.with_decoder else None
    ge_decoder     = GEDecoder(d=256).to(device) if cli.with_ge_decoder else None
    smiles_decoder = SMILESDecoder(d=256, vocab_size=dataset.vocab_size).to(device) if cli.with_smiles_decoder else None
    head           = ClassificationHead(d=256, head_type=cli.head_type, num_tasks=dataset.num_tasks).to(device)

    if dist.is_initialized():
        encoder = DDP(encoder, device_ids=[local_rank], output_device=local_rank)
        if decoder is not None: decoder = DDP(decoder, device_ids=[local_rank], output_device=local_rank)
        if ge_decoder is not None: ge_decoder = DDP(ge_decoder, device_ids=[local_rank], output_device=local_rank)
        if smiles_decoder is not None: smiles_decoder = DDP(smiles_decoder, device_ids=[local_rank], output_device=local_rank)
        head = DDP(head, device_ids=[local_rank], output_device=local_rank)

    print(f"Device: {device}")
    print(f"Train/valid/test: {len(split['train'])}/{len(split['valid'])}/{len(split['test'])}")
    print(f"Pretrain epochs: {cli.pretrain_epochs}  |  Finetune epochs: {cli.finetune_epochs}")
    decoders_str = ", ".join(filter(None, ["FP" if decoder else None, "GE" if ge_decoder else None, "SMILES" if smiles_decoder else None])) or "none"
    mode = "joint" if cli.joint else ("frozen" if not cli.no_freeze else "e2e")
    print(f"Decoders: {decoders_str}  |  Mode: {mode}  |  Head: {cli.head_type}")

    if cli.load_pretrained:
        encoder.load_state_dict(torch.load(cli.load_pretrained, map_location=device))
        print(f"Loaded pretrained encoder from {cli.load_pretrained}")

    if cli.pretrain_on_pretrain_raw:
        pre_sampler = DistributedSampler(pretrain_ds, shuffle=True) if dist.is_initialized() else None
        pretrain_loader = DataLoader(pretrain_ds, batch_size=per_gpu_batch, shuffle=(pre_sampler is None), sampler=pre_sampler, num_workers=cli.num_workers)
        args.steps = len(pretrain_loader)
        pretrain(encoder, decoder, ge_decoder, pretrain_loader, args, cli.pretrain_epochs, device)
        args.steps = len(train_loader)  # reset for finetune
        if cli.save_pretrained:
            os.makedirs(os.path.dirname(cli.save_pretrained) or ".", exist_ok=True)
            if local_rank <= 0:
                torch.save(unwrap(encoder).state_dict(), cli.save_pretrained)
            print(f"Saved pretrained encoder to {cli.save_pretrained}")

    if cli.joint:
        best_valid, best_test = joint_train(
            encoder, ge_decoder, smiles_decoder, head, train_loader, valid_loader, test_loader,
            args, cli.finetune_epochs, device=device,
        )
    else:
        if cli.pretrain_epochs > 0:
            pretrain(encoder, decoder, ge_decoder, train_loader, args, cli.pretrain_epochs, device)
        best_valid, best_test = finetune(
            encoder, head, train_loader, valid_loader, test_loader,
            args, cli.finetune_epochs, freeze_encoder=not cli.no_freeze, device=device,
        )

    if local_rank <= 0:
        os.makedirs("results", exist_ok=True)
        tag = f"{'joint' if cli.joint else f'pre{cli.pretrain_epochs}'}_ft{cli.finetune_epochs}_{'frozen' if not cli.no_freeze else 'e2e'}_{'ge' if ge_decoder else 'nodec'}_{cli.head_type}"
        with open(f"results/{tag}.txt", "w") as f:
            f.write(f"valid={best_valid:.4f}  test={best_test:.4f}\n")
        print(f"\nSaved to results/{tag}.txt")


if __name__ == "__main__":
    main()
