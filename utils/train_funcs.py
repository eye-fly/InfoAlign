import argparse
import math
import time

from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
import torch
from .misc import AverageMeter
from configures.arguments import get_args


cls_criterion = torch.nn.BCEWithLogitsLoss(reduction="none")
reg_criterion = torch.nn.L1Loss(reduction="none")


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps,
                                    num_cycles=7./16., last_epoch=-1):
    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        no_progress = float(current_step - num_warmup_steps) / \
            float(max(1, num_training_steps - num_warmup_steps))
        return max(0, math.cos(math.pi * num_cycles * no_progress))
    return LambdaLR(optimizer, _lr_lambda, last_epoch)


def parse_arguments():
    """
    Prepares args object, that stores training hyperparameters and data.
    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="finetune-chembl2k")
    parser.add_argument("--pretrain-epochs", type=int, default=60)
    parser.add_argument("--finetune-epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wdecay", type=float, default=1e-5)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-freeze", action="store_true", help="finetune encoder end-to-end")
    parser.add_argument("--with-decoder", action="store_true", help="use fingerprint decoder during pretraining")
    parser.add_argument("--no-print", action="store_true")
    parser.add_argument("--subset-ratio", type=float, default=1.0)

    cli = parser.parse_args()

    device = torch.device(f"cuda:{cli.gpu_id}" if torch.cuda.is_available() else "cpu")

    # Reuse get_args for dataset-level settings (eval metric, num_tasks etc.)
    args = get_args.__wrapped__() if hasattr(get_args, "__wrapped__") else argparse.Namespace(
        dataset=cli.dataset, batch_size=cli.batch_size, lr=cli.lr, wdecay=cli.wdecay,
        gpu_id=cli.gpu_id, num_workers=cli.num_workers, no_print=True, subset_ratio=cli.subset_ratio,
    )
    args.dataset = cli.dataset
    args.batch_size = cli.batch_size
    args.lr = cli.lr
    args.wdecay = cli.wdecay
    args.num_workers = cli.num_workers
    args.no_print = True
    args.subset_ratio = cli.subset_ratio
    args.device = device
    args.gpu_id = cli.gpu_id

    args.batch_size = cli.batch_size
    args.num_workers = cli.num_workers
    args.with_decoder = cli.with_decoder
    args.pretrain_epochs = cli.pretrain_epochs
    args.finetune_epochs = cli.finetune_epochs
    args.freeze = not cli.no_freeze

    return args

def train_one_epoch_initial(args, model, train_loaders, optimizer, scheduler, epoch):
    """
    Original train epoch from InfoAlign.
    :param args:
    :param model:
    :param train_loaders:
    :param optimizer:
    :param scheduler:
    :param epoch:
    :return:
    """
    if args.task_type == "regression":
        criterion = reg_criterion
    else:
        criterion = cls_criterion
    if not args.no_print:
        p_bar = tqdm(range(args.steps))
    batch_time = AverageMeter()
    losses = AverageMeter()
    device = args.device
    model.train()
    for batch_idx in range(args.steps):
        end = time.time()
        model.zero_grad()
        try:
            data, targets = next(train_loaders["train_iter"])
        except:
            train_loaders["train_iter"] = iter(train_loaders["train_loader"])
            data, targets = next(train_loaders["train_iter"])

        data = data.to(device, dtype=torch.float32)
        targets = targets.to(device, dtype=torch.float32)

        is_labeled = targets == targets

        preds = model(data)
        loss = criterion(
            preds.view(targets.size()).to(torch.float32)[is_labeled],
            targets[is_labeled],
        ).mean()

        loss.backward()
        optimizer.step()
        scheduler.step()
        losses.update(loss.item())
        batch_time.update(time.time() - end)
        if not args.no_print:
            p_bar.set_description(
                "Train Epoch: {epoch}/{epochs:4}. Iter: {batch:4}/{iter:4}. LR: {lr:.8f}. Batch: {bt:.3f}s. Loss: {loss:.4f}. ".format(
                    epoch=epoch + 1,
                    epochs=args.epochs,
                    batch=batch_idx + 1,
                    iter=args.steps,
                    lr=scheduler.get_last_lr()[0],
                    bt=batch_time.avg,
                    loss=losses.avg,
                )
            )
            p_bar.update()
    if not args.no_print:
        p_bar.close()

    return train_loaders

######################################################
###### ENCODER Part ######### TODO
######################################################

def train_one_epoch(args, encoder, train_loaders, optimizer, scheduler, epoch, decoder=None, decoder_lambda=0.25):
    """
    Train encoder and decoder if present for one epoch.
    :param args:
    :param encoder:
    :param train_loaders:
    :param optimizer:
    :param scheduler:
    :param epoch:
    :param decoder:
    :param decoder_lambda:
    :return:
    """
    # if args.task_type == "regression":
        # criterion = reg_criterion
    # else:
        # criterion = cls_criterion
    if not args.no_print:
        p_bar = tqdm(range(args.steps))
    batch_time = AverageMeter()
    losses = AverageMeter()
    device = args.device
    encoder.train()
    for batch_idx in range(args.steps):
        end = time.time()
        # encoder.zero_grad()
        optimizer.zero_grad() # Some encoder parameters do not have gradients - we only want to zero out optimizer parameters.

        # GET BATCH
        try:
            batch = next(train_loaders["train_iter"])
        except:
            train_loaders["train_iter"] = iter(train_loaders["train_loader"])
            batch = next(train_loaders["train_iter"])

        ### GET DATA
        if len(batch) == 3:
            data, fingerprints, _ = batch
            fingerprints = fingerprints.to(device, dtype=torch.float32)
        else:
            data, _ = batch
            fingerprints = None
        if data.dtype == torch.long:
            data = data.to(device)
        else:
            data = data.to(device, dtype=torch.float32)

        ### LOSS AND UPDATES
        enc_loss = encoder.loss(data, update_codebooks=True)
        if decoder is not None:
            if fingerprints is not None and decoder.modality == 'fingerprints':
                z = encoder(data)
                dec_loss = decoder.loss(z, fingerprints)
                loss = enc_loss + decoder_lambda * dec_loss
            elif decoder.modality == 'SMILES':
                # TODO
                loss = enc_loss
            elif decoder.modality == 'multiple':
                # TODO
                loss = enc_loss
            else:
                loss = enc_loss
        else:
            loss = enc_loss

        loss.backward()
        optimizer.step()
        scheduler.step()
        encoder.update_teacher()
        losses.update(loss.item())
        batch_time.update(time.time() - end)
        if not args.no_print:
            p_bar.set_description(
                "Train Epoch: {epoch}/{epochs:4}. Iter: {batch:4}/{iter:4}. LR: {lr:.8f}. Batch: {bt:.3f}s. Loss: {loss:.4f}. ".format(
                    epoch=epoch + 1,
                    epochs=args.epochs,
                    batch=batch_idx + 1,
                    iter=args.steps,
                    lr=scheduler.get_last_lr()[0],
                    bt=batch_time.avg,
                    loss=losses.avg,
                )
            )
            p_bar.update()
    if not args.no_print:
        p_bar.close()

    return train_loaders, losses.avg # I have added losses.avg to return