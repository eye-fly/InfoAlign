import time
from tqdm import tqdm
import torch
from .misc import AverageMeter

cls_criterion = torch.nn.BCEWithLogitsLoss(reduction="none")
reg_criterion = torch.nn.L1Loss(reduction="none")

def train_one_epoch(args, model, train_loaders, optimizer, scheduler, epoch):
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

def train_one_epoch_only_encoder(args, encoder, train_loaders, optimizer, scheduler, epoch, decoder=None, decoder_lambda=0.25):
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
        try:
            batch = next(train_loaders["train_iter"])
        except:
            train_loaders["train_iter"] = iter(train_loaders["train_loader"])
            batch = next(train_loaders["train_iter"])

        if len(batch) == 3:
            data, fingerprints, targets = batch
            fingerprints = fingerprints.to(device, dtype=torch.float32)
        else:
            data, targets = batch
            fingerprints = None

        if data.dtype == torch.long:
            data = data.to(device)
        else:
            data = data.to(device, dtype=torch.float32)
        targets = targets.to(device, dtype=torch.float32)

        enc_loss = encoder.loss(data, update_codebooks=True)
        if decoder is not None and fingerprints is not None:
            z = encoder(data)
            dec_loss = decoder.loss(z, fingerprints)
            loss = enc_loss + decoder_lambda * dec_loss
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
