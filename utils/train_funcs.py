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

def train_one_epoch_only_encoder(args, encoder, train_loaders, optimizer, scheduler, epoch, decoder=None, decoder_lambda=0.25, ge_decoder=None, ge_lambda=0.25):
    if not args.no_print:
        p_bar = tqdm(range(args.steps))
    batch_time = AverageMeter()
    total_losses = AverageMeter()
    enc_losses   = AverageMeter()
    fp_losses    = AverageMeter()
    ge_losses    = AverageMeter()
    device = args.device
    encoder.train()
    for batch_idx in range(args.steps):
        end = time.time()
        optimizer.zero_grad()
        try:
            batch = next(train_loaders["train_iter"])
        except:
            train_loaders["train_iter"] = iter(train_loaders["train_loader"])
            batch = next(train_loaders["train_iter"])

        if len(batch) == 1:
            data = batch[0].to(device)
            fingerprints = ge = targets = None
        else:
            data, fingerprints, ge, targets = batch
            data         = data.to(device)
            fingerprints = fingerprints.to(device, dtype=torch.float32)
            ge           = ge.to(device, dtype=torch.float32)
            targets      = targets.to(device, dtype=torch.float32)

        enc_loss = encoder.loss(data, update_codebooks=True)
        loss = enc_loss
        enc_losses.update(enc_loss.item())

        if decoder is not None and fingerprints is not None:
            z = encoder(data)
            fp_l = decoder.loss(z, fingerprints)
            loss = loss + decoder_lambda * fp_l
            fp_losses.update(fp_l.item())

        if ge_decoder is not None and ge is not None:
            z = encoder(data)
            ge_l = ge_decoder.loss(z, ge)
            loss = loss + ge_lambda * ge_l
            ge_losses.update(ge_l.item())

        loss.backward()
        optimizer.step()
        scheduler.step()
        encoder.update_teacher()
        total_losses.update(loss.item())
        batch_time.update(time.time() - end)

        if not args.no_print:
            desc = (f"Epoch {epoch+1}  "
                    f"[{batch_idx+1}/{args.steps}]  "
                    f"loss={total_losses.avg:.3f}  "
                    f"enc={enc_losses.avg:.3f}")
            if fp_losses.count > 0:
                desc += f"  fp={fp_losses.avg:.3f}"
            if ge_losses.count > 0:
                desc += f"  ge={ge_losses.avg:.4f}"
            desc += f"  {batch_time.avg*1000:.0f}ms/batch"
            p_bar.set_description(desc)
            p_bar.update()

    if not args.no_print:
        p_bar.close()

    component_losses = {"enc": enc_losses.avg}
    if fp_losses.count > 0:
        component_losses["fp"] = fp_losses.avg
    if ge_losses.count > 0:
        component_losses["ge"] = ge_losses.avg

    return train_loaders, total_losses.avg, component_losses
