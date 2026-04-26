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

def train_one_epoch_only_encoder(args, encoder, train_loaders, optimizer, scheduler, epoch, decoder=None, decoder_lambda=0.25, ge_decoder=None, ge_lambda=0.25, smiles_decoder=None, smiles_lambda=0.25):
    if not args.no_print:
        p_bar = tqdm(range(args.steps))
    batch_time = AverageMeter()
    total_losses = AverageMeter()
    enc_losses   = AverageMeter()
    fp_losses    = AverageMeter()
    ge_losses    = AverageMeter()
    smi_losses   = AverageMeter()
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

        data = batch['data'].to(device)
        fingerprints = ge_features = cp = targets = None

        if 'fingerprints' in batch:
            fingerprints = batch['fingerprints'].to(device, dtype=torch.float32)
        if 'ge_features' in batch:
            ge_features = batch['ge_features'].to(device, dtype=torch.float32)
        if 'cp_features' in batch:
            cp_features = batch['cp_features'].to(device, dtype=torch.float32)
        if 'targets' in batch:
            targets = batch['targets'].to(device, dtype=torch.float32)

        use_smiles = smiles_decoder is not None and data.dtype == torch.long
        enc_out = encoder(data, return_loss=True, update_codebooks=True, return_masked_info=use_smiles)
        
        if use_smiles:
            enc_loss, z_masked, mask, x_orig = enc_out
        else:
            enc_loss = enc_out
            
        loss = enc_loss
        enc_losses.update(enc_loss.item())

        z = encoder(data) if (decoder is not None or ge_decoder is not None) else None

        if decoder is not None and fingerprints is not None:
            fp_l = decoder(z, fingerprint=fingerprints)
            loss = loss + decoder_lambda * fp_l
            fp_losses.update(fp_l.item())

        if ge_decoder is not None and ge_features is not None:
            ge_l = ge_decoder(z, ge_targets=ge_features)
            loss = loss + ge_lambda * ge_l
            ge_losses.update(ge_l.item())
            
        if use_smiles:
            smi_l = smiles_decoder(z_masked, mask=mask, original_tokens=x_orig)
            loss = loss + smiles_lambda * smi_l
            smi_losses.update(smi_l.item())

        loss.backward()
        optimizer.step()
        scheduler.step()
        
        # unwrap encoder before calling update_teacher to avoid DDP errors
        model = encoder.module if hasattr(encoder, "module") else encoder
        model.update_teacher()
        
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
                desc += f"  ge_features={ge_losses.avg:.4f}"
            if smi_losses.count > 0:
                desc += f"  smi={smi_losses.avg:.3f}"
            desc += f"  {batch_time.avg*1000:.0f}ms/batch"
            p_bar.set_description(desc)
            p_bar.update()

    if not args.no_print:
        p_bar.close()

    component_losses = {"enc": enc_losses.avg}
    if fp_losses.count > 0:
        component_losses["fp"] = fp_losses.avg
    if ge_losses.count > 0:
        component_losses["ge_features"] = ge_losses.avg
    if smi_losses.count > 0:
        component_losses["smi"] = smi_losses.avg

    return train_loaders, total_losses.avg, component_losses
