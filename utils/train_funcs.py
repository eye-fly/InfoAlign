import math
import time

import numpy as np
from sklearn.metrics import roc_auc_score
from torch import optim, nn
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
import torch

from models.encoder import Encoder
from .misc import AverageMeter

cls_criterion = torch.nn.BCEWithLogitsLoss(reduction="none")
reg_criterion = torch.nn.L1Loss(reduction="none")


def get_cosine_schedule_with_warmup(optimizer,
                                    num_warmup_steps,
                                    num_training_steps,
                                    num_cycles=7. / 16.,
                                    last_epoch=-1):
    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        no_progress = float(current_step - num_warmup_steps) / \
                      float(max(1, num_training_steps - num_warmup_steps))
        return max(0., math.cos(math.pi * num_cycles * no_progress))

    return LambdaLR(optimizer, _lr_lambda, last_epoch)


def unwrap(model):
    return model.module if hasattr(model, "module") else model


def roc_auc_eval(encoder, head, loader, device):
    encoder.eval()
    head.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch in loader:
            data = batch['data'].to(device)
            targets = batch['targets'].to(device, dtype=torch.float32)
            logits = head(encoder(data))
            preds.append(torch.sigmoid(logits).cpu())
            trues.append(targets.cpu())
    preds = torch.cat(preds)
    trues = torch.cat(trues)
    import torch.distributed as dist
    if dist.is_initialized():
        # NCCL requires all tensors to be on the GPU for all_gather.
        preds = preds.to(device)
        trues = trues.to(device)
        gathered_preds = [torch.zeros_like(preds) for _ in range(dist.get_world_size())]
        gathered_trues = [torch.zeros_like(trues) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_preds, preds)
        dist.all_gather(gathered_trues, trues)
        preds = torch.cat(gathered_preds)
        trues = torch.cat(gathered_trues)

    preds = preds.cpu().numpy()
    trues = trues.cpu().numpy()
    scores = []
    for i in range(trues.shape[1]):
        mask = ~np.isnan(trues[:, i])
        if trues[mask, i].sum() > 0 and (1 - trues[mask, i]).sum() > 0:
            scores.append(roc_auc_score(trues[mask, i], preds[mask, i]))
    return float(np.mean(scores))


def extract_decoder_params(encoder=None, decoders=None):
    params = []
    if encoder is not None:
        params = list(unwrap(encoder).student_params())
    for decoder in decoders:
        if decoder is not None:
            params += list(unwrap(decoder).parameters())
    return params


def update_loss(decoder, z, decoder_lambda, losses, targets=None, mask=None, x_target=None):
    if decoder is None:
        return 0
    loss = 0
    if targets is not None:
        loss = decoder.loss(z, targets=targets)
    elif mask is not None:
        loss = decoder.loss(z, mask=mask, x_target=x_target)
    losses.update(loss.item())
    return decoder_lambda * loss


def train_step(
        batch, optimizer, scheduler, device,
        encoder, enc_losses,
        smiles_decoder_data: tuple[nn.Module, float, AverageMeter, str],
        decoders_data: list[tuple[nn.Module, float, AverageMeter, str]],
        total_losses, if_random_decoder_update=False):
    smiles_decoder, smiles_lambda, smi_losses, _ = smiles_decoder_data
    data = batch['data'].to(device)
    optimizer.zero_grad()

    loss = 0
    use_smiles = smiles_decoder is not None and data.dtype == torch.long
    enc_out = encoder(data, return_loss=True, update_codebooks=True, return_masked_info=use_smiles)

    if use_smiles:
        enc_loss, z_masked, mask, x_orig = enc_out
        loss += update_loss(smiles_decoder, z_masked, smiles_lambda, smi_losses, None, mask, x_orig)
    else:
        enc_loss = enc_out
    loss += enc_loss
    enc_losses.update(enc_loss.item())

    z = encoder(data)
    if if_random_decoder_update:
        nr_of_decoders = len(decoders_data)
        idx = torch.randint(nr_of_decoders, (1,))
        _decoder, _decoder_lambda, losses, features_name = decoders_data[idx]
        if features_name in batch:
            features = batch[features_name].to(device, dtype=torch.float32)
            loss += update_loss(_decoder, z, _decoder_lambda*nr_of_decoders, losses, features)
    else:
        for _decoder, _decoder_lambda, losses, features_name in decoders_data:
            if features_name in batch:
                features = batch[features_name].to(device, dtype=torch.float32)
                loss += update_loss(_decoder, z, _decoder_lambda, losses, features)

    loss.backward()
    optimizer.step()
    scheduler.step()
    total_losses.update(loss.item())
    unwrap(encoder).update_teacher()


def print_encoder_data(encoder: Encoder, epoch, epochs, loss, components):
    stats = unwrap(encoder).codebook_stats()
    ent = np.mean([s["entropy"] for s in stats.values()])
    act = int(np.mean([s["active"] for s in stats.values()]))
    comp_str = "  ".join(f"{k}={v:.3f}" for k, v in components.items())
    print(f"  epoch {epoch + 1:>3}/{epochs}  total={loss:.3f}  [{comp_str}]  entropy={ent:.3f}  active={act}/512")


def train_one_epoch_old(args, model, train_loaders, optimizer, scheduler, epoch):
    # From original InfoAlign, outdated
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

# TODO
# Rework joint_train_one_epoch and train_one_epoch to share more code or into one function
# Make a trainer class

def joint_train_one_epoch(args,
                          encoder,
                          train_loader,
                          scheduler, optimizer,
                          epochs, epoch,
                          head, ge_decoder, cp_decoder, smiles_decoder,
                          cls_lambda=1.0, ge_lambda=10.0, cp_lambda=10.0, smiles_lambda=0.25, if_random_decoder_update=False):
    # Meters
    total_losses = AverageMeter()
    enc_losses = AverageMeter()
    cls_losses = AverageMeter()
    ge_losses = AverageMeter()
    cp_losses = AverageMeter()
    smi_losses = AverageMeter()

    device = args.device

    if hasattr(train_loader.sampler, "set_epoch"):
        train_loader.sampler.set_epoch(epoch)

    progress = epoch / max(1, epochs - 1)
    current_mask = args.mask_prob_start + (args.mask_prob_end - args.mask_prob_start) * progress
    unwrap(encoder).mask_prob = current_mask

    encoder.train()
    head.train()

    for batch in train_loader:
        train_step(batch, optimizer, scheduler, device,
                   encoder, enc_losses,
                   (smiles_decoder, smiles_lambda, smi_losses, ''),
                   [
                       (head, cls_lambda, cls_losses, "targets"),
                       (ge_decoder, ge_lambda, ge_losses, "ge_features"),
                       (cp_decoder, cp_lambda, cp_losses, "cp_features")
                   ],
                   total_losses, if_random_decoder_update=if_random_decoder_update)

    component_losses = {}
    for name, losses in [("enc", enc_losses), ("ge", ge_losses), ("cp", cp_losses), ("smi", smi_losses)]:
        if losses.count > 0:
            component_losses[name] = losses.avg

    return train_loader, total_losses.avg, component_losses


def train_one_epoch(
        args, steps,
        encoder: Encoder,
        train_loaders,
        optimizer,
        scheduler,
        epoch,
        fp_decoder=None, ge_decoder=None, smiles_decoder=None, cp_decoder=None,
        fp_lambda=0.25, ge_lambda=0.25, smiles_lambda=0.25, cp_lambda=0.25, if_random_decoder_update=False
):
    batch_time = AverageMeter()
    total_losses = AverageMeter()
    enc_losses = AverageMeter()
    fp_losses = AverageMeter()
    ge_losses = AverageMeter()
    cp_losses = AverageMeter()
    smi_losses = AverageMeter()

    if not args.no_print:
        p_bar = tqdm(range(steps))
    device = args.device

    encoder.train()
    for batch_idx in range(steps):
        end = time.time()
        try:
            batch = next(train_loaders["train_iter"])
        except:
            # TODO Remove code reliant on undocumented exception?
            train_loaders["train_iter"] = iter(train_loaders["train_loader"])
            batch = next(train_loaders["train_iter"])

        train_step(batch, optimizer, scheduler, device,
                   encoder, enc_losses,
                   (smiles_decoder, smiles_lambda, smi_losses, ''),
                   [
                       (fp_decoder, fp_lambda, fp_losses, "targets"),
                       (ge_decoder, ge_lambda, ge_losses, "ge_features"),
                       (cp_decoder, cp_lambda, cp_losses, "cp_features")
                   ],
                   total_losses, if_random_decoder_update=if_random_decoder_update)

        batch_time.update(time.time() - end)

        if not args.no_print:
            desc = (f"Epoch {epoch + 1}  "
                    f"[{batch_idx + 1}/{steps}]  "
                    f"loss={total_losses.avg:.3f}  ")
            for name, losses in [
                ("enc", enc_losses),
                ("fp", fp_losses),
                ("ge", ge_losses),
                ("cp", cp_losses),
                ("smi", smi_losses)]:
                if losses.count > 0:
                    desc += f"  " + name + f"={losses.avg:.3f}"
            desc += f"  {batch_time.avg * 1000:.0f}ms/batch"
            p_bar.set_description(desc)
            p_bar.update()

    if not args.no_print:
        p_bar.close()

    component_losses = {}
    for name, losses in [
        ("enc", enc_losses),
        ("fp", fp_losses),
        ("ge", ge_losses),
        ("cp", cp_losses),
        ("smi", smi_losses)]:
        if losses.count > 0:
            component_losses[name] = losses.avg

    return train_loaders, total_losses.avg, component_losses


def joint_train(encoder,
                train_loader, valid_loader, test_loader,
                args, epochs,
                head, ge_decoder, cp_decoder, smiles_decoder,
                ge_lambda=10.0, cp_lambda=10.0, smiles_lambda=0.25, cls_lambda=1.0, if_random_decoder_update=False):
    device = args.device
    steps = len(train_loader)
    params = extract_decoder_params(encoder, decoders=[ge_decoder, cp_decoder, smiles_decoder])
    optimizer = optim.Adam(params, lr=args.lr, weight_decay=args.wdecay)
    scheduler = get_cosine_schedule_with_warmup(optimizer, 0, epochs * steps)

    print(f"\n{'=' * 50}")
    print(f"Joint training (encoder + GE decoder + head) for {epochs} epochs")
    print(f"{'=' * 50}")

    best_valid, best_test, best_epoch = 0.0, 0.0, 0
    for epoch in range(epochs):
        train_loaders, loss, components = (
            joint_train_one_epoch(args,
                                  encoder,
                                  train_loader, valid_loader, test_loader,
                                  scheduler, optimizer,
                                  epochs, epoch,
                                  head, ge_decoder, cp_decoder, smiles_decoder,
                                  cls_lambda=cls_lambda, ge_lambda=ge_lambda, cp_lambda=cp_lambda, smiles_lambda=smiles_lambda,
                                  if_random_decoder_update=if_random_decoder_update))
        print_encoder_data(encoder, epoch, epochs, loss, components)

        valid_auc = roc_auc_eval(encoder, head, valid_loader, device)
        if valid_auc > best_valid:
            best_valid = valid_auc
            best_test = roc_auc_eval(encoder, head, test_loader, device)
            best_epoch = epoch + 1
        print(f"valid={valid_auc:.4f}  best={best_valid:.4f}")

    print(f"\n  Best epoch {best_epoch}: valid={best_valid:.4f}  test={best_test:.4f}")
    return best_valid, best_test


def pretrain(encoder,
             train_loader,
             args, epochs,
             decoder, ge_decoder, cp_decoder, smiles_decoder, if_random_decoder_update=False):
    steps = len(train_loader)
    if hasattr(train_loader.sampler, "set_epoch"):
        train_loader.sampler.set_epoch(0)
    params = extract_decoder_params(encoder=encoder, decoders=[decoder, ge_decoder, cp_decoder, smiles_decoder])

    optimizer = optim.Adam(params, lr=args.lr, weight_decay=args.wdecay)
    scheduler = get_cosine_schedule_with_warmup(optimizer, 0, epochs * steps)
    train_loaders = {"train_iter": iter(train_loader), "train_loader": train_loader}

    print(f"\n{'=' * 50}")
    print(f"Pretraining encoder for {epochs} epochs")
    print(f"{'=' * 50}")
    for epoch in range(epochs):
        if hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)

        progress = epoch / max(1, epochs - 1)
        current_mask = args.mask_prob_start + (args.mask_prob_end - args.mask_prob_start) * progress
        unwrap(encoder).mask_prob = current_mask

        train_loaders, loss, components = train_one_epoch(
            args, steps, encoder, train_loaders, optimizer, scheduler, epoch,
            fp_decoder=decoder, ge_decoder=ge_decoder, cp_decoder=cp_decoder, smiles_decoder=smiles_decoder,
            if_random_decoder_update=if_random_decoder_update
        )
        print_encoder_data(encoder, epoch, epochs, loss, components)


def finetune(encoder, freeze_encoder,
             train_loader, valid_loader, test_loader,
             args, epochs,
             head):
    device = args.device
    steps = len(train_loader)
    if freeze_encoder:
        for p in encoder.parameters():
            p.requires_grad = False
        params = list(head.parameters())
    else:
        for p in encoder.parameters():
            p.requires_grad = True
        params = list(head.parameters()) + list(encoder.parameters())

    optimizer = optim.Adam(params, lr=args.lr, weight_decay=args.wdecay)
    scheduler = get_cosine_schedule_with_warmup(optimizer, 0, epochs * steps)

    mode = "frozen encoder" if freeze_encoder else "end-to-end"
    print(f"\n{'=' * 50}")
    print(f"Finetuning classification head ({mode}) for {epochs} epochs")
    print(f"{'=' * 50}")

    best_valid, best_test, best_epoch = 0.0, 0.0, 0
    for epoch in range(epochs):
        if hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)
        encoder.train() if not freeze_encoder else encoder.eval()
        head.train()
        for batch in train_loader:
            data = batch['data'].to(device)
            targets = batch['targets'].to(device, dtype=torch.float32)
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
            print(
                f"  epoch {epoch + 1:>3}/{epochs}  valid={valid_auc:.4f}  best_valid={best_valid:.4f}  best_test={best_test:.4f}")

    print(f"\n  Best epoch {best_epoch}: valid={best_valid:.4f}  test={best_test:.4f}")
    return best_valid, best_test
