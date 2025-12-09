import time
from tqdm import tqdm
import torch
from .misc import AverageMeter

cls_criterion = torch.nn.BCEWithLogitsLoss(reduction="none")
reg_criterion = torch.nn.L1Loss(reduction="none")

def finetune_func_mlp(args, model, train_loaders, optimizer, scheduler, epoch):
    if args.task_type == "regression":
        criterion = reg_criterion
    else:
        criterion = cls_criterion
    if not args.no_print:
        p_bar = tqdm(range(args.steps))
    batch_time = AverageMeter()
    data_time = AverageMeter()
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
        # scheduler.step()
        losses.update(loss.item())
        batch_time.update(time.time() - end)
        end = time.time()
        if not args.no_print:
            p_bar.set_description(
                "Train Epoch: {epoch}/{epochs:4}. Iter: {batch:4}/{iter:4}. LR: {lr:.8f}. Batch: {bt:.3f}s. Loss: {loss:.4f}. ".format(
                    epoch=epoch + 1,
                    epochs=args.epochs,
                    batch=batch_idx + 1,
                    iter=args.steps,
                    # lr=scheduler.get_last_lr()[0],
                    lr=args.lr,
                    bt=batch_time.avg,
                    loss=losses.avg,
                )
            )
            p_bar.update()
    if not args.no_print:
        p_bar.close()

    return train_loaders
