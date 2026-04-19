import warnings

warnings.filterwarnings("ignore", category=UserWarning)

import logging

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from configures.arguments import get_args
from dataset.create_datasets import get_data
from utils import validate, init_weights, save_prediction
from utils.train_funcs import train_one_epoch_initial, get_cosine_schedule_with_warmup
from models.mlp import MLP


def get_logger(name, logfile=None):
    logger = logging.getLogger(name)
    if logger.hasHandlers():
        logger.handlers.clear()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(message)s")
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    if logfile is not None:
        fh = logging.FileHandler(logfile)
        fh.setFormatter(formatter)
        fh.setLevel(logging.DEBUG)
        logger.addHandler(fh)
    logger.propagate = False
    return logger


def main(args, seed):
    device = torch.device("cuda", args.gpu_id)
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    dataset = get_data(args, "../raw_data", transform="fingerprint")
    split_idx = dataset.get_idx_split()

    if args.subset_ratio < 1.0:
        print(f"Resampling training set with ratio {args.subset_ratio}")
        split_idx["train"] = dataset.resample_train_idx(split_idx["train"], args.subset_ratio, seed=seed)

    args.num_trained = len(split_idx["train"])
    args.task_type = dataset.task_type
    args.steps = args.num_trained // args.batch_size + 1

    train_dataset = Subset(dataset, split_idx["train"])
    valid_dataset = Subset(dataset, split_idx["valid"])
    test_dataset = Subset(dataset, split_idx["test"])

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = MLP(
        num_tasks=dataset.num_tasks,
        emb_dim=args.emb_dim,
        drop_ratio=args.drop_ratio,
    ).to(device)

    optimizer = optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.wdecay
    )

    scheduler = get_cosine_schedule_with_warmup(optimizer, 0, args.epochs * args.steps)

    logging.warning(f"device: {args.device}, n_gpu: {args.n_gpu}")
    logger.info(dict(vars(args)))
    logger.info(model)
    logger.info("***** Running training *****")
    logger.info(
        f"  Task = {args.dataset}@{args.num_trained}/{len(split_idx['valid'])}/{len(split_idx['test'])}"
    )
    logger.info(f"  Num Epochs = {args.epochs}")
    logger.info(f"  Total train batch size = {args.batch_size}")
    logger.info(f"  Total optimization steps = {args.epochs * args.steps}")

    train_loaders = {"train_iter": iter(train_loader), "train_loader": train_loader}

    best_train, best_valid, best_test, best_count = None, None, None, None
    best_epoch = 0

    args.task_type = (
        "regression" if "mae" in dataset.eval_metric else "classification"
    )
    best_params = None
    for epoch in range(0, args.epochs):
        train_loaders = train_one_epoch_initial(
            args, model, train_loaders, optimizer, scheduler, epoch
        )
        valid_perf = validate(args, model, valid_loader)

        if epoch > 0:
            is_improved = (
                valid_perf[dataset.eval_metric] < best_valid
                if args.task_type == "regression"
                else valid_perf[dataset.eval_metric] > best_valid
            )
        if epoch == 0 or is_improved:
            train_perf = validate(args, model, train_loader)
            test_perf = validate(args, model, test_loader)
            best_params = parameters_to_vector(model.parameters())
            best_valid = valid_perf[dataset.eval_metric]
            best_test = test_perf[dataset.eval_metric]
            best_train = train_perf[dataset.eval_metric]
            best_epoch = epoch
            best_count = test_perf.get("count", None)
            if best_count is None:
                best_count = test_perf.get("mae_list", None)
            if not args.no_print:
                logger.info(
                    "Update Epoch {}: best_train: {:.4f} best_valid: {:.4f}, best_test: {:.4f}".format(
                        epoch, best_train, best_valid, best_test
                    )
                )
                if best_count is not None and args.task_type == "classification":
                    outstr = "Best Count: "
                    for key, value in best_count.items():
                        sum_num = int(np.nansum(value))
                        outstr += f"{key}: {sum_num/len(value):.4f} (nan {sum(np.isnan(value))} / {len(value)}), "
                    logger.info(outstr)
        else:
            if not args.no_print:
                logger.info(
                    "Epoch {}: best_valid: {:.4f}, current_valid: {:.4f}, patience: {}/{}".format(
                        epoch,
                        best_valid,
                        valid_perf[dataset.eval_metric],
                        epoch - best_epoch,
                        args.patience,
                    )
                )
            if epoch - best_epoch > args.patience:
                break

    logger.info(
        "Finished. \n {}-{} Best validation epoch {} with metric {}, train {:.4f}, valid {:.4f}, test {:.4f}".format(
            args.dataset, args.model_name, best_epoch, dataset.eval_metric, best_train, best_valid, best_test
        )
    )
    vector_to_parameters(best_params, model.parameters())
    save_prediction(model, device, test_loader, dataset, args.output_dir, seed)

    return (
        args.model_name,
        args.dataset,
        dataset.eval_metric,
        best_train,
        best_valid,
        best_test,
        best_epoch,
        best_count,
    )


if __name__ == "__main__":
    import os
    import pandas as pd

    args = get_args()

    model_name = "mlp_baseline"
    args.model_name = model_name
    args.output_dir = f"output/{args.dataset}/{model_name}"

    logger = get_logger(__name__)
    args.logger = logger
    print(vars(args))

    df = pd.DataFrame()
    for i in range(5):
        model, dataset, metric, train, valid, test, epoch, count = main(args, i)
        if "auc" in metric:
            new_results = {
                "model": model,
                "dataset": dataset,
                "seed": i,
                "metric": metric,
                "train": train,
                "valid": valid,
                "test": test,
                "epoch": epoch,
                "suc_80": round(np.nansum(count[80]) / len(count[80]), 4),
                "suc_85": round(np.nansum(count[85]) / len(count[85]), 4),
                "suc_90": round(np.nansum(count[90]) / len(count[90]), 4),
                "suc_95": round(np.nansum(count[95]) / len(count[95]), 4),
                "thr_80": count[80],
                "thr_85": count[85],
                "thr_90": count[90],
                "thr_95": count[95],
            }
        else:
            mae_list = count
            new_results = {
                "model": model,
                "dataset": dataset,
                "seed": i,
                "metric": metric,
                "train": train,
                "valid": valid,
                "test": test,
                "epoch": epoch,
                "mae_1": mae_list[0],
                "mae_2": mae_list[1],
                "mae_3": mae_list[2],
                "mae_4": mae_list[3],
                "mae_5": mae_list[4],
                "mae_6": mae_list[5],
            }
        df = pd.concat([df, pd.DataFrame([new_results])], ignore_index=True)

    summary_each = f"output/{args.dataset}/summary_each.csv"
    if os.path.exists(summary_each):
        df.to_csv(summary_each, mode="a", header=False, index=False)
    else:
        df.to_csv(summary_each, index=False)
    print(df)

    # Calculate mean and std
    if "auc" in metric:
        cols = [
            "model", "dataset", "metric", "train", "valid", "test",
            "suc_80", "suc_85", "suc_90", "suc_95",
        ]
    else:
        cols = [
            "model", "dataset", "metric", "train", "valid", "test",
            "mae_1", "mae_2", "mae_3", "mae_4", "mae_5", "mae_6",
        ]
    df_mean = df[cols].groupby(["model", "dataset", "metric"]).mean().round(4)
    df_std = df[cols].groupby(["model", "dataset", "metric"]).std().round(4)

    df_mean = df_mean.reset_index()
    df_std = df_std.reset_index()
    df_summary = df_mean[["model", "dataset", "metric"]].copy()
    for col in cols[3:]:
        df_summary[col] = df_mean[col].astype(str) + "±" + df_std[col].astype(str)

    summary_all = f"output/{args.dataset}/summary_all.csv"
    if os.path.exists(summary_all):
        df_summary.to_csv(summary_all, mode="a", header=False, index=False)
    else:
        df_summary.to_csv(summary_all, index=False)
    print(df_summary)
