import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import os

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from configures.arguments import get_args
from dataset.create_datasets import get_data
from utils import validate, save_prediction
from utils.train_funcs import train_one_epoch_initial, get_cosine_schedule_with_warmup
from models.mlp import MLP


def main(args, seed):
    device = torch.device("cuda", args.gpu_id)
    args.device = device

    dataset = get_data(args, "../raw_data", transform="fingerprint")
    split_idx = dataset.get_idx_split()

    if args.subset_ratio < 1.0:
        split_idx["train"] = dataset.resample_train_idx(split_idx["train"], args.subset_ratio, seed=seed)

    args.num_trained = len(split_idx["train"])
    args.task_type = "regression" if "mae" in dataset.eval_metric else "classification"
    args.steps = args.num_trained // args.batch_size + 1

    train_loader = DataLoader(Subset(dataset, split_idx["train"]), batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    valid_loader = DataLoader(Subset(dataset, split_idx["valid"]), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader  = DataLoader(Subset(dataset, split_idx["test"]),  batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = MLP(num_tasks=dataset.num_tasks, emb_dim=args.emb_dim, drop_ratio=args.drop_ratio).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wdecay)
    scheduler = get_cosine_schedule_with_warmup(optimizer, 0, args.epochs * args.steps)

    train_loaders = {"train_iter": iter(train_loader), "train_loader": train_loader}
    best_valid, best_test, best_train, best_epoch, best_params = None, None, None, 0, None

    for epoch in range(args.epochs):
        train_loaders = train_one_epoch_initial(args, model, train_loaders, optimizer, scheduler, epoch)
        valid_perf = validate(args, model, valid_loader)

        improved = (epoch == 0) or (
            valid_perf[dataset.eval_metric] < best_valid if args.task_type == "regression"
            else valid_perf[dataset.eval_metric] > best_valid
        )
        if improved:
            test_perf = validate(args, model, test_loader)
            train_perf = validate(args, model, train_loader)
            best_params = parameters_to_vector(model.parameters())
            best_valid = valid_perf[dataset.eval_metric]
            best_test = test_perf[dataset.eval_metric]
            best_train = train_perf[dataset.eval_metric]
            best_epoch = epoch
            if not args.no_print:
                print(f"  Epoch {epoch}: train={best_train:.4f} valid={best_valid:.4f} test={best_test:.4f}")
        else:
            if not args.no_print:
                print(f"  Epoch {epoch}: valid={valid_perf[dataset.eval_metric]:.4f} (best={best_valid:.4f}, patience {epoch - best_epoch}/{args.patience})")
            if epoch - best_epoch > args.patience:
                break

    print(f"Seed {seed} done — best epoch {best_epoch}, {dataset.eval_metric}: train={best_train:.4f} valid={best_valid:.4f} test={best_test:.4f}")

    vector_to_parameters(best_params, model.parameters())
    save_prediction(model, device, test_loader, dataset, args.output_dir, seed)

    return best_train, best_valid, best_test


if __name__ == "__main__":
    args = get_args()
    args.model_name = "mlp_baseline"
    args.output_dir = f"output/{args.dataset}/{args.model_name}"
    os.makedirs(args.output_dir, exist_ok=True)

    results = []
    for seed in range(5):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        train, valid, test = main(args, seed)
        results.append((train, valid, test))

    trains, valids, tests = zip(*results)
    print(f"\n=== Summary ({len(results)} seeds) ===")
    print(f"  train: {np.mean(trains):.4f} +/- {np.std(trains):.4f}")
    print(f"  valid: {np.mean(valids):.4f} +/- {np.std(valids):.4f}")
    print(f"  test:  {np.mean(tests):.4f} +/- {np.std(tests):.4f}")
