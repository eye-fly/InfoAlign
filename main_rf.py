import sklearn
import numpy as np
import pandas as pd
from utils.misc import eval_func
import os
import torch
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from dataset.create_datasets import get_data
from configures.arguments import (
    load_arguments_from_yaml,
    save_arguments_to_yaml,
    get_args,
)



data = np.load("embeddings.npz")




X_train, y_train = data['X_train'], data['y_train']
X_val, y_val     = data['X_val'],   data['y_val']
X_test, y_test   = data['X_test'],  data['y_test']
print(f"{X_train.shape=}, {y_train.shape=}")








def train(args,seed,X_train, y_train):
#     gr_space = {
#     'max_depth': [None, 10,  30],
#     'n_estimators': [100, 200, 300,1000],
#     'max_features': ['sqrt', 'log2', None]
# }
    gr_space = {
    'max_depth': [ 5, None],
    'learning_rate': [0.1, 0.001],
    'subsample': [0.5, 1],
    "n_estimators": [100,  300,1000],
}
    forest = XGBClassifier( random_state=seed, n_jobs=8,device ='cuda')
    models = []

    grid = GridSearchCV(forest, gr_space, cv = 3, scoring='roc_auc', verbose =3,n_jobs=8)
#model_grid = grid.fit(X_train_resampled, y_train_resampled)
    num_tasks = y_train.shape[1]
    print("Creating models for", num_tasks, "tasks")
    for i in range(num_tasks):
        
        mask = ~np.isnan(y_train[:, i])
        
        X_task = X_train[mask]
        y_task = y_train[mask, i]
        
        if len(y_task) > 0:
            # Clone or create a new model instance for this task
            from sklearn.base import clone
            model = clone(grid)
            model.fit(X_task, y_task)
            print(f"For task {i} best parameters were:",model.best_params_)
            print(f"For task {i} best score was:",model.best_score_)
            models.append(model)
        else:
            models.append(None) # Handle tasks with no data
            print(f"Warning: Task {i} has no labeled data.")
        
    return models
    
def predict(models, X):
    num_tasks = len(models)
    
    num_samples = X.shape[0]
    y_pred = np.full((num_samples, num_tasks), np.nan)  # Initialize with NaNs
    
    for i, model in enumerate(models):
        if model is not None:
            y_pred[:, i] = model.predict_proba(X)[:, 1]
        else:
            print(f"Warning: No model for task {i}, predictions remain NaN.")

    return y_pred


args = get_args()


if args.dataset == "pretrain":
        dataset, context_graph = get_data(args, "./raw_data", transform="pyg")
        context_graph = context_graph[0]
else:
    dataset = get_data(args, "./raw_data", transform="pyg")
    context_graph = None


def main(args, seed):
    models = train(args,seed,X_train, y_train)
    y_train_pred = predict(models,X_train)
    
    y_val_pred = predict(models,X_val)
    y_test_pred =predict(models,X_test) 

    


    
    train_perf = eval_func(y_train_pred, y_train)
    valid_perf = eval_func(y_val_pred, y_val)
    test_perf = eval_func(y_test_pred, y_test)
    
    best_valid = valid_perf[dataset.eval_metric]
    best_test = test_perf[dataset.eval_metric]
    best_train = train_perf[dataset.eval_metric]
    best_count = test_perf.get("count", None)
    if best_count is None:
        best_count = test_perf.get("mae_list", None)
    return "XGBOOST", args.dataset, dataset.eval_metric, best_train, best_valid, best_test, 0, best_count

df = pd.DataFrame()
for i in range(5):
    model, dataset_name, metric, training, valid, test, epoch, count = main(args, i)
    if "auc" in metric:
        new_results = {
            "model": model,
            "dataset": dataset_name,
            "seed": i,
            "metric": metric,
            "train": training,
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
            "dataset": dataset_name,
            "seed": i,
            "metric": metric,
            "train": training,
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
        "model",
        "dataset",
        "metric",
        "train",
        "valid",
        "test",
        "suc_80",
        "suc_85",
        "suc_90",
        "suc_95",
    ]
else:
    cols = [
        "model",
        "dataset",
        "metric",
        "train",
        "valid",
        "test",
        "mae_1",
        "mae_2",
        "mae_3",
        "mae_4",
        "mae_5",
        "mae_6",
    ]
df_mean = df[cols].groupby(["model", "dataset", "metric"]).mean().round(4)
df_std = df[cols].groupby(["model", "dataset", "metric"]).std().round(4)

df_mean = df_mean.reset_index()
df_std = df_std.reset_index()
df_summary = df_mean[["model", "dataset", "metric"]].copy()
if "auc" in metric:
    for col in [
        "train",
        "valid",
        "test",
        "suc_80",
        "suc_85",
        "suc_90",
        "suc_95",
    ]:
        df_summary[col] = (
            df_mean[col].astype(str) + "±" + df_std[col].astype(str)
        )
else:
    for col in [
        "train",
        "valid",
        "test",
        "mae_1",
        "mae_2",
        "mae_3",
        "mae_4",
        "mae_5",
        "mae_6",
    ]:
        df_summary[col] = (
            df_mean[col].astype(str) + "±" + df_std[col].astype(str)
        )

summary_all = f"output/{args.dataset}/summary_all.csv"
if os.path.exists(summary_all):
    df_summary.to_csv(summary_all, mode="a", header=False, index=False)
else:
    df_summary.to_csv(summary_all, index=False)
print(df_summary)