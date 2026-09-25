"""Pairwise GBDT matcher (XGBoost, GPU) with K-fold out-of-fold predictions on train.
Folds are grouped by S1 entity (all pairs of one S1 share a fold); one fold per GPU in parallel."""
import argparse
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

NON_FEATURES = {"ri", "si", "y", "fold", "p"}

PARAMS = dict(objective="binary:logistic", eval_metric="logloss", tree_method="hist", max_depth=10,
              learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
              reg_lambda=1.0, max_bin=256)


def fold_of(si: np.ndarray, n_folds: int) -> np.ndarray:
    return ((si.astype(np.uint64) * np.uint64(2654435761)) % np.uint64(2**32) % np.uint64(n_folds)).astype(np.int8)


def feature_cols(df: pd.DataFrame):
    return [c for c in df.columns if c not in NON_FEATURES]


def sample_train(df: pd.DataFrame, neg_rate: float, seed: int):
    rng = np.random.default_rng(seed)
    keep = (df.y.values == 1) | (rng.random(len(df)) < neg_rate)
    d = df[keep]
    w = np.where(d.y.values == 1, 1.0, 1.0 / neg_rate).astype(np.float32)
    return d, w


def train_one(df, cols, neg_rate, rounds, device, seed=0):
    d, w = sample_train(df, neg_rate, seed)
    dm = xgb.QuantileDMatrix(d[cols], label=d.y.values, weight=w, max_bin=PARAMS["max_bin"])
    return xgb.train({**PARAMS, "device": device, "seed": seed}, dm, num_boost_round=rounds)


def predict_booster(bst, X: pd.DataFrame, chunk: int = 20_000_000) -> np.ndarray:
    return np.concatenate([bst.inplace_predict(X.iloc[a:a + chunk]) for a in range(0, len(X), chunk)]
                          ).astype(np.float32)


def oof(feats: Path, out: Path, model_dir: Path, n_folds: int, neg_rate: float, rounds: int, devices):
    t = time.time()
    df = pd.read_parquet(feats)
    cols = feature_cols(df)
    fold = fold_of(df.si.values, n_folds)
    p = np.zeros(len(df), dtype=np.float32)
    model_dir.mkdir(parents=True, exist_ok=True)
    print(f"loaded {df.shape} in {time.time() - t:.0f}s", flush=True)
    def run(k):
        dev = devices[k % len(devices)]
        bst = train_one(df[fold != k], cols, neg_rate, rounds, dev, seed=k)
        bst.save_model(str(model_dir / f"xgb_fold{k}.json"))
        print(f"fold {k} trained on {dev} {time.time() - t:.0f}s", flush=True)

    todo = [k for k in range(n_folds) if not (model_dir / f"xgb_fold{k}.json").exists()]
    for a in range(0, len(todo), len(devices)):
        th = [threading.Thread(target=run, args=(k,)) for k in todo[a:a + len(devices)]]
        [x.start() for x in th]
        [x.join() for x in th]
    # predict sequentially with boosters reloaded from disk (thread-trained boosters crashed on GPU predict)
    boosters = {}
    for k in range(n_folds):
        bst = xgb.Booster(model_file=str(model_dir / f"xgb_fold{k}.json"))
        bst.set_param({"device": devices[0]})
        p[fold == k] = predict_booster(bst, df.loc[fold == k, cols])
        boosters[k] = bst
        print(f"fold {k} predicted {time.time() - t:.0f}s", flush=True)
    print(f"OOF predicted {time.time() - t:.0f}s", flush=True)
    imp = pd.Series(boosters[0].get_score(importance_type="gain")).reindex(cols).fillna(0)
    print((imp / imp.sum()).sort_values(ascending=False).round(4).to_string())
    pd.DataFrame({"ri": df.ri.values, "si": df.si.values, "y": df.y.values, "p": p}).to_parquet(out, index=False)


def predict(feats: Path, out: Path, model_dir: Path, devices):
    df = pd.read_parquet(feats)
    cols = feature_cols(df)
    ps = []
    for i, f in enumerate(sorted(model_dir.glob("xgb_fold*.json"))):
        bst = xgb.Booster(model_file=str(f))
        bst.set_param({"device": devices[0]})  # one GPU per process: multi-GPU predict from CPU data crashes
        ps.append(predict_booster(bst, df[cols]))
    pd.DataFrame({"ri": df.ri.values, "si": df.si.values, "p": np.mean(ps, axis=0).astype(np.float32)}
                 ).to_parquet(out, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["oof", "predict"])
    ap.add_argument("--feats", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--folds", type=int, default=2)
    ap.add_argument("--neg-rate", type=float, default=0.25)
    ap.add_argument("--rounds", type=int, default=600)
    ap.add_argument("--gpus", default="cuda:0,cuda:1")
    a = ap.parse_args()
    devs = [d for d in a.gpus.split(",") if d] or ["cpu"]
    if a.mode == "oof":
        oof(Path(a.feats), Path(a.out), Path(a.model_dir), a.folds, a.neg_rate, a.rounds, devs)
    else:
        predict(Path(a.feats), Path(a.out), Path(a.model_dir), devs)


if __name__ == "__main__":
    main()
