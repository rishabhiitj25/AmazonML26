"""Pairwise GBDT matcher (XGBoost, GPU) with K-fold out-of-fold predictions on train.

Folds are grouped by S1 entity (all pairs of one S1 share a fold). Feature files (feats.parquet,
xfeats.parquet) are row-aligned and streamed in batches, so memory is bounded by the sampled training rows
rather than the full pair table. Boosters store their feature names; prediction selects columns by name."""
import argparse
import json
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import xgboost as xgb

NON_FEATURES = {"ri", "si", "y", "fold", "p"}

PARAMS = dict(objective="binary:logistic", eval_metric="logloss", tree_method="hist", max_depth=10,
              learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
              reg_lambda=1.0, max_bin=256)


def fold_of(si: np.ndarray, n_folds: int) -> np.ndarray:
    return ((si.astype(np.uint64) * np.uint64(2654435761)) % np.uint64(2**32) % np.uint64(n_folds)).astype(np.int8)


def feature_cols(df: pd.DataFrame):
    return [c for c in df.columns if c not in NON_FEATURES]


def aligned_batches(paths, batch_rows: int = 5_000_000, columns=None):
    """Yield DataFrames of exactly batch_rows rows (last one shorter) with the columns of all row-aligned
    parquet files side by side. columns: optional set of names to read (others skipped)."""
    files = [pq.ParquetFile(p) for p in paths]
    n = files[0].metadata.num_rows
    if any(f.metadata.num_rows != n for f in files):
        raise ValueError(f"feature files are not row-aligned: {[f.metadata.num_rows for f in files]}")
    seen = set()
    cols = []
    for f in files:
        c = [x for x in f.schema_arrow.names if x not in seen and (columns is None or x in columns)]
        seen.update(c)
        cols.append(c)
    its = [f.iter_batches(batch_size=batch_rows, columns=c) if c else None for f, c in zip(files, cols)]
    bufs = [[] for _ in files]
    have = [0] * len(files)
    for start in range(0, n, batch_rows):
        need = min(batch_rows, n - start)
        parts = []
        for j, it in enumerate(its):
            if it is None:
                continue
            while have[j] < need:
                b = next(it)
                bufs[j].append(b)
                have[j] += b.num_rows
            t = pa.Table.from_batches(bufs[j]).combine_chunks()
            parts.append(t.slice(0, need).to_pandas())
            rest = t.slice(need)
            bufs[j] = rest.to_batches() if rest.num_rows else []
            have[j] -= need
        yield pd.concat(parts, axis=1)


def select_rows(paths, keep: np.ndarray, columns=None, batch_rows: int = 5_000_000) -> pd.DataFrame:
    out, a = [], 0
    for df in aligned_batches(paths, batch_rows, columns):
        m = keep[a:a + len(df)]
        if m.any():
            out.append(df[m])
        a += len(df)
    return pd.concat(out, ignore_index=True)


def read_ids(path):
    t = pq.read_table(path, columns=["ri", "si", "y"])
    return t.column("ri").to_numpy(), t.column("si").to_numpy(), t.column("y").to_numpy()


def es_holdout(si: np.ndarray, frac: float) -> np.ndarray:
    """Hash-selected fraction of S1 entities held out of training for early stopping (independent of fold_of)."""
    return ((si.astype(np.uint64) * np.uint64(40503)) % np.uint64(1000)).astype(np.int32) < frac * 1000


def train_one(paths, keep, si, y, neg_rate, rounds, device, seed, params, es_frac=0.0, es_rounds=50):
    """Train on the kept rows. With es_frac > 0, rows of a held-out fraction of the training S1s form the
    early-stopping set (never the evaluated fold); the booster is truncated at the best iteration."""
    va = keep & es_holdout(si, es_frac) if es_frac > 0 else np.zeros_like(keep)
    d = select_rows(paths, keep | va)
    is_va = va[keep | va]
    cols = feature_cols(d)
    w = np.where(d.y.values == 1, 1.0, 1.0 / neg_rate).astype(np.float32)
    tr = ~is_va
    dm = xgb.QuantileDMatrix(d.loc[tr, cols], label=d.y.values[tr], weight=w[tr], max_bin=params["max_bin"])
    evals, kw = [], {}
    if is_va.any():
        dv = xgb.QuantileDMatrix(d.loc[is_va, cols], label=d.y.values[is_va], weight=w[is_va], ref=dm)
        evals, kw = [(dv, "es")], {"early_stopping_rounds": es_rounds, "verbose_eval": False}
    del d
    bst = xgb.train({**params, "device": device, "seed": seed}, dm, num_boost_round=rounds, evals=evals, **kw)
    if evals:
        bst = bst[: bst.best_iteration + 1]
    return bst


def oof(paths, out: Path, model_dir: Path, n_folds: int, neg_rate: float, rounds: int, devices, params=None,
        es_frac: float = 0.0):
    params = {**PARAMS, **(params or {})}
    t = time.time()
    ri, si, y = read_ids(paths[0])
    fold = fold_of(si, n_folds)
    model_dir.mkdir(parents=True, exist_ok=True)
    json.dump({"n_folds": n_folds, "neg_rate": neg_rate, "rounds": rounds, "es_frac": es_frac, "params": params},
              open(model_dir / "model_config.json", "w"), indent=1)
    errors = []

    def run(k):
        try:
            rng = np.random.default_rng(k)
            keep = (fold != k) & ((y == 1) | (rng.random(len(y)) < neg_rate))
            bst = train_one(paths, keep, si, y, neg_rate, rounds, devices[k % len(devices)], k, params, es_frac)
            bst.save_model(str(model_dir / f"xgb_fold{k}.ubj"))   # binary JSON: ~40% smaller, lossless
            print(f"fold {k} trained ({bst.num_boosted_rounds()} rounds) {time.time() - t:.0f}s", flush=True)
        except BaseException as e:
            errors.append(e)

    todo = [k for k in range(n_folds) if not any((model_dir / f"xgb_fold{k}{e}").exists() for e in (".ubj", ".json"))]
    for a in range(0, len(todo), len(devices)):
        th = [threading.Thread(target=run, args=(k,)) for k in todo[a:a + len(devices)]]
        [x.start() for x in th]
        [x.join() for x in th]
        if errors:
            raise errors[0]
    # predict with boosters reloaded from disk on one device (thread-trained boosters crashed on GPU predict)
    boosters = load_boosters(model_dir, devices[0])
    p = np.zeros(len(y), dtype=np.float32)
    a = 0
    for df in aligned_batches(paths):
        f = fold[a:a + len(df)]
        for k, bst in enumerate(boosters):
            m = f == k
            if m.any():
                p[a:a + len(df)][m] = bst.inplace_predict(df.loc[m, bst.feature_names])
        a += len(df)
    print(f"OOF predicted {time.time() - t:.0f}s", flush=True)
    imp = pd.Series(boosters[0].get_score(importance_type="gain"))
    print((imp / imp.sum()).sort_values(ascending=False).head(30).round(4).to_string())
    pd.DataFrame({"ri": ri, "si": si, "y": y, "p": p}).to_parquet(out, index=False)


def load_boosters(model_dir: Path, device):
    files = sorted([f for f in Path(model_dir).glob("xgb_fold*") if f.suffix in (".ubj", ".json")],
                   key=lambda f: int(f.stem[len("xgb_fold"):]))
    if not files or len({f.stem for f in files}) != len(files):
        raise FileNotFoundError(f"need exactly one xgb_fold<k>.ubj|.json per fold in {model_dir}: {files}")
    out = []
    for f in files:
        bst = xgb.Booster(model_file=str(f))
        bst.set_param({"device": device})  # one GPU per process: multi-GPU predict from CPU data crashes
        out.append(bst)
    return out


def predict(paths, out: Path, model_dir: Path, devices, batch_rows: int = 5_000_000):
    """Average of the fold models; feature files are streamed in batches to bound memory."""
    boosters = load_boosters(model_dir, devices[0])
    need = set(boosters[0].feature_names) | {"ri", "si"}
    parts = []
    for df in aligned_batches(paths, batch_rows, need):
        p = np.mean([b.inplace_predict(df[b.feature_names]) for b in boosters], axis=0).astype(np.float32)
        parts.append(pd.DataFrame({"ri": df.ri.values, "si": df.si.values, "p": p}))
    pd.concat(parts, ignore_index=True).to_parquet(out, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["oof", "predict"])
    ap.add_argument("--feats", required=True, help="comma-separated row-aligned feature parquet files")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--folds", type=int, default=2)
    ap.add_argument("--neg-rate", type=float, default=0.25)
    ap.add_argument("--rounds", type=int, default=600)
    ap.add_argument("--gpus", default="cuda:0,cuda:1")
    a = ap.parse_args()
    devs = [d for d in a.gpus.split(",") if d] or ["cpu"]
    paths = [Path(p) for p in a.feats.split(",")]
    if a.mode == "oof":
        oof(paths, Path(a.out), Path(a.model_dir), a.folds, a.neg_rate, a.rounds, devs)
    else:
        predict(paths, Path(a.out), Path(a.model_dir), devs)


if __name__ == "__main__":
    main()
