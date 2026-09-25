"""End-to-end: data -> preprocess -> blocking -> features -> XGBoost-GPU (OOF on train) -> decision tuning
-> test predictions -> output/matching_results.tsv + output/candidate_pairs.tsv.

Each stage caches its output under --work and is skipped if the file already exists."""
import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import blocking, features, model
from .decide import assign_hybrid, assign_threshold, best_per_record, score_rows
from .io_utils import CAND_HEADER, MATCH_HEADER, write_id_lists


def step(name, path: Path, fn):
    if path.exists():
        print(f"[skip] {name}: {path}", flush=True)
        return
    t = time.time()
    fn()
    print(f"[done] {name} in {time.time() - t:.0f}s", flush=True)


def apply_decision(pr: pd.DataFrame, decision: dict, n_s1: int) -> pd.DataFrame:
    best = best_per_record(pr)
    if decision["rule"] == "threshold":
        return assign_threshold(best, decision["t"])
    return assign_hybrid(pr, best, n_s1, decision["p_min"], decision["gate"])


def tune_decision(oof_path: Path, cache: Path, gt_path: Path, out: Path):
    """Grid over decision rules on OOF train predictions; keep the best macro F0.5."""
    s1, recs = features.load_tables(cache, "train")
    true_si = features.labels(recs, s1, gt_path)
    pr = pd.read_parquet(oof_path, columns=["ri", "si", "p"])
    grid = [{"rule": "threshold", "t": float(t)} for t in np.round(np.arange(0.3, 0.96, 0.05), 2)]
    grid += [{"rule": "hybrid", "p_min": pm, "gate": float(g)}
             for pm in (0.3, 0.5) for g in np.round(np.arange(0.6, 0.96, 0.05), 2)]
    res = []
    for d in grid:
        a = apply_decision(pr, d, len(s1))
        res.append({**d, "oof_f05": float(score_rows(a.ri.values, a.si.values, true_si, len(s1)).mean())})
        print(res[-1], flush=True)
    best = max(res, key=lambda r: r["oof_f05"])
    print("chosen:", best)
    json.dump(best, open(out, "w"), indent=1)


def write_outputs(cache: Path, cands_path: Path, pred_path: Path, decision: dict, out_dir: Path):
    s1_ids = pd.read_parquet(cache / "test_s1.parquet", columns=["id"]).id.values
    rec_ids = np.concatenate([pd.read_parquet(cache / f"test_s{s}.parquet", columns=["id"]).id.values
                              for s in (2, 3)])

    def group(si, ri):
        order = np.argsort(si, kind="stable")
        si, ri = si[order], ri[order]
        cut = np.flatnonzero(np.r_[True, si[1:] != si[:-1]])
        return {s1_ids[si[a]]: rec_ids[ri[a:b]] for a, b in zip(cut, np.r_[cut[1:], len(si)])}

    c = pd.read_parquet(cands_path, columns=["ri", "si"])
    write_id_lists(out_dir / "candidate_pairs.tsv", CAND_HEADER, s1_ids, group(c.si.values, c.ri.values))
    a = apply_decision(pd.read_parquet(pred_path, columns=["ri", "si", "p"]), decision, len(s1_ids))
    write_id_lists(out_dir / "matching_results.tsv", MATCH_HEADER, s1_ids, group(a.si.values, a.ri.values))
    print(f"wrote outputs: {len(c):,} candidate pairs, {len(a):,} matches, {len(s1_ids):,} S1 rows")


def export_artifacts(work: Path, dst: Path):
    """Everything inference needs, learned from train: fold models, token map, decision rule."""
    (dst / "models").mkdir(parents=True, exist_ok=True)
    for f in sorted((work / "models").glob("xgb_fold*.json")):
        shutil.copy2(f, dst / "models" / f.name)
    shutil.copy2(work / "cache" / "token_map.json", dst / "token_map.json")
    shutil.copy2(work / "decision.json", dst / "decision.json")
    print(f"artifacts exported to {dst}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, help="student_resource/dataset")
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["full", "inference"], default="full",
                    help="full: train from scratch (then export artifacts); inference: test only, using --artifacts")
    ap.add_argument("--artifacts", default=None,
                    help="dir with models/xgb_fold*.json, token_map.json, decision.json "
                         "(read in inference mode, written in full mode if given)")
    ap.add_argument("--threads", type=int, default=64)
    ap.add_argument("--gpus", default="cuda:0,cuda:1", help="comma-separated CUDA devices; '' = CPU only")
    a = ap.parse_args()
    gpus = [d for d in a.gpus.split(",") if d]
    blocking.DEVICES[:] = gpus            # empty -> CPU sparse top-k
    model_devs = gpus or ["cpu"]
    data, work, out = Path(a.data_dir).resolve(), Path(a.work).resolve(), Path(a.out).resolve()
    art = Path(a.artifacts).resolve() if a.artifacts else None
    cache = work / "cache"
    gt = data / "train" / "train_ground_truth.tsv"
    inference = a.mode == "inference"
    if inference and art is None:
        ap.error("--mode inference needs --artifacts")

    cmd = [sys.executable, "-m", "src.preprocess", "--data-dir", str(data), "--cache", str(cache)]
    if inference:
        cmd += ["--splits", "test", "--token-map", str(art / "token_map.json")]
    subprocess.run(cmd, check=True, cwd=Path(__file__).resolve().parents[1])

    for split in (("test",) if inference else ("train", "test")):
        cp = work / split / "cands.parquet"
        step(f"blocking {split}", cp, lambda: blocking.run(cache, split, cp, 10, a.threads))
        fp = work / split / "feats.parquet"
        step(f"features {split}", fp, lambda: features.build(cache, split, cp, fp,
                                                             gt if split == "train" else None, a.threads))
    if inference:
        models, dec = art / "models", art / "decision.json"
    else:
        models, dec = work / "models", work / "decision.json"
        oof = work / "train" / "oof.parquet"
        step("train + OOF", oof, lambda: model.oof(work / "train" / "feats.parquet", oof, models, 2, 0.25, 600,
                                                   model_devs))
        step("decision tuning", dec, lambda: tune_decision(oof, cache, gt, dec))
        if art is not None:
            export_artifacts(work, art)
    pred = work / "test" / "pred.parquet"
    step("test prediction", pred, lambda: model.predict(work / "test" / "feats.parquet", pred, models, model_devs))
    write_outputs(cache, work / "test" / "cands.parquet", pred, json.load(open(dec)), out)


if __name__ == "__main__":
    main()
