"""Pair features for (record ri, S1 si) candidates. Country-agnostic: no country identity features."""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz import distance, fuzz
from rapidfuzz.process import cpdist

from .io_utils import read_ground_truth

STR_FEATS = {
    "n_ratio": ("name", fuzz.ratio),
    "n_tset": ("name", fuzz.token_set_ratio),
    "n_tsort": ("name", fuzz.token_sort_ratio),
    "n_partial": ("name", fuzz.partial_ratio),
    "n_jw": ("name", distance.JaroWinkler.normalized_similarity),
    "nc_partial": ("name_c", fuzz.partial_ratio),
    "a_ratio": ("addr", fuzz.ratio),
    "a_tset": ("addr", fuzz.token_set_ratio),
    "a_partial": ("addr", fuzz.partial_ratio),
    "d_tset": ("addr_digits", fuzz.token_set_ratio),
    "d_ratio": ("addr_digits", fuzz.ratio),
}


def load_tables(cache: Path, split: str):
    cols = ["id", "name", "addr", "addr_digits", "country"]
    s1 = pd.read_parquet(cache / f"{split}_s1.parquet", columns=cols)
    parts = []
    for s in (2, 3):
        d = pd.read_parquet(cache / f"{split}_s{s}.parquet", columns=cols + ["name_mapped"])
        d["src3"] = np.int8(s == 3)
        parts.append(d)
    recs = pd.concat(parts, ignore_index=True)
    for d in (s1, recs):
        d["name_c"] = d.name.str.replace(" ", "", regex=False)
        d["first_digit"] = d.addr_digits.str.split(" ", n=1).str[0]
        d["n_tok"] = d.name.str.count(" ") + (d.name != "")
        d["a_tok"] = d.addr.str.count(" ") + (d.addr != "")
    # how many S1 in the same country share this exact normalised name (ambiguity)
    s1["twins"] = s1.groupby(["country", "name"]).name.transform("size").astype(np.int32)
    return s1, recs


def labels(recs: pd.DataFrame, s1: pd.DataFrame, gt_path: Path) -> np.ndarray:
    """rec row -> true S1 row (-1 if unmatched)."""
    gt = read_ground_truth(gt_path)
    s1pos = pd.Series(np.arange(len(s1)), index=s1.id)
    rid, sid = zip(*[(r, s) for s, lst in gt.items() for r in lst])
    rpos = pd.Series(np.arange(len(recs)), index=recs.id)
    out = np.full(len(recs), -1, dtype=np.int32)
    out[rpos.reindex(list(rid)).values] = s1pos.reindex(list(sid)).values
    return out


def _gap_to_best_other(key: np.ndarray, val: np.ndarray):
    """For each row: val - max(val of other rows with the same key); rank of val within key (0=best)."""
    order = np.lexsort((-val, key))
    k, v = key[order], val[order]
    start = np.r_[True, k[1:] != k[:-1]]
    gid = np.cumsum(start) - 1
    first_idx = np.flatnonzero(start)
    best = v[first_idx][gid]
    second_pos = first_idx + 1
    has2 = np.r_[first_idx[1:], len(k)] - first_idx > 1
    second = np.where(has2, v[np.minimum(second_pos, len(k) - 1)], np.nan)[gid]
    rank = np.arange(len(k)) - first_idx[gid]
    other = np.where(rank == 0, second, best)
    gap = np.full(len(k), np.nan, dtype=np.float32)
    gap[order] = v - other
    r = np.empty(len(k), dtype=np.int32)
    r[order] = rank
    cnt = np.empty(len(k), dtype=np.int32)
    cnt[order] = np.bincount(gid)[gid]
    return gap, r, cnt


def add_context(F: pd.DataFrame, cols=("cos_both", "n_tset", "a_tset")):
    ri, si = F.ri.values, F.si.values
    for c in cols:
        v = np.nan_to_num(F[c].values.astype(np.float32), nan=-1.0)
        F[f"rg_{c}"], F[f"rr_{c}"], cnt_r = _gap_to_best_other(ri, v)
        F[f"sg_{c}"], F[f"sr_{c}"], cnt_s = _gap_to_best_other(si, v)
    F["r_ncand"] = cnt_r
    F["s_ncand"] = cnt_s
    return F


def build(cache: Path, split: str, cands: Path, out: Path, gt_path: Path | None, workers: int):
    t = time.time()
    s1, recs = load_tables(cache, split)
    F = pd.read_parquet(cands)
    ri, si = F.ri.values, F.si.values
    print(f"[{split}] {len(F):,} pairs; tables loaded {time.time() - t:.0f}s", flush=True)
    for fname, (col, scorer) in STR_FEATS.items():
        a, b = recs[col].values[ri], s1[col].values[si]
        v = cpdist(a, b, scorer=scorer, workers=workers, dtype=np.float32)
        if col in ("addr", "addr_digits"):
            v[(recs[col].values == "")[ri] | (s1[col].values == "")[si]] = np.nan
        F[fname] = v
        print(f"  {fname} {time.time() - t:.0f}s", flush=True)
    fd_r, fd_s = recs.first_digit.values[ri], s1.first_digit.values[si]
    F["d_first_eq"] = np.where((fd_r == "") | (fd_s == ""), np.nan, (fd_r == fd_s)).astype(np.float32)
    F["r_ntok"] = recs.n_tok.values[ri].astype(np.int16)
    F["s_ntok"] = s1.n_tok.values[si].astype(np.int16)
    F["r_atok"] = recs.a_tok.values[ri].astype(np.int16)
    F["s_atok"] = s1.a_tok.values[si].astype(np.int16)
    F["r_mapped"] = recs.name_mapped.values[ri].astype(np.int8)
    F["r_src3"] = recs.src3.values[ri]
    F["s_twins"] = s1.twins.values[si]
    F["name_eq"] = (recs.name.values[ri] == s1.name.values[si]).astype(np.int8)
    F = add_context(F)
    if gt_path is not None:
        true_si = labels(recs, s1, gt_path)
        F["y"] = (true_si[ri] == si).astype(np.int8)
    out.parent.mkdir(parents=True, exist_ok=True)
    F.to_parquet(out, index=False)
    print(f"[{split}] features {F.shape} written in {time.time() - t:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--split", choices=["train", "test"], required=True)
    ap.add_argument("--cands", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gt", default=None)
    ap.add_argument("--workers", type=int, default=64)
    args = ap.parse_args()
    build(Path(args.cache), args.split, Path(args.cands), Path(args.out),
          Path(args.gt) if args.gt else None, args.workers)


if __name__ == "__main__":
    main()
