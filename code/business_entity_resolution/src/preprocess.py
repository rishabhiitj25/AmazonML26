"""Read raw TSVs, normalise, cache to parquet: <cache>/<split>_s{1,2,3}.parquet."""
import argparse
import os
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from . import token_map
from .io_utils import gt_pairs, read_ground_truth, read_source
from .normalize import digits, has_non_latin, norm_addr, norm_name
from .token_map import apply_token_map, learn_token_map


def _norm_chunk(args):
    names, addrs = args
    nn = [norm_name(x) for x in names]
    na = [norm_addr(x) for x in addrs]
    return nn, na, [digits(x) for x in na], [has_non_latin(x) for x in names]


def normalise_df(df: pd.DataFrame, workers: int) -> pd.DataFrame:
    n = len(df)
    bounds = np.linspace(0, n, max(1, workers * 4) + 1, dtype=int)
    chunks = [(df.business_name.values[a:b].tolist(), df.business_address.values[a:b].tolist())
              for a, b in zip(bounds[:-1], bounds[1:])]
    with Pool(workers) as pool:
        res = pool.map(_norm_chunk, chunks)
    out = pd.DataFrame({
        "id": df.entity_id.values,
        "name_raw": df.business_name.values,
        "addr_raw": df.business_address.values,
        "country": df.country.values,
    })
    out["name"] = [x for r in res for x in r[0]]
    out["addr"] = [x for r in res for x in r[1]]
    out["addr_digits"] = [x for r in res for x in r[2]]
    out["name_nonlatin"] = [x for r in res for x in r[3]]
    return out


def build_token_map(data_dir: Path, cache: Path) -> dict:
    s1 = pd.read_parquet(cache / "train_s1.parquet", columns=["id", "name_raw"]).set_index("id").name_raw
    recs = pd.concat([pd.read_parquet(cache / f"train_s{s}.parquet", columns=["id", "name_raw", "name_nonlatin"])
                      for s in (2, 3)])
    recs = recs[recs.name_nonlatin].set_index("id").name_raw
    pairs = gt_pairs(read_ground_truth(data_dir / "train" / "train_ground_truth.tsv"))
    pairs = pairs[pairs.rid.isin(recs.index)]
    return learn_token_map(s1.reindex(pairs.s1).values, recs.reindex(pairs.rid).values)


def apply_map_to_cache(path: Path, tmap: dict):
    df = pd.read_parquet(path)
    if "name_mapped" in df.columns:
        return
    m = df.name_nonlatin.values
    df["name_mapped"] = False
    df.loc[m, "name"] = [norm_name(apply_token_map(x, tmap)) for x in df.name_raw.values[m]]
    df.loc[m, "name_mapped"] = True
    df.to_parquet(path, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, help="folder containing train/ and test/")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--workers", type=int, default=min(32, os.cpu_count()))
    args = ap.parse_args()
    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)
    for split in ("train", "test"):
        for s in (1, 2, 3):
            dst = cache / f"{split}_s{s}.parquet"
            if dst.exists():
                continue
            t = time.time()
            df = read_source(Path(args.data_dir) / split / f"{split}_source{s}.tsv")
            normalise_df(df, args.workers).to_parquet(dst, index=False)
            print(f"{dst.name}: {len(df):,} rows in {time.time() - t:.0f}s", flush=True)

    map_path = cache / "token_map.json"
    if map_path.exists():
        tmap = token_map.load(map_path)
    else:
        tmap = build_token_map(Path(args.data_dir), cache)
        token_map.save(tmap, map_path)
    print(f"token map: {len(tmap):,} entries", flush=True)
    for split in ("train", "test"):
        for s in (1, 2, 3):
            apply_map_to_cache(cache / f"{split}_s{s}.parquet", tmap)
    print("token map applied", flush=True)


if __name__ == "__main__":
    main()
