"""Record-side candidate generation.

For every S2/S3 record, retrieve the top-k S1 entities with the same country label using two sparse
char-3gram TF-IDF blockers: name+address, and name only (covers records with empty/partial address).
Vectorisers are fit per split and per country on that split's S1 text, so unseen countries (France)
get their own vocabulary. Output: one row per (record, S1) candidate pair.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn

DEVICES: list = []  # e.g. ["cuda:0", "cuda:1"]; empty -> CPU sparse top-k

BLOCKERS = {
    "both": lambda d: d["name"] + " | " + d["addr"],
    "name": lambda d: d["name"],
}


def make_vectorizer():
    return TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3), min_df=2, sublinear_tf=True,
                           dtype=np.float32)


def topk(Q: sp.csr_matrix, XT: sp.csr_matrix, k: int, threads: int, chunk: int = 500_000):
    qi, si, sc = [], [], []
    for a in range(0, Q.shape[0], chunk):
        R = sp_matmul_topn(Q[a:a + chunk], XT, top_n=k, sort=True, n_threads=threads).tocoo()
        qi.append(R.row.astype(np.int64) + a)
        si.append(R.col.astype(np.int64))
        sc.append(R.data.astype(np.float32))
    return np.concatenate(qi), np.concatenate(si), np.concatenate(sc)


def _to_torch_csr(M: sp.csr_matrix, device):
    import torch
    return torch.sparse_csr_tensor(torch.from_numpy(M.indptr.astype(np.int64)),
                                   torch.from_numpy(M.indices.astype(np.int64)),
                                   torch.from_numpy(M.data.astype(np.float32)),
                                   size=M.shape, device=device)


def topk_gpu(Q: sp.csr_matrix, X: sp.csr_matrix, k: int, device: str, chunk: int = 2048):
    """Exact cosine top-k (rows are L2-normalised TF-IDF): X (S1, sparse) @ dense(Q chunk).T on GPU."""
    import torch
    Xg = _to_torch_csr(X, device)
    qi, si, sc = [], [], []
    with torch.no_grad():
        for a in range(0, Q.shape[0], chunk):
            Qc = _to_torch_csr(Q[a:a + chunk], device).to_dense()
            S = torch.sparse.mm(Xg, Qc.T)                      # (n_s1, b)
            v, i = torch.topk(S, min(k, S.shape[0]), dim=0)    # (k, b), sorted desc
            v, i = v.T.contiguous().cpu().numpy(), i.T.contiguous().cpu().numpy()
            keep = v > 0
            rows = np.broadcast_to(np.arange(a, a + v.shape[0])[:, None], v.shape)
            qi.append(rows[keep].astype(np.int64))
            si.append(i[keep].astype(np.int64))
            sc.append(v[keep].astype(np.float32))
    return np.concatenate(qi), np.concatenate(si), np.concatenate(sc)


def _dense_on_gpu(M: sp.csr_matrix, device, rows=50_000):
    import torch
    parts = [_to_torch_csr(M[a:a + rows], device).to_dense().half() for a in range(0, M.shape[0], rows)]
    return torch.cat(parts)


def topk_dense_multi_gpu(Q: sp.csr_matrix, X: sp.csr_matrix, k: int, devices, chunk: int = 4096):
    """Top-k inner products with S1 rows sharded densely (fp16) across GPUs; queries streamed.
    Used only to rank candidates; exact float32 cosines are recomputed afterwards."""
    import threading

    import torch
    n = X.shape[0]
    bounds = np.linspace(0, n, len(devices) + 1, dtype=int)
    nq = Q.shape[0]
    vals = [np.zeros((nq, k), np.float32) for _ in devices]
    idxs = [np.zeros((nq, k), np.int64) for _ in devices]

    def work(g, dev):
        with torch.no_grad(), torch.cuda.device(dev):
            Xd = _dense_on_gpu(X[bounds[g]:bounds[g + 1]], dev)
            kk = min(k, Xd.shape[0])
            for a in range(0, nq, chunk):
                Qh = _to_torch_csr(Q[a:a + chunk], dev).to_dense().half()
                v, i = torch.topk(Qh @ Xd.T, kk, dim=1)
                vals[g][a:a + len(v), :kk] = v.float().cpu().numpy()
                idxs[g][a:a + len(v), :kk] = i.cpu().numpy() + bounds[g]
            del Xd
            torch.cuda.empty_cache()

    th = [threading.Thread(target=work, args=(g, d)) for g, d in enumerate(devices)]
    [t.start() for t in th]
    [t.join() for t in th]
    V, I = np.concatenate(vals, axis=1), np.concatenate(idxs, axis=1)
    order = np.argsort(-V, axis=1, kind="stable")[:, :k]
    V, I = np.take_along_axis(V, order, 1), np.take_along_axis(I, order, 1)
    keep = V > 0
    rows = np.broadcast_to(np.arange(nq)[:, None], V.shape)
    return rows[keep].astype(np.int64), I[keep].astype(np.int64), V[keep]


def rowwise_dot(Q: sp.csr_matrix, X: sp.csr_matrix, qi, si, chunk: int = 5_000_000):
    out = np.empty(len(qi), dtype=np.float32)
    for a in range(0, len(qi), chunk):
        b = a + chunk
        out[a:b] = np.asarray(Q[qi[a:b]].multiply(X[si[a:b]]).sum(axis=1)).ravel()
    return out


def block_country(s1: pd.DataFrame, recs: pd.DataFrame, k: int, threads: int) -> pd.DataFrame:
    mats, found = {}, []
    for bname, text in BLOCKERS.items():
        t = time.time()
        vec = make_vectorizer()
        X = vec.fit_transform(text(s1)).tocsr()
        Q = vec.transform(text(recs)).tocsr()
        mats[bname] = (Q, X)
        t1 = time.time()
        if DEVICES:
            qi, si, sc = topk_dense_multi_gpu(Q, X, k, DEVICES)
        else:
            qi, si, sc = topk(Q, X.T.tocsr(), k, threads)
        print(f"    {bname}: vectorise {t1 - t:.0f}s, top-k {time.time() - t1:.0f}s", flush=True)
        rank = (np.arange(len(qi)) - np.searchsorted(qi, qi, side="left")).astype(np.int16)
        found.append(pd.DataFrame({"qi": qi, "si": si, f"rank_{bname}": rank}))
    pairs = found[0].merge(found[1], on=["qi", "si"], how="outer")
    qi, si = pairs.qi.values, pairs.si.values
    for bname, (Q, X) in mats.items():
        pairs[f"cos_{bname}"] = rowwise_dot(Q, X, qi, si)
    for bname in BLOCKERS:
        pairs[f"rank_{bname}"] = pairs[f"rank_{bname}"].fillna(k).astype(np.int16)
    # ri / si: row positions in the split's concatenated S2+S3 table / S1 table
    pairs["ri"] = recs["row"].values[qi].astype(np.int32)
    pairs["si"] = s1["row"].values[si].astype(np.int32)
    return pairs.drop(columns=["qi"])


def run(cache: Path, split: str, out: Path, k: int, threads: int):
    s1 = pd.read_parquet(cache / f"{split}_s1.parquet", columns=["id", "name", "addr", "country"])
    recs = pd.concat([pd.read_parquet(cache / f"{split}_s{s}.parquet", columns=["id", "name", "addr", "country"])
                      for s in (2, 3)], ignore_index=True)
    s1["row"] = np.arange(len(s1))
    recs["row"] = np.arange(len(recs))
    out.parent.mkdir(parents=True, exist_ok=True)
    parts = []
    for country in sorted(s1.country.unique()):
        part_path = out.parent / f"{out.stem}.{country}.part.parquet"
        if part_path.exists():
            parts.append(pd.read_parquet(part_path))
            continue
        t = time.time()
        a = s1[s1.country == country].reset_index(drop=True)
        b = recs[recs.country == country].reset_index(drop=True)
        p = block_country(a, b, k, threads)
        p.to_parquet(part_path, index=False)
        parts.append(p)
        print(f"[{split}] {country}: S1={len(a):,} recs={len(b):,} pairs={len(p):,} "
              f"({len(p) / max(len(b), 1):.1f}/rec) {time.time() - t:.0f}s", flush=True)
    pairs = pd.concat(parts, ignore_index=True)
    pairs.to_parquet(out, index=False)
    for f in out.parent.glob(f"{out.stem}.*.part.parquet"):
        f.unlink()
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--split", choices=["train", "test"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--threads", type=int, default=64)
    ap.add_argument("--gpus", default="", help="comma-separated CUDA devices, e.g. cuda:0,cuda:1")
    args = ap.parse_args()
    DEVICES[:] = [d for d in args.gpus.split(",") if d]
    run(Path(args.cache), args.split, Path(args.out), args.k, args.threads)


if __name__ == "__main__":
    main()
