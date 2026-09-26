"""Token-level pair features (second feature block, row-aligned with feats.parquet).

Targets the observed failure modes of the base features: near-miss records (same business pattern with one
number or one distinctive name token changed) and equivalent spellings (truncations, OCR-style digit/letter
swaps, leading zeros).

Everything is learned from the split's own S1 table, per country label, so an unseen country gets its own
statistics: token IDF (name, address) and a "frequent token" set (tokens in >= STOP_DF of that country's S1
names and >= STOP_MIN names, plus 1-letter tokens) that is excluded from the "core" name. No hand-written word lists."""
import argparse
import math
import re
import time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein

from .features import _gap_to_best_other

STOP_DF = 0.02          # a name token in >= 2% of a country's S1 names is "frequent" (legal form, generic word)
STOP_MIN = 50           # ... and in at least this many names (tiny countries: no frequency evidence)
FUZZY_SIM = 0.75        # token pairs (len >= 4) at this normalised Levenshtein similarity count as fuzzy matches
FUZZY_W = 0.8           # coverage credit of a fuzzy token match (exact = 1)
_OCR = str.maketrans({"0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "7": "t", "8": "b", "9": "g"})
_HASD = re.compile(r"\d")
_HASA = re.compile(r"[a-z]")

DIG_COLS = ["dz_rex", "dz_rtr", "dz_rnear", "dz_rnone", "dz_sex", "hn_rel", "hn_logdiff", "hn_reldiff",
            "dz_minnear_log", "hn_eqz", "dz_nearonly"]
NAME_COLS = ["n_cov_s", "n_excov_s", "n_cov_r", "n_excov_r", "n_unm_s_maxidf", "n_unm_r_maxidf", "n_unm_s_cnt",
             "n_unm_r_cnt", "core_ratio", "core_tset", "core_eq", "core_first_eq", "ocr_ratio", "n_idf_sum_s"]
ADDR_COLS = ["a_cov_s", "a_cov_r", "a_excov_s", "a_unm_s_maxidf"]
COLS = ["x_" + c for c in DIG_COLS + NAME_COLS + ADDR_COLS]
CTX = ("x_n_cov_s", "x_dz_rex", "x_a_cov_s", "x_hn_eqz")   # record-side gap/rank context (density-invariant)

G = {}


def ocr_tok(t: str) -> str:
    """Mixed letter+digit tokens: map look-alike digits to letters (0->o, 1->l, ...)."""
    return t.translate(_OCR) if _HASD.search(t) and _HASA.search(t) else t


def country_stats(names: np.ndarray, addrs: np.ndarray) -> dict:
    """IDF of name/address tokens and the frequent-name-token set, from one country's S1 table."""
    n = len(names)
    cn, ca = Counter(), Counter()
    for s in names:
        cn.update({ocr_tok(t) for t in s.split()})
    for s in addrs:
        ca.update({t for t in s.split() if not t.isdigit()})
    return {"idf_n": {k: math.log(n / v) for k, v in cn.items()},
            "idf_a": {k: math.log(n / v) for k, v in ca.items()},
            "dflt": math.log(n),
            "stop": {k for k, v in cn.items() if v >= max(STOP_DF * n, STOP_MIN) or len(k) == 1}}


def _init(stats):
    G.update(stats)


def _tok_match(a, b):
    return len(a) >= 4 and len(b) >= 4 and Levenshtein.normalized_similarity(a, b) >= FUZZY_SIM


def soft_cover(A, B, idf, dflt):
    """IDF mass of tokens A covered by B (exact 1, fuzzy FUZZY_W): (cover, exact cover, #uncovered, max idf uncovered)."""
    tot = cov = ex = 0.0
    n_un, mx = 0, 0.0
    Bs = set(B)
    for t in A:
        w = idf.get(t, dflt)
        tot += w
        if t in Bs:
            cov += w
            ex += w
        elif any(_tok_match(t, u) for u in B):
            cov += FUZZY_W * w
        else:
            n_un += 1
            mx = max(mx, w)
    if tot == 0:
        return math.nan, math.nan, n_un, mx
    return cov / tot, ex / tot, n_un, mx


def _dnum(t):
    t = t.lstrip("0")
    return t or "0"


def _num_rel(a, b):
    """0 equal, 1 truncation (one is a prefix/suffix of the other), 2 same length, 3 other."""
    if a == b:
        return 0
    if len(a) != len(b) and (a.startswith(b) or b.startswith(a) or a.endswith(b) or b.endswith(a)):
        return 1
    return 2 if len(a) == len(b) else 3


def digit_feats(rd, sd):
    R = [_dnum(t) for t in rd.split()]
    S = [_dnum(t) for t in sd.split()]
    if not R or not S:
        return [math.nan] * len(DIG_COLS)
    Ss, Rs = set(S), set(R)
    n_ex = n_tr = n_near = n_none = 0
    min_near = None
    for t in R:
        if t in Ss:
            n_ex += 1
            continue
        best = 3
        for u in S:
            r = _num_rel(t, u)
            best = min(best, r)
            if r == 2:
                d = abs(int(t[:12]) - int(u[:12]))
                min_near = d if min_near is None else min(min_near, d)
        n_tr += best == 1
        n_near += best == 2
        n_none += best == 3
    nR = len(R)
    hd = abs(int(R[0][:12]) - int(S[0][:12]))
    return [n_ex / nR, n_tr / nR, n_near / nR, n_none / nR, sum(u in Rs for u in S) / len(S),
            float(_num_rel(R[0], S[0])), math.log1p(hd), hd / max(int(R[0][:12]), int(S[0][:12]), 1),
            math.nan if min_near is None else math.log1p(min_near), float(R[0] == S[0]),
            float(n_near > 0 and n_ex == 0)]


def pair_block(args):
    rn, sn, ra, sa, rd, sd = args
    idf_n, idf_a, dflt, stop = G["idf_n"], G["idf_a"], G["dflt"], G["stop"]
    out = np.full((len(rn), len(COLS)), np.nan, dtype=np.float32)
    for i in range(len(rn)):
        row = digit_feats(rd[i], sd[i])
        R = [ocr_tok(t) for t in rn[i].split()]
        S = [ocr_tok(t) for t in sn[i].split()]
        Rc = [t for t in R if t not in stop]
        Sc = [t for t in S if t not in stop]
        c_s, e_s, u_s, m_s = soft_cover(Sc, Rc, idf_n, dflt)
        c_r, e_r, u_r, m_r = soft_cover(Rc, Sc, idf_n, dflt)
        cr, cs = " ".join(Rc), " ".join(Sc)
        row += [c_s, e_s, c_r, e_r, m_s, m_r, u_s, u_r,
                fuzz.ratio(cr, cs), fuzz.token_set_ratio(cr, cs), float(cr == cs and cr != ""),
                float(bool(Rc) and bool(Sc) and (Rc[0] == Sc[0] or _tok_match(Rc[0], Sc[0]))),
                fuzz.ratio(" ".join(R), " ".join(S)), sum(idf_n.get(t, dflt) for t in Sc)]
        RA = [t for t in ra[i].split() if not t.isdigit()]
        SA = [t for t in sa[i].split() if not t.isdigit()]
        if RA and SA:
            ac_s, ae_s, _, amx = soft_cover(SA, RA, idf_a, dflt)
            ac_r = soft_cover(RA, SA, idf_a, dflt)[0]
            row += [ac_s, ac_r, ae_s, amx]
        else:
            row += [math.nan] * 4
        out[i] = row
    return out


def compute(ri, si, s1: pd.DataFrame, recs: pd.DataFrame, stats: dict, workers: int, chunk=100_000):
    tasks = ((recs.name.values[ri[a:a + chunk]], s1.name.values[si[a:a + chunk]],
              recs.addr.values[ri[a:a + chunk]], s1.addr.values[si[a:a + chunk]],
              recs.addr_digits.values[ri[a:a + chunk]], s1.addr_digits.values[si[a:a + chunk]])
             for a in range(0, len(ri), chunk))
    with Pool(workers, initializer=_init, initargs=(stats,)) as pool:
        X = np.vstack(list(pool.imap(pair_block, tasks, chunksize=1)))
    F = pd.DataFrame(X, columns=COLS)
    for c in CTX:
        v = np.nan_to_num(F[c].values, nan=-1.0)
        g, r, _ = _gap_to_best_other(ri, v)
        F["rg_" + c], F["rr_" + c] = g, r.astype(np.int16)
    return F


def build(cache: Path, split: str, cands: Path, out: Path, workers: int):
    t = time.time()
    cols = ["name", "addr", "addr_digits", "country"]
    s1 = pd.read_parquet(cache / f"{split}_s1.parquet", columns=cols)
    recs = pd.concat([pd.read_parquet(cache / f"{split}_s{s}.parquet", columns=cols[:3]) for s in (2, 3)],
                     ignore_index=True)
    C = pd.read_parquet(cands, columns=["ri", "si"])
    ri, si = C.ri.values, C.si.values
    cty = s1.country.values[si]
    start = np.flatnonzero(np.r_[True, cty[1:] != cty[:-1]])
    segments = list(zip(start, np.r_[start[1:], len(C)]))
    if len({cty[a] for a, _ in segments}) != len(segments):
        raise ValueError("candidate pairs are not grouped by country")
    tmp = out.with_suffix(".tmp.parquet")
    writer = None
    for a, b in segments:
        g = s1[s1.country.values == cty[a]]
        stats = country_stats(g.name.values, g.addr.values)
        part = compute(ri[a:b], si[a:b], s1, recs, stats, workers)
        tbl = pa.Table.from_pandas(part, preserve_index=False)
        writer = writer or pq.ParquetWriter(tmp, tbl.schema)
        writer.write_table(tbl)
        print(f"  [{split}] {cty[a]}: {b - a:,} pairs, {len(stats['stop'])} frequent tokens, "
              f"{time.time() - t:.0f}s", flush=True)
    writer.close()
    tmp.rename(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--split", choices=["train", "test"], required=True)
    ap.add_argument("--cands", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=64)
    a = ap.parse_args()
    build(Path(a.cache), a.split, Path(a.cands), Path(a.out), a.workers)


if __name__ == "__main__":
    main()
