"""Decision layer: record-side one-to-one assignment, then per-S1 match sets."""
import numpy as np
import pandas as pd


def best_per_record(pr: pd.DataFrame) -> pd.DataFrame:
    """For each record keep its highest-probability candidate (each record belongs to <= 1 S1)."""
    order = np.lexsort((-pr.p.values, pr.ri.values))
    d = pr.iloc[order]
    first = np.r_[True, d.ri.values[1:] != d.ri.values[:-1]]
    best = d[first].copy()
    # margin to the record's second-best candidate
    second = np.full(len(best), 0.0, dtype=np.float32)
    idx = np.flatnonzero(first)
    has2 = np.r_[idx[1:], len(d)] - idx > 1
    second[has2] = d.p.values[idx[has2] + 1]
    best["p2"] = second
    return best.reset_index(drop=True)


def assign_threshold(best: pd.DataFrame, t: float) -> pd.DataFrame:
    return best[best.p.values >= t]


def assign_hybrid(pr, best, n_s1, p_min: float, gate: float, mass_scale: float = 1.0):
    """Expected-F0.5 set size, but an S1 stays empty unless its best assigned record has p >= gate."""
    a = assign_expected_f(pr, best, n_s1, p_min=p_min, mass_scale=mass_scale, use_empty=False)
    top = pd.Series(a.p.values).groupby(a.si.values).transform("max").values
    return a[top >= gate]


def assign_expected_f(pr: pd.DataFrame, best: pd.DataFrame, n_s1: int, p_min: float = 0.02,
                      mass_scale: float = 1.0, use_empty: bool = True) -> pd.DataFrame:
    """Per S1 choose the top-m of its assigned records maximising expected F0.5, or the empty set.

    Expected true-match count N = mass_scale * sum of p over ALL candidate pairs of the S1;
    E[F | top-m] ~= 1.25 * sum_{i<=m} p_i / (m + 0.25 * max(N, sum_{i<=m} p_i));
    E[F | empty] = P(no match) ~= prod(1 - p) over the S1's candidate pairs."""
    si_all, p_all = pr.si.values, pr.p.values.astype(np.float64)
    n_hat = mass_scale * np.bincount(si_all, weights=p_all, minlength=n_s1)
    log_p0 = np.bincount(si_all, weights=np.log1p(-np.clip(p_all, 0, 1 - 1e-7)), minlength=n_s1)
    p0 = np.exp(log_p0)

    b = best[best.p.values >= p_min]
    order = np.lexsort((-b.p.values, b.si.values))
    b = b.iloc[order].reset_index(drop=True)
    si, p = b.si.values, b.p.values.astype(np.float64)
    start = np.r_[True, si[1:] != si[:-1]]
    gid = np.cumsum(start) - 1
    first = np.flatnonzero(start)
    m = np.arange(len(b)) - first[gid] + 1
    cs = np.cumsum(p)
    cs = cs - np.r_[0, cs[first[1:] - 1]][gid]
    ef = 1.25 * cs / (m + 0.25 * np.maximum(n_hat[si], cs))
    # best m per S1
    best_ef = np.maximum.reduceat(ef, first)
    m_star = np.zeros(len(first), dtype=np.int64)
    for_max = ef >= best_ef[gid] - 1e-12
    # first position (smallest m) attaining the max
    pos = np.flatnonzero(for_max)
    g_of_pos = gid[pos]
    firsthit = np.r_[True, g_of_pos[1:] != g_of_pos[:-1]]
    m_star[g_of_pos[firsthit]] = m[pos[firsthit]]
    if use_empty:
        m_star[p0[si[first]] >= best_ef] = 0
    keep = m <= m_star[gid]
    return b[keep]


def f05_arrays(n_pred, n_true, tp):
    """Vectorised per-entity F0.5 with the challenge's conventions."""
    n_pred, n_true, tp = (np.asarray(x, dtype=np.float64) for x in (n_pred, n_true, tp))
    f = np.zeros_like(n_pred)
    single = n_true == 0
    f[single] = (n_pred[single] == 0).astype(np.float64)
    ok = ~single & (tp > 0)
    p, r = tp[ok] / n_pred[ok], tp[ok] / n_true[ok]
    f[ok] = 1.25 * p * r / (0.25 * p + r)
    return f


def score_rows(assigned_ri, assigned_si, true_si: np.ndarray, n_s1: int):
    """assigned_*: predicted (record row, S1 row) pairs; true_si: record row -> S1 row or -1."""
    n_pred = np.bincount(assigned_si, minlength=n_s1)
    t = true_si[true_si >= 0]
    n_true = np.bincount(t, minlength=n_s1)
    hit = true_si[assigned_ri] == assigned_si
    tp = np.bincount(assigned_si[hit], minlength=n_s1)
    return f05_arrays(n_pred, n_true, tp)
