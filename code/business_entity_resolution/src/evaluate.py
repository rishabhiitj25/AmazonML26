"""Local scorer: F0.5 per Source 1 entity, macro-averaged over all S1 entities (singletons included)."""
import numpy as np
import pandas as pd

BETA2 = 0.25


def entity_f05(pred: set, truth: set) -> float:
    if not truth:
        return 1.0 if not pred else 0.0
    if not pred:
        return 0.0  # true matches exist but nothing predicted: precision undefined, scored 0
    tp = len(pred & truth)
    if tp == 0:
        return 0.0
    p, r = tp / len(pred), tp / len(truth)
    return (1 + BETA2) * p * r / (BETA2 * p + r)


def score(pred: dict, gt: dict, country: dict | None = None, weights: dict | None = None) -> dict:
    """pred/gt: s1 -> iterable of ids. Scored over every S1 in gt.
    country: optional s1 -> label for a breakdown; weights: optional label -> weight for a reweighted mean."""
    s1s = list(gt)
    f = np.array([entity_f05(set(pred.get(s, ())), set(gt[s])) for s in s1s])
    single = np.array([len(gt[s]) == 0 for s in s1s])
    out = {"f05": f.mean(), "n": len(s1s),
           "f05_singleton": f[single].mean() if single.any() else np.nan,
           "f05_matched": f[~single].mean() if (~single).any() else np.nan}
    if country is not None:
        c = pd.Series([country[s] for s in s1s])
        by = pd.Series(f).groupby(c).mean()
        out["by_country"] = by.to_dict()
        if weights:
            w = {k: v for k, v in weights.items() if k in by.index}
            out["f05_reweighted"] = sum(by[k] * v for k, v in w.items()) / sum(w.values())
    return out
