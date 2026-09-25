"""Learn a non-Latin-token -> Latin-token map from training pairs (provided data only).

When a matched record's raw name has the same token count as its S1 name, tokens are aligned by
position; each non-Latin record token votes for the aligned S1 token. Majority vote with support and
purity thresholds gives the map. Unmapped tokens fall back to generic romanisation (anyascii).
"""
import json
import re
import unicodedata
from collections import Counter, defaultdict

from .normalize import has_non_latin, norm_name

_SPLIT = re.compile(r"[\s,.;:()\[\]{}\"'|/\\-]+")


def raw_tokens(s: str):
    return [t for t in _SPLIT.split(unicodedata.normalize("NFKC", s)) if t]


def learn_token_map(s1_names, rec_names, min_support=3, min_purity=0.6) -> dict:
    votes = defaultdict(Counter)
    for a, b in zip(s1_names, rec_names):
        if not has_non_latin(b):
            continue
        ta = norm_name(a).split()
        tb = raw_tokens(b)
        if len(ta) != len(tb):
            continue
        for x, y in zip(tb, ta):
            if has_non_latin(x):
                votes[x][y] += 1
    out = {}
    for tok, c in votes.items():
        (best, n), total = c.most_common(1)[0], sum(c.values())
        if n >= min_support and n / total >= min_purity:
            out[tok] = best
    return out


def apply_token_map(raw: str, tmap: dict) -> str:
    if not tmap or not has_non_latin(raw):
        return raw
    return " ".join(tmap.get(t, t) for t in raw_tokens(raw))


def save(tmap: dict, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(tmap, f, ensure_ascii=False, indent=0, sort_keys=True)


def load(path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)
