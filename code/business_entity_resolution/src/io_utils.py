import csv
from pathlib import Path

import pandas as pd

SOURCE_COLS = ["entity_id", "business_name", "business_address", "country"]
MATCH_HEADER = ("source1_entity_id", "matched_entity_ids")
CAND_HEADER = ("source1_entity_id", "candidate_entity_ids")


def read_source(path) -> pd.DataFrame:
    # Files use standard CSV quoting (doubled quotes); QUOTE_NONE leaves """ artefacts.
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, na_filter=False,
                     quoting=csv.QUOTE_MINIMAL, engine="c")
    if list(df.columns) != SOURCE_COLS:
        raise ValueError(f"{path}: unexpected columns {list(df.columns)}")
    if df.entity_id.duplicated().any():
        raise ValueError(f"{path}: duplicated entity_id")
    return df


def read_ground_truth(path) -> dict:
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, na_filter=False)
    return {s1: [x.strip() for x in m.split(",") if x.strip()]
            for s1, m in zip(df.source1_entity_id, df.matched_entity_ids)}


def gt_pairs(gt: dict) -> pd.DataFrame:
    rows = [(s1, r) for s1, lst in gt.items() for r in lst]
    return pd.DataFrame(rows, columns=["s1", "rid"])


def write_id_lists(path, header, s1_ids, mapping: dict):
    """One row per S1 id in s1_ids order; unique ids joined by ',' with no spaces; '\\n' endings."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\t".join(header) + "\n")
        for s1 in s1_ids:
            ids = mapping.get(s1, ())
            seen, out = set(), []
            for i in ids:
                if i not in seen:
                    seen.add(i)
                    out.append(i)
            f.write(f"{s1}\t{','.join(out)}\n")
