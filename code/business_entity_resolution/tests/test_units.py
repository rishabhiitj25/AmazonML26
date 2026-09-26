"""Unit tests for each pipeline stage, including edge cases (empty fields, placeholders, non-Latin text,
quoted TSV fields, singletons, ties, S1 without candidates)."""
import numpy as np
import pandas as pd
import pytest

from src import evaluate, normalize, xfeatures
from src.decide import assign_hybrid, assign_threshold, best_per_record, f05_arrays, score_rows
from src.features import _gap_to_best_other
from src.io_utils import read_ground_truth, read_source, write_id_lists
from src.token_map import apply_token_map, learn_token_map


# ---------------------------------------------------------------- io
def test_read_source_quoting_and_na(tmp_path):
    p = tmp_path / "s.tsv"
    p.write_text('entity_id\tbusiness_name\tbusiness_address\tcountry\n'
                 'a1\t"Joe ""The Man"" Pizza"\t<NULL>\tUS\n'
                 'a2\tNA\tNone\tIndia\n'
                 'a3\t\t\tFrance\n', encoding="utf-8")
    d = read_source(p)
    assert d.business_name.tolist() == ['Joe "The Man" Pizza', "NA", ""]
    assert d.business_address.tolist() == ["<NULL>", "None", ""]   # no NA conversion
    assert d.country.tolist() == ["US", "India", "France"]


def test_read_source_rejects_duplicates(tmp_path):
    p = tmp_path / "s.tsv"
    p.write_text("entity_id\tbusiness_name\tbusiness_address\tcountry\na\tx\ty\tUS\na\tx\ty\tUS\n")
    with pytest.raises(ValueError):
        read_source(p)


def test_ground_truth_parsing(tmp_path):
    p = tmp_path / "gt.tsv"
    p.write_text("source1_entity_id\tmatched_entity_ids\ns1\tr1, r2\ns2\t\n")
    assert read_ground_truth(p) == {"s1": ["r1", "r2"], "s2": []}


def test_writer_order_dedup_empty(tmp_path):
    p = tmp_path / "m.tsv"
    write_id_lists(p, ("source1_entity_id", "matched_entity_ids"), ["b", "a", "c"], {"a": ["x", "y", "x"], "b": []})
    assert p.read_bytes() == b"source1_entity_id\tmatched_entity_ids\nb\t\na\tx,y\nc\t\n"


# ---------------------------------------------------------------- normalisation
@pytest.mark.parametrize("raw,exp", [
    ("Café  Crème S.A.R.L.", "cafe creme s a r l"),
    ("<NULL>", ""),
    ("", ""),
    ("A&B Traders", "a and b traders"),
    ("ＡＢＣ　Ltd", "abc ltd"),                     # full-width -> NFKC
    ("N°18 R. Bobby Sands", "n 18 r bobby sands"),   # symbols are separators, not spelled out
    ("L’Atelier ¢", "l atelier"),
])
def test_norm_name(raw, exp):
    assert normalize.norm_name(raw) == exp


def test_norm_name_non_latin_is_romanised():
    out = normalize.norm_name("मुंबई ट्रेडर्स")
    assert out and out.isascii()


def test_norm_addr_drops_missing_components():
    assert normalize.norm_addr("12 Main St, null, N/A, <CITY_NAME>, 560001") == "12 main st 560001"
    assert normalize.norm_addr("null") == ""
    assert normalize.digits("12 main st 560001") == "12 560001"


# ---------------------------------------------------------------- token map
def test_token_map_learns_and_applies():
    s1 = ["Sharma Traders"] * 3 + ["Gupta Stores"]
    rec = ["शर्मा Traders"] * 3 + ["Gupta Stores"]
    tm = learn_token_map(s1, rec, min_support=3)
    assert tm == {"शर्मा": "sharma"}
    assert apply_token_map("शर्मा Foods", tm) == "sharma Foods"
    assert apply_token_map("Plain Latin", tm) == "Plain Latin"


# ---------------------------------------------------------------- metric
def test_entity_f05_conventions():
    assert evaluate.entity_f05(set(), set()) == 1.0          # singleton, empty prediction
    assert evaluate.entity_f05({"x"}, set()) == 0.0          # singleton, any prediction
    assert evaluate.entity_f05(set(), {"x"}) == 0.0          # matched, empty prediction
    assert evaluate.entity_f05({"y"}, {"x"}) == 0.0
    p, r = 1 / 2, 1 / 3
    assert evaluate.entity_f05({"a", "z"}, {"a", "b", "c"}) == pytest.approx(1.25 * p * r / (0.25 * p + r))


def test_vectorised_f05_matches_reference():
    rng = np.random.default_rng(0)
    n_true = rng.integers(0, 5, 2000)
    n_pred = rng.integers(0, 5, 2000)
    tp = np.minimum(rng.integers(0, 5, 2000), np.minimum(n_true, n_pred))
    f = f05_arrays(n_pred, n_true, tp)
    for i in range(2000):
        truth = {f"t{j}" for j in range(n_true[i])}
        pred = {f"t{j}" for j in range(tp[i])} | {f"f{j}" for j in range(n_pred[i] - tp[i])}
        assert f[i] == pytest.approx(evaluate.entity_f05(pred, truth))


# ---------------------------------------------------------------- context features
def test_gap_to_best_other_bruteforce():
    rng = np.random.default_rng(1)
    key = rng.integers(0, 50, 3000)
    val = rng.random(3000).astype(np.float32)
    gap, rank, cnt = _gap_to_best_other(key, val)
    for k in range(50):
        idx = np.flatnonzero(key == k)
        for i in idx:
            others = val[idx[idx != i]]
            exp = val[i] - others.max() if len(others) else np.nan
            np.testing.assert_allclose(gap[i], exp, rtol=1e-6)
            assert cnt[i] == len(idx)
            assert rank[i] == (val[idx] > val[i]).sum()


# ---------------------------------------------------------------- decision layer
def _pairs():
    # record 0 -> S1 0 (0.9) or S1 1 (0.6); record 1 -> S1 0 (0.8); record 2 -> S1 1 (0.2); S1 2 has no candidates
    return pd.DataFrame({"ri": [0, 0, 1, 2], "si": [0, 1, 0, 1], "p": np.float32([0.9, 0.6, 0.8, 0.2])})


def test_one_to_one_and_subset():
    pr = _pairs()
    best = best_per_record(pr)
    assert best.ri.is_unique and len(best) == 3
    for a in (assign_threshold(best, 0.5), assign_hybrid(pr, best, 3, 0.3, 0.75)):
        assert a.ri.is_unique
        cand = set(zip(pr.ri, pr.si))
        assert set(zip(a.ri, a.si)) <= cand


def test_hybrid_gate_keeps_low_confidence_s1_empty():
    pr = _pairs()
    a = assign_hybrid(pr, best_per_record(pr), 3, 0.1, 0.75)
    assert set(a.si) == {0}                 # S1 1's best record has p=0.2 < gate


def test_score_rows_singletons_and_no_candidates():
    true_si = np.array([0, 0, -1], np.int32)          # S1 0 has 2 matches; S1 1, 2 are singletons
    f = score_rows(np.array([0]), np.array([0]), true_si, 3)
    assert f[1] == 1.0 and f[2] == 1.0 and f[0] == pytest.approx(1.25 * 1 * 0.5 / (0.25 + 0.5))


# ---------------------------------------------------------------- token-level features
def test_xfeature_digit_relations():
    row = dict(zip(xfeatures.DIG_COLS, xfeatures.digit_feats("006 560001", "6 560001")))
    assert row["hn_eqz"] == 1.0 and row["dz_rex"] == 1.0          # leading zeros ignored
    row = dict(zip(xfeatures.DIG_COLS, xfeatures.digit_feats("1234", "123")))
    assert row["hn_rel"] == 1.0                                    # truncation
    row = dict(zip(xfeatures.DIG_COLS, xfeatures.digit_feats("125", "123")))
    assert row["hn_rel"] == 2.0 and row["dz_nearonly"] == 1.0      # near-miss number
    assert all(np.isnan(xfeatures.digit_feats("", "12")))


def test_xfeature_block_edge_cases():
    st = xfeatures.country_stats(np.array(["alpha traders", "beta traders", "gamma ltd"]),
                                 np.array(["1 main st", "2 main st", ""]))
    xfeatures._init(st)
    rn = np.array(["alpha traders", "", "alpha", "4lpha traders"], dtype=object)
    sn = np.array(["alpha traders", "beta traders", "", "alpha traders"], dtype=object)
    e = np.array(["", "", "", ""], dtype=object)
    out = xfeatures.pair_block((rn, sn, e, e, e, e))
    assert out.shape == (4, len(xfeatures.COLS))
    f = pd.DataFrame(out, columns=xfeatures.COLS)
    assert f.x_n_cov_s[0] == 1 and f.x_core_eq[0] == 1
    assert f.x_n_cov_s[3] == 1                                      # OCR-style digit->letter
    assert np.isnan(f.x_n_cov_r[1])                                 # empty record name -> NaN, no crash


# ---------------------------------------------------------------- model I/O
def test_aligned_batches_across_row_group_layouts(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from src.model import aligned_batches, select_rows
    n = 10_007
    a = pa.table({"ri": np.arange(n), "si": np.arange(n) * 2, "f1": np.arange(n, dtype=np.float32)})
    b = pa.table({"x1": np.arange(n, dtype=np.float32) * 3})
    pq.write_table(a, tmp_path / "a.parquet", row_group_size=997)
    pq.write_table(b, tmp_path / "b.parquet", row_group_size=3001)
    paths = [tmp_path / "a.parquet", tmp_path / "b.parquet"]
    got = pd.concat(list(aligned_batches(paths, batch_rows=1234)), ignore_index=True)
    assert len(got) == n and (got.x1.values == got.f1.values * 3).all() and (got.si == got.ri * 2).all()
    keep = np.zeros(n, bool); keep[::7] = True
    s = select_rows(paths, keep, batch_rows=500)
    assert (s.ri.values == np.flatnonzero(keep)).all() and (s.x1.values == s.ri.values * 3).all()
    with pytest.raises(ValueError):
        pq.write_table(b.slice(0, n - 1), tmp_path / "c.parquet")
        next(aligned_batches([tmp_path / "a.parquet", tmp_path / "c.parquet"]))
