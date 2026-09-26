"""End-to-end run of src.pipeline (--mode full, CPU) on a small synthetic dataset with extreme cases:
a country that exists only in test, a country with a single S1, S1 without any record, empty / placeholder
fields, non-Latin names, quoted fields. Outputs must pass the official validator and the subset rule."""
import csv
import random
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
VALIDATOR = ROOT.parents[1] / "student_resource" / "utils" / "validate_submission.py"
WORDS = ["alpha", "bright", "cedar", "delta", "ember", "falcon", "granite", "harbor", "iris", "juniper", "kite",
         "lotus", "maple", "nova", "orchid", "pine", "quartz", "river", "summit", "tulip", "umber", "violet"]
KINDS = {"US": ["LLC", "Inc", "Corp"], "India": ["Private Limited", "Pvt Ltd", "LLP"], "France": ["SARL", "SAS"],
         "Tinyland": ["Co"]}
STREETS = {"US": "Main St", "India": "MG Road", "France": "rue de la Paix", "Tinyland": "Harbour Way"}


def _entity(rng, country):
    name = f"{rng.choice(WORDS).title()} {rng.choice(WORDS).title()} {rng.choice(KINDS[country])}"
    addr = f"{rng.randint(1, 999)} {STREETS[country]}, City{rng.randint(1, 30)}, {rng.randint(10000, 99999)}"
    return name, addr


def _variant(rng, name, addr):
    r = rng.random()
    if r < 0.2:
        return name.upper(), addr.replace(",", "")
    if r < 0.35:
        return name, "<NULL>"
    if r < 0.45:
        return name.replace("a", "@", 1), addr
    if r < 0.55:
        return '"' + name + '"', addr                     # quoted field content
    return name, addr


def _write(path, rows, header):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n", quoting=csv.QUOTE_MINIMAL)
        w.writerow(header)
        w.writerows(rows)


def make_split(d: Path, split: str, countries: dict, seed: int):
    rng = random.Random(seed)
    s1, s2, s3, gt = [], [], [], []
    n = 0
    for country, n_s1 in countries.items():
        for _ in range(n_s1):
            n += 1
            sid = f"S1-{split}{n}"
            name, addr = _entity(rng, country)
            if rng.random() < 0.05:
                addr = ""                                        # empty address
            s1.append((sid, name, addr, country))
            matches = []
            for j in range(rng.choice([0, 1, 1, 2, 3])):          # 0 -> singleton
                src = 2 if rng.random() < 0.5 else 3
                rid = f"S{src}-{split}{n}x{j}"
                vn, va = _variant(rng, name, addr)
                (s2 if src == 2 else s3).append((rid, vn, va, country))
                matches.append(rid)
            gt.append((sid, ",".join(matches)))
        for j in range(n_s1 // 3):                                # unmatched distractors
            name, addr = _entity(rng, country)
            s3.append((f"S3-{split}u{country}{j}", name, addr, country))
    s2.append((f"S2-{split}hindi", "शर्मा ट्रेडर्स", "12 MG Road", "India"))  # non-Latin, unmatched
    s3.append((f"S3-{split}empty", "", "", "US"))                               # fully empty record
    d.mkdir(parents=True, exist_ok=True)
    h = ["entity_id", "business_name", "business_address", "country"]
    _write(d / f"{split}_source1.tsv", s1, h)
    _write(d / f"{split}_source2.tsv", s2, h)
    _write(d / f"{split}_source3.tsv", s3, h)
    if split == "train":
        _write(d / "train_ground_truth.tsv", gt, ["source1_entity_id", "matched_entity_ids"])
    return {r[0] for r in s1}


def _gpus():
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


@pytest.mark.parametrize("gpus", ["", pytest.param("cuda:0", marks=pytest.mark.skipif(not _gpus(), reason="no GPU"))])
def test_pipeline_end_to_end(tmp_path, gpus):
    data = tmp_path / "dataset"
    make_split(data / "train", "train", {"US": 300, "India": 300}, 0)
    test_s1 = make_split(data / "test", "test", {"US": 120, "India": 120, "France": 120, "Tinyland": 1}, 1)
    out, work = tmp_path / "out", tmp_path / "work"
    r = subprocess.run([sys.executable, "-m", "src.pipeline", "--mode", "full", "--data-dir", str(data),
                        "--work", str(work), "--out", str(out), "--gpus", gpus, "--threads", "4",
                        "--artifacts", str(tmp_path / "art")],
                       cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    v = subprocess.run([sys.executable, str(VALIDATOR), "--matching", str(out / "matching_results.tsv"),
                        "--candidate", str(out / "candidate_pairs.tsv"), "--test-dir", str(data / "test"),
                        "--check-ids"], capture_output=True, text=True)
    assert v.returncode == 0, v.stdout + v.stderr

    def read(p):
        rows = [l.rstrip("\n").split("\t") for l in open(p, encoding="utf-8")][1:]
        return {a: set(filter(None, b.split(","))) for a, b in rows}
    m, c = read(out / "matching_results.tsv"), read(out / "candidate_pairs.tsv")
    assert set(m) == test_s1 == set(c)
    assert all(m[s] <= c[s] for s in m)
    assigned = [x for s in m.values() for x in s]
    assert len(assigned) == len(set(assigned))                  # each record matched to at most one S1
    # inference mode from the exported artifacts reproduces the same file
    out2 = tmp_path / "out2"
    r = subprocess.run([sys.executable, "-m", "src.pipeline", "--mode", "inference", "--data-dir", str(data),
                        "--work", str(tmp_path / "work2"), "--out", str(out2), "--gpus", gpus, "--threads", "4",
                        "--artifacts", str(tmp_path / "art")], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    assert (out2 / "matching_results.tsv").read_bytes() == (out / "matching_results.tsv").read_bytes()
