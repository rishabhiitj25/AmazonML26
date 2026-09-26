# Business Entity Resolution — reproducible pipeline

Produces `matching_results.tsv` and `candidate_pairs.tsv` from the challenge data only. No external data,
APIs or pretrained models; the only learned model is an XGBoost classifier (Apache-2.0) trained on the
provided training data. Trained artifacts are shipped in `artifacts/`, so the test predictions can be
reproduced without retraining.

```
business_entity_resolution/
├── src/                        # pipeline code (entry point: src/pipeline.py)
├── tests/                      # pytest suite (unit + end-to-end with extreme cases)
├── artifacts/                  # learned from train — enough for inference
│   ├── models/xgb_fold{0..4}.ubj   # 5 XGBoost fold models (binary JSON); test prediction = their average
│   ├── models/model_config.json    # K, sampling, learning rate, early-stopping settings used
│   ├── token_map.json          # non-Latin → Latin token map learned from training pairs
│   └── decision.json           # decision rule: probability threshold 0.80 (+ OOF score for reference)
├── requirements.txt
└── README.md
```

## 1. Setup

1. Python 3.10 on Linux.
2. Get the dataset from the challenge portal (it is **not** in this repo) and unzip it so you have
   `student_resource/dataset/{train,test}/*.tsv`.
3. Environment:

```bash
cd code/business_entity_resolution
python3.10 -m venv .venv && source .venv/bin/activate
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124   # CUDA 12.x driver
pip install -r requirements.txt
```

## 2. Quick start — test predictions with the shipped models (no training)

```bash
python -m src.pipeline --mode inference --artifacts artifacts \
    --data-dir /path/to/student_resource/dataset \
    --work /path/to/work --out /path/to/output \
    --gpus cuda:0 --threads 32
```

Writes `/path/to/output/matching_results.tsv` and `candidate_pairs.tsv`. Validate with the official script:

```bash
cd /path/to/student_resource
python3 utils/validate_submission.py --matching /path/to/output/matching_results.tsv \
    --candidate /path/to/output/candidate_pairs.tsv --test-dir dataset/test
```

## 3. Full run — retrain from scratch

```bash
python -m src.pipeline --mode full --artifacts artifacts_new \
    --data-dir /path/to/student_resource/dataset \
    --work /path/to/work --out /path/to/output --gpus cuda:0,cuda:1 --threads 64
```

Runs everything on train (blocking, features, token features, 5-fold XGBoost with early stopping and
out-of-fold predictions, decision-grid report), then the test side, and exports the new artifacts to
`--artifacts`. The configuration (`MODEL`, `DECISION` in `src/pipeline.py`) was selected with the protocol
in section 8.

Each stage caches its output under `--work` and is **skipped if that file already exists** — delete the
file (or use a new `--work`) to recompute a stage. Runs are resumable after interruption.

## 4. Hardware and runtime

| | Inference (`--mode inference`) | Full training (`--mode full`) |
|---|---|---|
| GPU | 1 CUDA GPU; ≥ 16 GB recommended (S1 is sharded to fit free memory) | ≥ 24 GB free per GPU (a fold trains on ~80M sampled rows) |
| RAM | ≥ 64 GB (measured peaks: blocking 25 GB, features 36 GB, token features 56 GB with 96 workers, prediction 21 GB) | ~180 GB with 2 GPUs (two folds load their rows at once), ~100 GB with 1 |
| Disk (`--work`) | ~30 GB | ~60 GB |
| Time, measured on 2× L40S / 64–96 threads | ~92 min: preprocess 2, blocking 49, features 19, token features 11, prediction 11 | + train (5 folds, early stopping) and OOF 39 min on 2 GPUs, decision grid 3.5 min |

- `--gpus ''` runs without a GPU (CPU sparse top-k blocking and CPU XGBoost) — correct but slow
  (blocking alone takes several hours).
- `--gpus cuda:0,cuda:1` splits blocking and fold training across GPUs. Prediction always uses the first GPU
  listed (XGBoost GPU prediction on a second device ordinal crashes; pin with `CUDA_VISIBLE_DEVICES`).
- Lower `--threads` on smaller machines (string features, token features and top-k use it); token-feature
  memory scales with the number of workers.

## 5. Evaluating on the training data

Local scorer: `src/evaluate.py` (exact macro F0.5, per-S1, singletons included). `--mode full` prints the
decision-rule grid with out-of-fold macro F0.5 on the whole training set. Reference result of the shipped
configuration: **OOF macro F0.5 = 0.9872** (US 0.9877, India 0.9865) at threshold 0.80; 0.9863 under
test-like distractor density; 0.9521 on an unseen country (leave-one-country-out). See section 8.

## 6. Pipeline stages

| Stage | Module | Output (`--work/…`) |
|---|---|---|
| Normalise + token map | `preprocess.py`, `normalize.py`, `token_map.py` | `cache/*.parquet`, `cache/token_map.json` |
| Blocking | `blocking.py` | `{train,test}/cands.parquet` |
| Pair features | `features.py` | `{train,test}/feats.parquet` |
| Token features | `xfeatures.py` | `{train,test}/xfeats.parquet` (row-aligned with feats) |
| XGBoost 5-fold OOF, early stopping (full mode) | `model.py` | `models/`, `train/oof.parquet` |
| Decision grid report + chosen rule (full mode) | `pipeline.py`, `decide.py` | `decision.json` |
| Test prediction + outputs | `model.py`, `pipeline.py` | `test/pred.parquet`, `--out/*.tsv` |

## 7. Tests

```bash
pip install pytest==8.3.3
python -m pytest -q tests/            # ~1 min; the GPU end-to-end case is skipped without CUDA
```

`tests/test_units.py` covers every stage with edge cases (quoted TSV fields, `NA`/`None` strings, placeholders,
full-width and non-Latin text, symbols, metric conventions for singletons, context gap/rank vs brute force,
one-to-one assignment, candidates ⊇ matches, token features, aligned streaming of the two feature files).
`tests/test_e2e.py` runs the whole pipeline (full mode, then inference from the exported artifacts) on a
synthetic dataset with a test-only country, a country with a single S1, S1 without records, empty records
and non-Latin names, and checks the result with the official validator.

## 8. Model selection (how the shipped configuration was chosen)

Plain out-of-fold F0.5 over-estimated the leaderboard and ranked a stacked model wrongly (OOF +0.0045,
leaderboard −0.004), because the test pool has ~1.9× more near-miss distractor records per S1. Every
candidate change was therefore scored on three train universes plus an unseen-country split, keeping only
changes that help in all of them:

- `base`: plain OOF (folds grouped by S1);
- `clone09`: every unmatched record gets 0.9 exact clones on average (test-like distractor density), S1-side
  context recomputed — reproduced both leaderboard scores (0.971 vs 0.969, 0.966 vs 0.965);
- `drop19`: 19% of S1 removed so their records become distractors, record-side context recomputed;
- leave-one-country-out (train US → predict India and vice versa) as a proxy for France.

Selected this way: token features (+0.015 under test density), K = 5 folds (K was increased until the next
step gained less than twice the seed noise of ±0.0001: K=10 added ≤ 0.00007), 50 % negatives, learning rate
0.05 with early stopping, and threshold 0.80 (maximising 0.85·density + 0.15·unseen-country). Rejected with
evidence: stacked second-stage models, reweighting unmatched negatives, removing S1-side context, two-level
thresholds. The experiment harness is kept out of the package; full tables are in `Documentation_template.md`.

## 9. Code map

- `io_utils.py` — TSV reading (standard CSV quoting, no NA conversion), ground-truth parsing, strict writer.
- `normalize.py` — placeholder removal (`<NULL>`, `<CITY_NAME>`, null/N/A components), NFKC, romanisation
  (anyascii) and accent folding, lowercase, punctuation → space, digit tokens.
- `token_map.py` — non-Latin → Latin token map learned from aligned training pairs.
- `blocking.py` — record-side top-10 S1 per record for two char-3gram TF-IDF blockers (name+address, name),
  within the same country label; GPU dense fp16 ranking with memory-adaptive sharding, exact fp32 cosines.
- `xfeatures.py` — 37 token-level features: IDF-weighted soft token coverage (exact / fuzzy), frequent-token-free
  "core" names, digit→letter OCR folding, number relations with leading zeros stripped (exact / truncation /
  near-miss), address coverage, record-side context. IDF and the frequent-token set are learned per country
  from that split's S1 table (no word lists).
- `features.py` — 38 pair features (string similarities, digit agreement, token counts, name-twin count,
  record-side / S1-side context), computed one country at a time.
- `model.py` — XGBoost (`hist`), 5 folds grouped by S1 entity, 50 % negative sampling with weights, early
  stopping on held-out training S1s; streams the row-aligned feature files; models saved as UBJSON.
- `decide.py` — each record assigned to its best S1 if p ≥ threshold (also the expected-F0.5 hybrid rule,
  reported for comparison).
- `evaluate.py` — exact local macro F0.5 scorer.
- `pipeline.py` — orchestration, `--mode full|inference`, artifact export.
