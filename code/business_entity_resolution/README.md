# Business Entity Resolution — reproducible pipeline

Produces `matching_results.tsv` and `candidate_pairs.tsv` from the challenge data only. No external data,
APIs or pretrained models; the only learned model is an XGBoost classifier (Apache-2.0) trained on the
provided training data. Trained artifacts are shipped in `artifacts/`, so the test predictions can be
reproduced without retraining.

```
business_entity_resolution/
├── src/                        # pipeline code (entry point: src/pipeline.py)
├── artifacts/                  # learned from train — enough for inference
│   ├── models/xgb_fold0.json   # XGBoost fold models (test prediction = their average)
│   ├── models/xgb_fold1.json
│   ├── token_map.json          # non-Latin → Latin token map learned from training pairs
│   └── decision.json           # tuned decision rule (hybrid, p_min=0.3, gate=0.75)
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

Runs everything on train (blocking, features, 2-fold XGBoost with out-of-fold predictions, decision-rule
tuning on OOF), then the test side, and exports the new artifacts to `--artifacts`.

Each stage caches its output under `--work` and is **skipped if that file already exists** — delete the
file (or use a new `--work`) to recompute a stage. Runs are resumable after interruption.

## 4. Hardware and runtime

| | Inference (`--mode inference`) | Full training (`--mode full`) |
|---|---|---|
| GPU | 1 CUDA GPU; ≥ 16 GB recommended (S1 is sharded to fit free memory) | 1–2 GPUs |
| RAM | ≥ 48 GB (measured peaks: blocking 25 GB, features 36 GB, prediction 13 GB) | ~128 GB (OOF stage loads 192M × 38 features) |
| Disk (`--work`) | ~20 GB | ~35 GB |
| Time on 2× L40S, 64 threads | 70 min measured (blocking 41 min, features 19 min, prediction 4 min) | ~4 h |

- `--gpus ''` runs without a GPU (CPU sparse top-k blocking and CPU XGBoost) — correct but slow
  (blocking alone takes several hours).
- `--gpus cuda:0,cuda:1` splits blocking across GPUs. Prediction always uses the first GPU listed.
- Lower `--threads` on smaller machines (string features and top-k use it).

## 5. Evaluating on the training data

Local scorer: `src/evaluate.py` (exact macro F0.5, per-S1, singletons included). `--mode full` prints the
decision-rule grid with out-of-fold macro F0.5 on the whole training set. Reference result:
**OOF macro F0.5 = 0.9768** (US 0.9775, India 0.9759).

## 6. Pipeline stages

| Stage | Module | Output (`--work/…`) |
|---|---|---|
| Normalise + token map | `preprocess.py`, `normalize.py`, `token_map.py` | `cache/*.parquet`, `cache/token_map.json` |
| Blocking | `blocking.py` | `{train,test}/cands.parquet` |
| Pair features | `features.py` | `{train,test}/feats.parquet` |
| XGBoost 2-fold OOF (full mode) | `model.py` | `models/`, `train/oof.parquet` |
| Decision tuning (full mode) | `pipeline.py`, `decide.py` | `decision.json` |
| Test prediction + outputs | `model.py`, `pipeline.py` | `test/pred.parquet`, `--out/*.tsv` |

## 7. Code map

- `io_utils.py` — TSV reading (standard CSV quoting, no NA conversion), ground-truth parsing, strict writer.
- `normalize.py` — placeholder removal (`<NULL>`, `<CITY_NAME>`, null/N/A components), NFKC, romanisation
  (anyascii) and accent folding, lowercase, punctuation → space, digit tokens.
- `token_map.py` — non-Latin → Latin token map learned from aligned training pairs.
- `blocking.py` — record-side top-10 S1 per record for two char-3gram TF-IDF blockers (name+address, name),
  within the same country label; GPU dense fp16 ranking with memory-adaptive sharding, exact fp32 cosines.
- `features.py` — 38 pair features (string similarities, digit agreement, token counts, name-twin count,
  record-side / S1-side context), computed one country at a time.
- `model.py` — XGBoost (`hist`), 2 folds grouped by S1 entity, 25 % negative sampling with weights.
- `decide.py` — each record assigned to its best S1; per-S1 set size by expected F0.5 with confidence gates.
- `evaluate.py` — exact local macro F0.5 scorer.
- `pipeline.py` — orchestration, `--mode full|inference`, artifact export.
