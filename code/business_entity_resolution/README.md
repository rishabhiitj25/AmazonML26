# Business Entity Resolution — reproducible pipeline

Regenerates `output/matching_results.tsv` and `output/candidate_pairs.tsv` from the challenge data only.
No external data, APIs or pretrained models are used; the only learned model is an XGBoost classifier
(Apache-2.0) trained here on the provided training data.

## Environment

- Python 3.10, Linux. Tested on 2× NVIDIA L40S (46 GB), 512 CPU cores, 503 GB RAM.
- GPUs are optional: without `--gpus` blocking uses CPU sparse top-k and XGBoost runs on CPU (much slower).

```bash
python3.10 -m venv .venv && source .venv/bin/activate
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

## Run end-to-end

From this folder (`code/business_entity_resolution/`), with the challenge's `student_resource/dataset/`:

```bash
python -m src.pipeline \
    --data-dir /path/to/student_resource/dataset \
    --work /path/to/work \
    --out /path/to/output \
    --gpus cuda:0,cuda:1 --threads 64
```

Then validate:

```bash
cd /path/to/student_resource
python3 utils/validate_submission.py --matching /path/to/output/matching_results.tsv \
    --candidate /path/to/output/candidate_pairs.tsv --test-dir dataset/test --check-ids
```

Each stage caches its result under `--work` and is skipped when that file exists (delete it to recompute).

| Stage | Module | Output (`--work/…`) | Time on reference machine |
|---|---|---|---|
| Normalise + learned token map | `src/preprocess.py`, `src/normalize.py`, `src/token_map.py` | `cache/*.parquet`, `cache/token_map.json` | ~3 min |
| Blocking train / test | `src/blocking.py` | `train/cands.parquet`, `test/cands.parquet` | ~62 / 47 min (2 GPUs) |
| Pair features train / test | `src/features.py` | `train/feats.parquet`, `test/feats.parquet` | ~24 / 20 min (64 threads) |
| XGBoost 2-fold OOF | `src/model.py` | `models/xgb_fold{0,1}.json`, `train/oof.parquet` | ~15 min |
| Decision-rule tuning on OOF | `src/pipeline.py`, `src/decide.py` | `decision.json` | ~45 min |
| Test prediction + outputs | `src/model.py`, `src/pipeline.py` | `test/pred.parquet`, `--out/*.tsv` | ~10 min |

Peak RAM ≈ 105 GB (OOF stage), disk for `--work` ≈ 35 GB.

## Code map

- `io_utils.py` — TSV reading (standard CSV quoting, no NA conversion), ground-truth parsing, strict writer.
- `normalize.py` — placeholder removal (`<NULL>`, `<CITY_NAME>`, null/N/A components), NFKC, romanisation
  (anyascii) and accent folding, lowercase, punctuation → space, digit tokens.
- `token_map.py` — non-Latin → Latin token map learned from aligned training pairs.
- `blocking.py` — record-side top-10 S1 per record for two char-3gram TF-IDF blockers (name+address, name),
  restricted to the same country label; vectorisers fit per split and per country.
- `features.py` — 38 pair features: string similarities, digit agreement, token counts, name-twin count,
  record-side / S1-side context (gap to best other candidate, rank, candidate counts).
- `model.py` — XGBoost (GPU `hist`), 2 folds grouped by S1 entity, 25 % negative sampling with weights.
- `decide.py` — each record assigned to its best S1; per-S1 set size by expected F0.5 with confidence gates.
- `evaluate.py` — exact local macro F0.5 scorer.
- `pipeline.py` — orchestration.
