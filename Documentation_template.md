# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** [Your Team Name]  
**Team Members:** [List all team members]  
**Submission Date:** [Date]

---

## 1. Executive Summary

We cast the task as **record-side assignment**. Each Source 2 / Source 3 record belongs to at most one
Source 1 entity, so the pipeline retrieves candidate S1 entities for every record, scores each (record, S1)
pair with a gradient-boosted classifier, and assigns each record to its best S1 if the match probability
clears a threshold. Four components carry most of the result:
- a **cross-script token map learned from the training pairs**, which maps Indic-script names to their
  Latin form;
- **GPU-accelerated exact TF-IDF blocking** on name + address;
- **token-level features** (IDF-weighted token coverage, number relations with leading zeros stripped,
  near-miss house numbers), learned per country from S1 data, which recognise near-miss distractors;
- a **validation protocol that reproduces the leaderboard**: models and decision rules are chosen on a
  distractor-density simulation and a leave-one-country-out split, not on plain out-of-fold scores.

Macro F0.5 on the training data: **0.9872** out-of-fold, **0.9863** under test-like distractor density,
**0.9521** when predicting a country never seen in training.

---

## 2. Methodology

### 2.1 Problem Analysis

Findings from EDA on the training data (2.21M S1, 5.03M S2, 5.29M S3):

- **Singletons are rare, and matched entities have several matches.** Only 5.58% of S1 entities have no
  match; matched entities average 3.67 matches (up to 11). So recall *within* an entity drives the
  metric, not only the singleton decision.
- **Each record matches at most one S1.** No S2/S3 id appears in two S1 match lists (7.64M true pairs),
  and 26% of S2/S3 records match nothing. So the problem is "which S1, or none?" per record.
- **Country labels always agree** between matched records (100%), so blocking within an equal
  country label is lossless. Country is treated as an open label set, and test France is handled by the
  same code path.
- **Name twins are common.** 38% of S1 entities share their exact name with another S1 (groups of up to
  526), so the address decides most matches.
- **Names appear in many scripts.** In India, 23% of matched S2 and 13% of matched S3 records have
  names in Indic scripts: Devanagari, Tamil, Malayalam, Gujarati, Bengali, Telugu, Kannada. About 90% of
  them share no token with their S1 name.
- **Noise patterns.** Name noise: typos and OCR-style swaps (l→1), accents injected into English words,
  junk prefixes (`--`, `<<`, `***`, `#`), honorifics (Sri, Smt, Dr, The), bracketed words, legal-suffix
  moves and abbreviations (Pvt/Private, LLC/L.L.C.), domain names or hashtags used as the name, and
  DBA / "formerly" / "F/K/A" / "née" names. Address noise: street-type abbreviations, reordered
  components, state full names vs codes vs native script, truncated or mangled house numbers
  (4/792→4/79, C-208→C-206), different city names for the same place, and missing components.
- **The test pool is harder than train in a measurable way.** Test has ~1.9× more near-miss unmatched
  records per S1 (copies of a real business with one number or one distinctive word changed). This is why
  our first submission scored 0.969 against 0.977 out-of-fold, and why a stacked model that gained +0.0045
  out-of-fold *lost* 0.004 on the leaderboard: it relied on per-S1 counts that the extra distractors inflate.
- **France-only artefacts.** "N°" (degree sign) appears in ~70k test records, all French, and never in train
  or in any S1; generic romanisation spelled it out as "deg" ("N°12" → "ndeg12").
- **Hidden missing values and quoting.** The literal markers `<NULL>`, `<CITY_NAME>`, `null` and `N/A`
  appear inside fields, and 2.3–3.3% of S2/S3 addresses are empty. Files use standard CSV quoting.

### 2.2 Solution Strategy

**Approach Type:** Blocking + pairwise gradient-boosted classifier + one-to-one record assignment with a
probability threshold.

**Core Innovation:**
- A non-Latin→Latin token map learned only from aligned training pairs. On held-out entities, names in
  Indic scripts with token-set similarity ≥ 90 to their S1 name rose from 5.8% to 94.9%.
- Exact TF-IDF top-k retrieval on GPU, with Source 1 sharded densely across GPUs.
- Token-level near-miss features whose statistics (IDF, frequent-token set) are learned per country from
  that split's own S1 table, so an unseen country gets its own statistics without any word list.
- Model selection that anticipates the test distribution: every change is scored on plain out-of-fold
  data, on a simulation with test-like distractor density (which reproduced both of our leaderboard
  scores), on a second density simulation, and on leave-one-country-out. Only changes that help in all of
  them are kept.

---

## 3. Candidate Generation (Blocking)

### Normalisation

Applied to every source before blocking:
- NFKC normalisation.
- Placeholders removed.
- The learned token map applied to non-Latin tokens, with generic romanisation (anyascii) as the
  fallback.
- Non-ASCII punctuation and symbols (e.g. "°", "’") turned into spaces *before* romanisation, so they are
  not spelled out as words. This changes no training record (verified on all 2.1M non-ASCII rows) and fixes
  ~70k French test records.
- Accents folded, lowercase, `&`→"and", punctuation replaced by spaces.

### Blocking keys used

For every S2/S3 record we retrieve the **top-10 S1 entities with the same country label** under each of
two blockers:
1. **Name + address:** character-3-gram TF-IDF (`char_wb`, sublinear tf), cosine similarity.
2. **Name only:** the same representation, so records with an empty or thin address still get
   candidates.

Other settings:
- Vectorisers are fit per split and per country on that split's S1 text, so an unseen country (France)
  gets its own vocabulary and IDF. No labels are involved.
- Retrieval ranks candidates with fp16 dense matrix products on 2 GPUs. Compared with exact CPU sparse
  top-k, 99.8% of pairs are identical, and every difference is a near-tie at rank 10.
- Exact float32 cosines are then recomputed for all union pairs.
- The per-record candidate lists are inverted into per-S1 lists, which form `candidate_pairs.tsv`.
- Extreme cases are handled explicitly and covered by tests: a country whose S1 table shares no 3-gram
  between two names (e.g. a single S1) falls back to `min_df=1`; a failed GPU shard raises instead of
  silently dropping candidates; empty records simply get no candidates.

Blocker selection (200k training records):

| blocker (top-10) | US recall | India recall |
|---|---|---|
| name only | 0.795 | 0.756 |
| address only (char or word) | ~0.91 | ~0.86 |
| name + address | 0.987 | 0.972 |
| **name+address ∪ name (chosen)** | **0.992** | **0.986** |

Address-only blockers added ≤ 0.07 pp on top of name+address, so they were dropped. Dropping frequent
n-grams (`max_df`) was rejected because it cost 2 pp of India recall.

### Candidate pairs generated

- **Train:** 191,712,264 pairs (≈18.6 per record).
- **Test:** 186,380,066 pairs (France 26.8M, India 90.3M, US 69.3M).
- Reduction ratio 0.9999916 against the full cross-product.

### How we ensured true matches were not lost

- Pair recall was measured on the full training set: **US 99.14%, India 98.59%** (98.92% overall). A
  perfect matcher on these candidates would reach macro F0.5 = 0.9966.
- The remaining misses were inspected. In the US, 95% are records with an empty address and a generic
  name that has many exact-name S1 twins. In India, they are generic names with thin partial addresses.
  Neither group can be resolved reliably even if retrieved.

---

## 4. Matching Model

**Features used (75 per pair):**

*String and context features (38):*
- **Name:** rapidfuzz ratio, token-set ratio, token-sort ratio, partial ratio, Jaro-Winkler; partial
  ratio on space-free names (catches glued domains and hashtags); exact-name equality; token counts;
  and S1 name-twin count (how many S1 in the same country share the normalised name).
- **Address:** ratio, token-set ratio and partial ratio on normalised addresses (NaN when either is
  empty); token-set ratio and ratio on the sequence of digit tokens; whether the first digit tokens are
  equal; token counts.
- **Retrieval:** cosine and rank under both blockers.
- **Context:** for three similarity scores (name+address cosine, name token-set, address token-set), the
  gap to the best *other* candidate and the rank, on the record side and on the S1 side, with each side's
  number of candidates. This captures one-to-one competition between name twins.
- **Other:** whether the name was mapped from a non-Latin script, and the source (S2/S3).

*Token-level features (37, `xfeatures.py`):*
- **Name coverage:** IDF-weighted share of the S1's name tokens found in the record and vice versa, exact
  and fuzzy (normalised Levenshtein ≥ 0.75 for tokens of ≥ 4 characters); number and maximum IDF of
  uncovered tokens (a distinctive extra word signals a different business).
- **Core name:** similarity after removing the country's frequent tokens (in ≥ 2% of its S1 names, e.g.
  legal forms, plus 1-letter tokens); first core token match; OCR-style digit→letter folding (0→o, 1→l).
- **Numbers:** with leading zeros stripped, the share of the record's numbers that match exactly, are
  truncations/extensions of an S1 number, are same-length near misses, or are unrelated; house-number
  relation, absolute and relative difference.
- **Address coverage:** IDF-weighted token coverage of non-numeric address tokens, both directions.
- **Record-side context:** gap to the record's best other candidate and rank, for name coverage, exact
  number share, address coverage and house-number equality.
- IDF and the frequent-token set are computed per split and per country from S1 only, so France gets its
  own statistics. No hand-written word lists and no target encoding of tokens.
- There is no country-identity feature.

**Model type:**
- XGBoost binary classifier (GPU `hist`, max depth 10, learning rate 0.05, subsample and colsample 0.8,
  min child weight 5).
- 5-fold cross-validation grouped by S1 entity, all positives and 50% of negatives with inverse-rate
  weights. The number of trees is set per fold by early stopping (patience 50, at most 3000) on a
  hash-selected 5% of the *training* S1 entities, never on the evaluated fold: 1977–2299 trees.
- Test predictions average the five fold models.
- XGBoost is Apache-2.0. No pretrained neural model is used, so the total parameter count is far below
  8B.

**How the configuration was selected (no overfitting to one validation set):**

Plain out-of-fold scores were not trusted alone, because they had ranked our stacked model wrongly. Every
candidate was scored on four views of the training data:
- `base`: plain out-of-fold, folds grouped by S1;
- `clone09`: every unmatched record receives 0.9 exact clones on average (the test's ~1.9× distractor
  density), with the S1-side context recomputed. This simulation reproduced both leaderboard results
  (0.971 vs 0.969 for our first model; 0.966 vs 0.965 for the stacked one);
- `drop19`: 19% of S1 removed, so their records become distractors, record-side context recomputed;
- leave-one-country-out: train on US and predict India, and vice versa, as a proxy for France.

Noise floor: retraining with another seed moves scores by about ±0.0001; per-S1 paired differences have a
standard error of ~0.00006. Changes below 0.0002 were treated as noise.

- **K (number of folds) was chosen by a stopping rule:** increase K while the next step gains more than
  twice the noise floor. At threshold 0.85: K=2 0.98604 / 0.98499 / 0.98399 (base / clone09 / drop19);
  K=3 0.98622 / 0.98525 / 0.98425; K=5 0.98642 / 0.98549 / 0.98441; K=10 0.98644 / 0.98553 / 0.98448. K=5→10
  gains ≤ 0.00007 → **K = 5**.
- Kept: token features (+0.012…0.015 under density), early stopping with lr 0.05 (+0.0002…0.0003),
  50% negative sampling (+0.0003; 100% gains more but needs a 41.5 GB GPU allocation).
- Rejected: weighting unmatched negatives ×1.9 (no gain); removing S1-side context (mixed); a second-stage
  model on probability context and on agreement with the S1's most confident other record (better
  out-of-fold, worse under density — the same failure as our stacked submission); a two-level rule
  admitting extra records below the threshold at confident S1s (worse in every view).

**Threshold selection method:**
Each record is assigned to its highest-probability S1 if that probability is ≥ t. The objective was
J = 0.85 · F(clone09) + 0.15 · F(leave-one-country-out), mirroring the test mix (India and US are 85% of test
S1 entities, under test-like density; France is 15% and unseen):

| t | base | clone09 | drop19 | leave-one-country-out | J |
|---|---|---|---|---|---|
| 0.750 | 0.98722 | 0.98609 | 0.98525 | 0.95181 | 0.98095 |
| 0.775 | 0.98724 | 0.98620 | 0.98532 | 0.95195 | 0.98106 |
| **0.800** | 0.98721 | 0.98627 | 0.98537 | 0.95206 | 0.98113 |
| 0.825 | 0.98713 | 0.98628 | 0.98534 | 0.95202 | 0.98114 |
| 0.850 | 0.98699 | 0.98623 | 0.98522 | 0.95190 | 0.98108 |
| 0.900 | 0.98634 | 0.98576 | 0.98462 | 0.95086 | 0.98053 |

J is flat between 0.775 and 0.85; **t = 0.80** was chosen (tied with 0.825 within 1e-5, better on base
and drop19). The expected-F0.5 set-size rule of our first submission was dropped: it sizes each S1's set
from the summed probability of all its candidates, which extra distractors inflate.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro), training data, final configuration:**

  | view | first submission (v1) | final |
  |---|---|---|
  | out-of-fold (2.21M S1) | 0.9769 | **0.9872** |
  | test-like distractor density (clone09) | 0.9708 | **0.9863** |
  | 19% S1 removed (drop19) | 0.9736 | **0.9854** |
  | unseen country (leave-one-country-out) | 0.9251 | **0.9521** |

  - Final out-of-fold (threshold 0.80): US 0.9877, India 0.9865; singletons 0.9893, matched entities 0.9871;
    wrong matches per S1 0.0085 (0.0143 under test-like density).
  - Public leaderboard: first submission 0.969; stacked model 0.965 (rejected, see §4); final: [to be filled].
- **Where the remaining loss is** (out-of-fold, before the final tuning; shares of the total loss):
  - 62% partial recall: true records scored below the threshold at correctly matched S1s;
  - 18% matched S1 left empty;
  - 15% a wrong extra record at a matched S1;
  - 5% a singleton given a match;
  - 1.1% of true pairs are never retrieved by blocking.
- **Common false positives (wrong merges):**
  - Generic names ("City Management Private", "Capital Services") at the same or a similar address as
    another business.
  - Name twins with partial addresses, where the house number is missing.
- **Common false negatives (missed matches):**
  - Blocking misses, mostly empty-address records with generic names that have many exact-name twins.
  - DBA or renamed records whose name is an unrelated trade name, matched only by address.
  - Heavily truncated addresses, where the model abstains to protect precision.

---

## 6. Conclusion

Treating the problem as one-to-one record assignment, bridging scripts with a token map learned from the
training data, and running exact TF-IDF blocking on GPUs gave 0.977 out-of-fold. The larger lesson came
from the leaderboard: the test pool contains about twice as many near-miss distractors per entity, so a
model can win out-of-fold and still lose on test. Building a validation set that reproduces this (it
matched both of our leaderboard scores), then choosing features, K, regularisation and the threshold on it
together with a leave-one-country-out split, led to token-level near-miss features and a plain threshold.
That gave 0.987 out-of-fold and 0.986 under test-like density, with the density penalty cut from 0.006 to
0.001.

---

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/`:

- `src/pipeline.py`: the entry point. Run it as
  `python -m src.pipeline --mode full|inference --data-dir <dataset> --work <dir> --out <dir> --gpus cuda:0`.
  The selected configuration is pinned there (`MODEL`, `DECISION`).
- `src/preprocess.py`, `normalize.py`, `token_map.py`: normalisation and the learned script map.
- `src/blocking.py`: candidate generation (GPU exact TF-IDF top-k, with a CPU fallback).
- `src/features.py`: string and context features; `src/xfeatures.py`: token-level features.
- `src/model.py`: XGBoost K-fold OOF training with early stopping, and streaming prediction.
- `src/decide.py`: one-to-one assignment and decision rules.
- `src/evaluate.py`: the local macro-F0.5 scorer.
- `tests/`: 25 pytest cases, covering every stage plus an end-to-end run on synthetic data with extreme
  cases. The synthetic data has a test-only country, a country with one S1, S1 without records, empty
  records and non-Latin names; the result is checked by the official validator. The e2e case runs on CPU
  and on GPU.
- `artifacts/`: fold models, token map and decision rule, enough for inference without retraining.
- `README.md` gives exact run and validation commands; `requirements.txt` pins versions.

### B. Additional Results

Bugs found by the tests and audits of the first version, all fixed:
1. Blocking crashed for a country without two S1 names sharing a 3-gram (TF-IDF `min_df=2` → empty
   vocabulary).
2. An exception in a GPU blocking thread was swallowed, which would silently drop that shard's candidates.
   A shard with zero rows also crashed.
3. Non-ASCII symbols were romanised into words ("N°" → "ndeg", ~70k French test records).
4. Ops: XGBoost GPU prediction on a non-zero device ordinal crashes, so each process is pinned to one GPU.

Decision rules of the first model (v1 features, out-of-fold), showing why a mass-based rule was replaced:

| decision | base | clone09 (test-like density) |
|---|---|---|
| hybrid expected-F0.5 (p_min 0.3, gate 0.75) | **0.9769** | 0.9708 |
| threshold 0.85 | 0.9760 | **0.9726** |

Learned token map (held-out entities):
- Covers 98.3% of non-Latin tokens.
- Name similarity ≥ 90 rises from 5.8% to 94.9%.
- Name+address recall@10 for non-Latin names rises from 0.904 to 0.981.
