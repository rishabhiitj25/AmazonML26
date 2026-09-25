# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** [Your Team Name]  
**Team Members:** [List all team members]  
**Submission Date:** [Date]

---

## 1. Executive Summary

We cast the task as **record-side assignment**. Each Source 2 / Source 3 record belongs to at most one
Source 1 entity, so the pipeline retrieves candidate S1 entities for every record, scores each (record, S1)
pair with a gradient-boosted classifier, and assigns each record to its best S1. It then chooses each S1's
final match set to maximise expected F0.5. Two components carry most of the result:
- a **cross-script token map learned from the training pairs**, which maps Indic-script names to their
  Latin form;
- **GPU-accelerated exact TF-IDF blocking** on name + address.

Out-of-fold macro F0.5 on the full training set is **0.9768**.

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
- **Hidden missing values and quoting.** The literal markers `<NULL>`, `<CITY_NAME>`, `null` and `N/A`
  appear inside fields, and 2.3–3.3% of S2/S3 addresses are empty. Files use standard CSV quoting.

### 2.2 Solution Strategy

**Approach Type:** Blocking + pairwise classifier + record-side assignment with an expected-F0.5 decision
layer (hybrid).

**Core Innovation:**
- A non-Latin→Latin token map learned only from aligned training pairs. On held-out entities, names in
  Indic scripts with token-set similarity ≥ 90 to their S1 name rose from 5.8% to 94.9%.
- Exact TF-IDF top-k retrieval on GPU, with Source 1 sharded densely across two GPUs.
- A decision layer that follows the macro-F0.5 metric: one-to-one record assignment, then an
  expected-F0.5 set size per S1.

---

## 3. Candidate Generation (Blocking)

### Normalisation

Applied to every source before blocking:
- NFKC normalisation.
- Placeholders removed.
- The learned token map applied to non-Latin tokens, with generic romanisation (anyascii) as the
  fallback.
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
- **Test:** 186,380,784 pairs (France 26.8M, India 90.3M, US 69.3M).
- Reduction ratio 0.9999916 against the full cross-product.

### How we ensured true matches were not lost

- Pair recall was measured on the full training set: **US 99.14%, India 98.59%** (98.92% overall). A
  perfect matcher on these candidates would reach macro F0.5 = 0.9966.
- The remaining misses were inspected. In the US, 95% are records with an empty address and a generic
  name that has many exact-name S1 twins. In India, they are generic names with thin partial addresses.
  Neither group can be resolved reliably even if retrieved.

---

## 4. Matching Model

**Features used (38 per pair):**
- **Name:** rapidfuzz ratio, token-set ratio, token-sort ratio, partial ratio, Jaro-Winkler; partial
  ratio on space-free names (catches glued domains and hashtags); exact-name equality; token counts;
  and S1 name-twin count (how many S1 in the same country share the normalised name).
- **Address:** ratio, token-set ratio and partial ratio on normalised addresses (NaN when either is
  empty); token-set ratio and ratio on the sequence of digit tokens; whether the first digit tokens are
  equal; token counts.
- **Retrieval:** cosine and rank under both blockers.
- **Context:** for three similarity scores (name+address cosine, name token-set, address token-set), the
  gap to the best *other* candidate and the rank. Both are computed on the record side and on the S1
  side, together with each side's number of candidates. This captures one-to-one competition between
  name twins.
- **Other:** whether the name was mapped from a non-Latin script, and the source (S2/S3).
- There is no country-identity feature.

**Model type:**
- XGBoost binary classifier (GPU `hist`, max depth 10, learning rate 0.1, 600 rounds, subsample and
  colsample 0.8).
- Trained with 2-fold cross-validation grouped by S1 entity, using all positives and 25% of negatives
  with inverse-rate weights.
- Test predictions average the two fold models.
- XGBoost is Apache-2.0. No pretrained neural model is used, so the total parameter count is far below
  8B.

**Threshold selection method:**
1. Each record is assigned to its highest-probability S1, if that probability is ≥ `p_min`.
2. For each S1, the assigned records are sorted by probability, and we keep the top-m that maximise
   expected F0.5: 1.25·Σp / (m + 0.25·N̂), where N̂ is the sum of probabilities over all of the S1's
   candidate pairs.
3. The S1 is left empty unless its best record reaches probability ≥ `gate`.

`p_min` and `gate` were grid-searched on out-of-fold predictions over the whole training set. The
chosen values are `p_min = 0.3` and `gate = 0.75`. A plain probability threshold of 0.8 scored 0.9765.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** **0.9768**, out-of-fold, on the full training set (2.21M S1 entities).
  - US 0.9775, India 0.9759.
  - Singletons 0.965, matched entities 0.9775.
  - In-sample (training) score 0.9800, so the generalisation gap is small.
  - Pair level: ROC-AUC 0.99989, PR-AUC 0.99762.
  - After the decision layer: match precision 99.07%, match recall 95.37%.
  - Public leaderboard: [to be filled].
- **Common false positives (wrong merges):**
  - Records whose name is a generic combination ("City Management Private", "Capital Services") that
    sit at the same or a similar address as another business.
  - Name twins with partial addresses, where the house number is missing or mangled.
- **Common false negatives (missed matches):**
  - Blocking misses (1.1% of true pairs, mostly empty-address records with generic names).
  - DBA or renamed records whose name is a random trade name, matched only by address.
  - Heavily truncated addresses, where the model abstains to protect precision.

---

## 6. Conclusion

Treating the problem as one-to-one record assignment, bridging scripts with a token map learned from the
training data, and running exact TF-IDF blocking on GPUs gives macro F0.5 0.9768 out-of-fold, against a
blocking ceiling of 0.9966. The precision-weighted metric is best served by an expected-F0.5 set size per
entity plus a confidence gate, rather than a single probability threshold. The main lessons were to
measure every data assumption before designing (singletons were rare, not common) and to validate on the
full candidate universe, because name twins make small subsets misleadingly easy.

---

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/`:

- `src/pipeline.py`: the entry point. Run it as
  `python -m src.pipeline --data-dir <dataset> --work <dir> --out <dir> --gpus cuda:0,cuda:1`.
- `src/preprocess.py`, `normalize.py`, `token_map.py`: normalisation and the learned script map.
- `src/blocking.py`: candidate generation (GPU exact TF-IDF top-k, with a CPU fallback).
- `src/features.py`: pair and context features.
- `src/model.py`: XGBoost OOF training and prediction.
- `src/decide.py`: assignment and the expected-F0.5 decision layer.
- `src/evaluate.py`: the local macro-F0.5 scorer.
- `README.md` gives exact run and validation commands; `requirements.txt` pins versions.

### B. Additional Results

Decision-rule comparison (OOF, full training set):

| decision | macro F0.5 | singleton | matched |
|---|---|---|---|
| threshold 0.5 | 0.9729 | 0.930 | 0.975 |
| threshold 0.8 | 0.9765 | 0.970 | 0.977 |
| expected-F0.5, empty if Π(1−p) is larger | 0.9755 | 0.920 | 0.979 |
| **hybrid (p_min 0.3, gate 0.75)** | **0.9768** | 0.965 | 0.9775 |

Learned token map (held-out entities):
- Covers 98.3% of non-Latin tokens.
- Name similarity ≥ 90 rises from 5.8% to 94.9%.
- Name+address recall@10 for non-Latin names rises from 0.904 to 0.981.
