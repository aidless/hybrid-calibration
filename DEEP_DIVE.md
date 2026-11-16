# Hybrid calibration — what was actually measured

**STATUS: WORK-IN-PROGRESS — not A yet**

The question and the gap list live in [README.md](README.md). This file is the long table. Do not cite a number from memory; the block between the markers is regenerated from the raw artifacts.

<!-- DEEP-OBSERVED:START -->
The block below is generated from the raw JSON and `.npy` files by `checks/render_observed.py`. `make check` fails if this block is edited by hand.

### Historical JSON files (stored metrics, not recomputed from probabilities)

These five files have no `y_true` or `y_prob`. `make check` averages the stored per-seed ECE, MCE, Brier, and accuracy and requires that average to match `aggregated`. It cannot rebuild ECE from predictions that were never saved.

| File | n | seeds recorded | independent seeds | variants | proba width | config classes | RF-Tabular acc | RF-Embed acc | RF-Embed ECE | RF-Embed Brier |
|---|---:|---:|---:|---:|---:|---:|---|---|---|---|
| `newsgroups_20260626_232811.json` | 400 | 2 | 1 | 4 | 4 | 8 | 0.765±0.000 | 0.247±0.000 | 0.007±0.000 | 0.750±0.000 |
| `newsgroups_20260626_232910.json` | 400 | 2 | 2 | 4 | 4 | 8 | 0.796±0.044 | 0.247±0.000 | 0.007±0.001 | 0.750±0.000 |
| `newsgroups_20260627_174217.json` | 800 | 3 | 3 | 7 | 4 | 8 | 0.588±0.026 | 0.996±0.007 | 0.094±0.005 | 0.073±0.010 |
| `newsgroups_20260627_175253.json` | 500 | 2 | 2 | 4 | 4 | 8 | 0.525±0.014 | 0.970±0.000 | 0.184±0.009 | 0.265±0.017 |
| `newsgroups_20260627_181524.json` | 400 | 2 | 2 | 6 | 4 | 8 | 0.457±0.070 | 0.963±0.035 | 0.178±0.006 | 0.255±0.016 |

Per-seed accuracy, RF-Tabular then RF-Embed:

| File | RF-Tabular seeds | RF-Embed seeds |
|---|---|---|
| `newsgroups_20260626_232811.json` | 0.765, 0.765 | 0.247, 0.247 |
| `newsgroups_20260626_232910.json` | 0.827, 0.765 | 0.247, 0.247 |
| `newsgroups_20260627_174217.json` | 0.596, 0.559, 0.609 | 1.000, 1.000, 0.988 |
| `newsgroups_20260627_175253.json` | 0.515, 0.535 | 0.970, 0.970 |
| `newsgroups_20260627_181524.json` | 0.407, 0.506 | 0.938, 0.988 |

Every stored probability vector has width 4. `config.num_classes` is 8 in every file. Those two facts disagree. Pairwise tests stored in the files: 0 of 54 have `significant_corrected` true. That count is not a significance result; recorded seed counts are 2 or 3. Files whose seed blocks match to 1e-9 on ECE, accuracy, and Brier, so the independent-seed count is 1: `newsgroups_20260626_232811.json`.

Single-bin reliability with Brier above 0.5 occurs for 8 embedding model-rows, in `newsgroups_20260626_232811.json`, `newsgroups_20260626_232910.json`. A forest on constant features produces Brier 1-1/K. The diagnostic `zero_embed` arm lands on that identity for K=8 and K=6 (table below). Those historical rows sit near Brier 0.750, which is 1-1/4, matching the stored probability width. The files do not say whether those features were zeros. Single-bin reliability with Brier at or below 0.5 is a different pattern (a peaked, mostly correct predictor), and it occurs in: `newsgroups_20260627_174217.json seed 123 MLP-Embed`.

Stored MCE equals the largest |bin accuracy − bin confidence| gap on the saved reliability curve for every seed-model row (mismatches: 0). The bin that realizes MCE has minimum count 1 and median count 3. At test n of 81–161 and 10 bins, MCE is a small-bin statistic.

### Committed embedding matrices (exact float32 row match)

Groups are inferred: one model family whose train+val+test row counts sum to the stated n. Result JSON does not store the cache filename.

| Group | dim | test n | test rows also in train | test rows also in val |
|---|---:|---:|---:|---:|
| `bge_n400` | 512 | 81 | 7 | 2 |
| `bge_n500` | 512 | 101 | 11 | 2 |
| `bge_n996` | 512 | 200 | 8 | 1 |
| `minilm_n800` | 384 | 161 | 27 | 7 |

`minilm_n800` test n=161 is the only historical file with that test size: `newsgroups_20260627_174217.json`, RF-Embed accuracy mean 0.996. If the 27 overlapping test rows were perfect and the other rows were at 4-class chance 0.25, accuracy would be 0.376. 0.996 is above that ceiling. Exact duplicate vectors in that cache are a real split overlap and are not sufficient to produce the stored accuracy. The link from this cache to that JSON is the shared test size, not an id stored in the JSON.

Test n=81 appears in more than one JSON (newsgroups_20260626_232811.json, newsgroups_20260626_232910.json, newsgroups_20260627_181524.json), so the BGE n=400 cache is not assigned to a single file. Across sample sizes, `newsgroups_test_BAAI_bge_small_zh_v1.5_39bec81c_81.npy` has 43/81 rows exactly equal to a row in `newsgroups_train_BAAI_bge_small_zh_v1.5_8324f553_349.npy`. That pair is not one split.

### Diagnostic `leakdiag-2026-11-16` (ECE and Brier recomputed from stored probabilities)

Random forest, 100 trees, depth 10, `random_state=42`, `n_jobs=1`. Seeds 0–4 are split seeds only. No Wilcoxon test is computed. Not GPT-2, not the committed BGE/MiniLM matrices, not a rerun of the five JSON files.

Real 20 Newsgroups, the loader's 8 categories, headers/footers/quotes removed, stratified subset n=800. TF-IDF (`max_features=2000`, word n-grams 1–2, `min_df=2`) is fit on the training split only.

| Arm | ECE | Brier | Accuracy |
|---|---|---|---|
| `tabular` | 0.040±0.005 | 0.852±0.011 | 0.217±0.028 |
| `tfidf` | 0.070±0.008 | 0.720±0.019 | 0.606±0.054 |
| `tfidf_cue_ablated` | 0.055±0.005 | 0.769±0.010 | 0.511±0.064 |
| `shuffled_tfidf` | 0.010±0.003 | 0.879±0.004 | 0.102±0.023 |
| `zero_embed` | 0.002±0.000 | 0.875±0.000 | 0.124±0.000 |
| `dup_inject_tfidf` | 0.084±0.006 | 0.687±0.013 | 0.681±0.032 |

Chance accuracy at 8 classes is 0.125. `shuffled_tfidf` is 0.102±0.023. `zero_embed` Brier is 0.875±0.000 (1-1/8 = 0.875) and accuracy is 0.124±0.000. TF-IDF accuracy on rows whose exact text is absent from train is 0.622±0.055. Exact train/test text overlap by seed: 4, 4, 5, 6, 3. `dup_inject_tfidf` replaces 40 test rows with copies of training rows (fraction 0.25) and accuracy moves from 0.606±0.054 to 0.681±0.032.

Cue-word ablation (`tfidf_cue_ablated`) uses a label-independent word-boundary deletion of 19 tokens, including ordinary words (`car`, `gun`, `windows`). Recorded on the subset at generation time, and not recomputed by `make check` because the raw text is not committed: own-class cue rate 0.340, any-cue rate 0.362, full corpus n=7769 unique=7530 empty=236, subset n=800 unique=781 empty=23.

TF-IDF ECE at bin counts 5:0.066±0.007, 10:0.070±0.008, 15:0.095±0.010, 20:0.095±0.015. The same probabilities, three different bin widths, three different ECE means.

Isotonic probe, seed 0 only, same TF-IDF forest. Uncalibrated on that seed: ECE 0.071, Brier 0.721, accuracy 0.602. Isotonic fit on validation, scored on test: ECE 0.049, Brier 0.551, accuracy 0.584. Isotonic fit on the test labels, scored on those same labels: ECE 0.037, Brier 0.445, accuracy 0.677. The last row is contamination, not a result.

Synthetic style corpus from the current `_load_synthetic(600)` (6 classes). The class is the template family that wrote the text. This generator's class count is 6, so it is not what the 4-column historical files scored.

| Arm | ECE | Brier | Accuracy |
|---|---|---|---|
| `tabular` | 0.037±0.006 | 0.089±0.028 | 0.947±0.031 |
| `hash` | 0.034±0.004 | 0.023±0.005 | 1.000±0.000 |
| `hash_label_in_text` | 0.029±0.004 | 0.018±0.004 | 1.000±0.000 |
| `shuffled_hash` | 0.067±0.015 | 0.879±0.029 | 0.182±0.033 |
| `zero_embed` | 0.003±0.001 | 0.833±0.000 | 0.167±0.004 |

What this supports: the five JSON files disagree with each other and with `config.num_classes`; their stored pairwise flags are all false; committed embedding splits contain exact duplicate vectors; a shuffled-label control on the new 8-class TF-IDF run collapses; constant features reproduce the uniform Brier identity; injecting train rows into the test set raises accuracy; fitting isotonic on the test labels lowers that seed's ECE.

What this does not support: a stable calibration effect of embeddings; GPT-2 at 768 dimensions; that the historical near-perfect embedding accuracies are accounted for by the duplicate-vector counts above; any Wilcoxon or Bonferroni claim; the planned 14 variants × 10 seeds × 4 datasets, which is NOT YET MEASURED.

GitHub About replacement (the live About field is not changed by a pull request):

> WIP pilot, not a finding. Five committed runs are 4-class files labeled as 8-class newsgroups; on-disk embeddings are BGE-512 and MiniLM-384, not GPT-2-768. Scores flip from chance to near-perfect; train/test duplicate vectors are observed; 14x10x4 is not measured.

### Per-group duplicate counts inside each matrix

| Group | split | rows | unique | extra copies | max multiplicity |
|---|---|---:|---:|---:|---:|
| `bge_n400` | train | 279 | 257 | 22 | 4 |
| `bge_n400` | val | 40 | 39 | 1 | 2 |
| `bge_n400` | test | 81 | 78 | 3 | 2 |
| `bge_n500` | train | 349 | 315 | 34 | 4 |
| `bge_n500` | val | 50 | 49 | 1 | 2 |
| `bge_n500` | test | 101 | 101 | 0 | 1 |
| `bge_n996` | train | 697 | 688 | 9 | 3 |
| `bge_n996` | val | 99 | 99 | 0 | 1 |
| `bge_n996` | test | 200 | 200 | 0 | 1 |
| `minilm_n800` | train | 559 | 508 | 51 | 4 |
| `minilm_n800` | val | 80 | 78 | 2 | 2 |
| `minilm_n800` | test | 161 | 154 | 7 | 3 |

Diagnostic environment recorded in the JSON: sklearn 1.9.1, numpy 2.5.5. `make check` does not retrain the forests and does not download 20 Newsgroups; it recomputes ECE, Brier, and accuracy from the stored probability matrices.
<!-- DEEP-OBSERVED:END -->

## Code paths that matter for leakage

- `experiment.py` pools the seed-42 split and splits again per seed. As of this commit, tabular features are re-extracted after that split and `StandardScaler` is fit on that seed's training rows only. Historical JSON files were written before that change. They are not recomputed here.
- Tree `random_state` stays 42. The seed axis is the split, not the forest RNG. That was true of the historical runs' code path and is still true.
- `embedding_extractor.py` runs a frozen encoder under `torch.no_grad`. It does not fit the encoder on labels. The cache key now hashes every text. The committed `.npy` files were written with the previous prefix/suffix hash; a collision is not demonstrated, and it is not ruled out.
- `--skip-embeddings` fills the embedding block with zeros. The diagnostic's `zero_embed` arm is the controlled version of that path.
- `CalibratedModel` in `models.py` refuses to fit without a validation split. The diagnostic's `fit_on_test` isotonic row is a separate probe that deliberately breaks that rule, and it is labeled as contamination.

## What is not in this file

Publication prose, cross-paper citation drafts, and a claim that hybrid features have a stable calibration cost. Those were removed because the five JSON files do not support them jointly. The planned design remains **NOT YET MEASURED**.
