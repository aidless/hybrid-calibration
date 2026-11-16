# hybrid-calibration

**STATUS: WORK-IN-PROGRESS — not A yet**

<!-- PROBLEM:START -->
Does adding a frozen text embedding to 14 surface-form features change probability calibration (ECE and Brier), or only accuracy?
The planned comparison is tabular versus embeddings versus hybrid, on RF, XGB, LGB, and MLP, with Platt and isotonic calibration, across 10 seeds and 4 datasets.
The committed files do not carry that comparison.
<!-- PROBLEM:END -->

The five historical `results/newsgroups_*.json` files are pilots on one dataset name, 400–800 rows, and 2–3 recorded seeds. They disagree with each other. Read the block below, which `make check` rebuilds from the raw files, before citing any number.

<!-- README-OBSERVED:START -->
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
<!-- README-OBSERVED:END -->

## Gaps that keep this off the A bar

1. The five historical JSON files store metrics and reliability bins, not `y_true` / `y_prob`. Their ECE and Brier cannot be recomputed from predictions. `make check` only recomputes ECE and Brier from probabilities for the new diagnostic under `results/leakage/`.
2. The shuffled-label and duplicate-injection controls were run on a fresh protocol (real 8-class 20 Newsgroups with TF-IDF, and the current 6-class synthetic generator). They were not run on the texts behind the historical JSON files, because those texts were not saved. The historical near-perfect embedding accuracies are therefore not identified.
3. Historical JSON does not record `data_source` or the embedding cache filename. Matching a `.npy` file to a JSON by test-set length is an inference, and the BGE n=400 cache matches more than one JSON.
4. The planned 14 variants × 10 seeds × 4 datasets design is **NOT YET MEASURED**. No Wilcoxon or Bonferroni result is claimed.
5. The live GitHub About field still has the previous sentence until a repo admin replaces it. The replacement text is in the observed block. A pull request cannot edit that field.

## Failure modes

See [FAILURE_MODES.md](FAILURE_MODES.md): leakage, ECE binning, small-n, and calibration fit on the test split. The counts are in the observed block.

## Offline check

Fresh clone, no model download:

```bash
python3 -m pip install -r requirements-check.txt
make check
```

`make check` runs `checks/check_offline.py` and `pytest`. It does not import torch, does not download GPT-2, BGE, or MiniLM, and does not retrain the diagnostic forests. It recomputes ECE, Brier, and accuracy from `results/leakage/*.json` probabilities, checks historical aggregates against per-seed metrics, recounts exact duplicate rows in `data/embeddings/`, and fails if the observed blocks in this file or [DEEP_DIVE.md](DEEP_DIVE.md) differ.

Regenerating the diagnostic (network only if the sklearn 20 Newsgroups cache is cold; not part of `make check`):

```bash
python3 scripts/run_leakage_diagnostic.py
python3 checks/check_offline.py --write
```

Change `PROTOCOL_ID` in `leakage.py` if the protocol constants change.

## What is in the tree

| Path | Role |
|---|---|
| `calibration.py` | ECE, MCE, Brier |
| `leakage.py` | Split, scaler-on-train, shuffle, duplicate injection, embedding-row overlap |
| `results/leakage/` | Diagnostic probabilities for the protocol above |
| `results/newsgroups_*.json` | Historical pilots. No prediction vectors |
| `data/embeddings/` | Committed BGE-512 and MiniLM-384 matrices. Not GPT-2 |
| `experiment.py` | Future runs record `data_source`, empirical class count, and per-seed text overlap, and refit the tabular scaler after each seed's split. Pairwise tests are marked non-interpretable when `n_seeds < 10` |
| [DEEP_DIVE.md](DEEP_DIVE.md) | Same observed block, plus within-file duplicate counts |
| [PREREG.md](PREREG.md) | Original plan. It is not a result |
| [FAILURE_MODES.md](FAILURE_MODES.md) | The four failure modes |

`figures/*.png` were redrawn from `results/newsgroups_20260626_232910.json` only. They are not a picture of the other four files. The heatmap title says a star would mean `significant_corrected` and that fewer than 10 seeds is not a significance result. Every committed pair has that flag false.

## Author-environment notes

These are notes about machines, not the offline contract:

- `embedding_extractor.py` looks for author-local weight directories (`models/bge`, `~/.cache/modelscope/...`). They are not in this repository. `make check` does not load them.
- `data_loader.py` sets `HF_HUB_DISABLE_SSL_VERIFY` and an unverified SSL context. That is an author-machine workaround. It is not part of the metric definition.
- An earlier note that `fetch_20newsgroups` failed on the authoring host is an author-host observation. The diagnostic in this commit called `sklearn.datasets.fetch_20newsgroups` for the loader's 8 categories. The raw text was not committed. Corpus counts in the observed block are records from that run; `make check` does not re-download the corpus.

## NOT YET MEASURED

- 14 model variants × 10 seeds × 4 datasets
- imdb, ag_news, and yelp_polarity (loaders exist, no committed result file)
- A shuffled-label control on the historical embedding matrices
- Which historical file, if any, used which committed `.npy` cache, beyond the test-size matches in the observed block

## License

Code is MIT ([LICENSE](LICENSE)). The experiment logs and result files in this repository are released under [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/) ([LICENSE-DATA](LICENSE-DATA)). Upstream 20 Newsgroups text is not included.

GitHub's license detector reports `NOASSERTION` because it reads one SPDX id and this repository carries two.

## 验收题库与复算

`eval/cases.jsonl` 是 20 条可执行验收题，覆盖 ECE、MCE、Brier、可靠性曲线、配置、Wilcoxon/Bonferroni 实现、表格特征、校准器契约和已提交结果。`make check` 不依赖这一题库（题库会导入 xgboost / lightgbm）。题库本身的判据不变：

```bash
python3 eval/run_cases.py
python3 eval/run_cases.py --selftest
```

`HC-17` 只数 `results/*.json`（不含 `results/leakage/`）。`HC-19` 只数 `tests/test_committed_results.py` 里的 `test_` 函数。
