# hybrid-calibration

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Data](https://img.shields.io/badge/data-CC--BY--4.0-lightgrey.svg)](LICENSE)

**Does adding LLM embeddings to a tree ensemble buy calibration, or does it cost accuracy?**

Mixing a 14-dimensional tabular view with 768-dimensional GPT-2 embeddings changes a
model's calibration profile in a way that plain accuracy metrics hide. This repository
measures it directly.

## What is here

```
Text -> [tabular features (14 dims)] + [GPT-2 embeddings (768 dims)]
     -> 3 feature sets: tabular | embeddings | hybrid
     -> models: RF / XGB / LGB / MLP, plus Platt and isotonic calibration
     -> metrics: ECE, MCE, Brier, accuracy, reliability
```

Planned design is 14 model variants x 10 seeds x 4 datasets (up to 560 training runs),
analysed with paired Wilcoxon tests under a Bonferroni correction.

## Status: quick test only

The committed results come from **2 seeds, 1 dataset, 4 of 14 models** — a pilot, not a
result. Read it as such:

| Model | ECE ↓ | MCE ↓ | Brier ↓ | Acc ↑ |
|---|:---:|:---:|:---:|:---:|
| RF-Tabular | 0.074 | 0.351 | 0.273 | **0.796** |
| RF-Hybrid | 0.087 | 0.334 | 0.280 | 0.790 |
| RF-Embed | **0.007** | **0.008** | 0.750 | 0.247 |
| MLP-Embed | 0.006 | 0.006 | 0.750 | 0.247 |

The pattern the pilot suggests: tabular features give high accuracy and middling
calibration; pure-embedding models look superbly calibrated while being close to
useless (accuracy 0.247); and the hybrid keeps tabular accuracy while calibrating
*worse* than tabular alone.

**Two seeds cannot separate a real effect from noise** — at n=2 the paired test is
close to uninformative. Do not cite these numbers; see [DEEP_DIVE.md](DEEP_DIVE.md)
for the blocking issues that have to be cleared before publication.

## Layout

| Path | Contents |
|---|---|
| `calibration.py` `models.py` | model construction and the calibration wrappers |
| `experiment.py` `data_loader.py` | run orchestration and dataset loading |
| `embedding_extractor.py` | GPT-2 embedding extraction |
| `data/` `results/` `figures/` | inputs, outputs, plots |
| [DEEP_DIVE.md](DEEP_DIVE.md) | architecture, results, and the publication blockers |

## Environment

- Python: CI runs on **3.12** (`.github/workflows/ci.yml`).
- Dependencies: there is no `requirements.txt`; the de-facto list comes from
  the imports — numpy, scipy, scikit-learn, matplotlib, torch + transformers
  (GPT-2 embedding extraction), xgboost, lightgbm, and the Hugging Face
  `datasets` library (imdb / ag_news / yelp loaders in `data_loader.py`).
  One-line install from [DEEP_DIVE.md](DEEP_DIVE.md) Quick Start:
  `pip install torch transformers xgboost lightgbm scikit-learn scipy matplotlib seaborn datasets`
- Seeds: `config.py` sets `RANDOM_SEEDS = list(range(10))` (the planned
  design); the committed pilot runs used `n_seeds: 2` (recorded in each
  results JSON's `config` block). The initial train/val/test split is
  stratified with `seed=42` (`data_loader.py`).

## Benchmark (offline verification)

`benchmarks/bench_offline_verify.py` verifies — without rerunning any model
training — that the committed pilot tables are real and recomputable:

```bash
python benchmarks/bench_offline_verify.py   # RESULT: PASS
```

It checks result-file reproducibility metadata, that ECE/MCE/Brier/accuracy
aggregates are finite probabilities in valid ranges, and that the README pilot
table matches its named JSON source byte-for-byte (round 3). It does **not**
claim statistical meaningfulness — 2 seeds cannot separate a real effect from
noise (see `DEEP_DIVE.md` and `PREREG.md` §5).

## Offline verification

```bash
python -m pip install pytest
pytest -q
# 3 passed
```

These lightweight tests do not rerun model training. They verify that every
committed result has its dataset, seed count, and model-variant metadata;
that reported calibration metrics stay in their valid ranges; and that the
pilot table above matches its named JSON source. The three tests are:
`test_result_files_have_reproducibility_metadata`,
`test_result_aggregates_are_finite_probabilities`, and
`test_readme_pilot_table_matches_named_result_file`.

## Quick start

```bash
# 1) Install deps (one line, from DEEP_DIVE.md Quick Start)
pip install torch transformers xgboost lightgbm scikit-learn scipy matplotlib seaborn datasets

# 2) Run the committed-results verification (no model training, no network)
pytest -q        # 3 passed

# 3) Regenerate the pilot table's numbers from the committed JSON (no retraining)
python -c "import json; d=json.load(open('results/newsgroups_20260626_232910.json')); [print(k, tuple(round(m[n+'_mean'],3) for n in ('ece','mce','brier','accuracy'))) for k,m in d['aggregated'].items()]"

# 4) Rerun the experiment itself (downloads data; may take long)
python experiment.py --dataset newsgroups
```

> The full planned design (14 variants × 10 seeds × 4 datasets, up to 560 training runs)
> has **not** been run. The committed numbers are a 2-seed / 1-dataset / 4-model pilot —
> see the boundary statement above and [DEEP_DIVE.md](DEEP_DIVE.md).

## Data availability

- **Raw text** is downloaded at run time: newsgroups via
  `sklearn.datasets.fetch_20newsgroups`, imdb / ag_news / yelp via the
  Hugging Face `datasets` library (`data_loader.py`). If a download fails the
  loader falls back to synthetic data (cached with a `data_source` flag) —
  check that flag before citing numbers.
- **Embedding caches** for the newsgroups splits are committed under
  `data/embeddings/` as `.npy` matrices. Filenames follow
  `{dataset}_{split}_{model}_{md5-of-texts}_{n_samples}.npy`
  (`embedding_extractor.py`), so a cache hit can be verified against the text.
- **Results ↔ files**: the table above corresponds exactly to
  `results/newsgroups_20260626_232910.json` (4 model variants, 2 seeds, 400
  samples; see its `aggregated` block). The other `results/newsgroups_*.json`
  files are later exploratory runs with more variants/samples (e.g. 7 variants
  at 800 samples in `newsgroups_20260627_174217.json`); `figures/*.png` are
  rendered from those JSONs by `visualize.py`.
- The full planned design (14 variants x 10 seeds x 4 datasets) has **not**
  been run yet; the blocking issues are listed in
  [DEEP_DIVE.md](DEEP_DIVE.md).

## License

Code is MIT-licensed ([LICENSE](LICENSE)). The calibration study releases the
experiment logs and result files under
[CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/).

GitHub's license detector reports this repository as `NOASSERTION` because it reads a
single SPDX id per repository and this one carries two. The split is deliberate: the
code stays permissively licensed so it can be reused, and the research material stays
attributable so a citation is required.
