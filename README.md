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
