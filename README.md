# hybrid-calibration

> **English TL;DR**: A calibration study that mixes 14 tabular features with 768-d GPT-2 embeddings to compare ECE, Brier score and accuracy across tabular / embedding / hybrid models. Every committed result is a pilot (400–800 samples, 2–3 seeds, 1 dataset, 4–7 of 14 planned variants); the planned 14 × 10 × 4 design is **NOT YET MEASURED**, and the embedding variants flip between useless and near-perfect across runs, which is itself the honest finding.

[![CI](https://github.com/aidless/hybrid-calibration/actions/workflows/ci.yml/badge.svg)](https://github.com/aidless/hybrid-calibration/actions/workflows/ci.yml)

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Data](https://img.shields.io/badge/data-CC--BY--4.0-lightgrey.svg)](LICENSE-DATA)

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

### All committed result files (ECE ↓ / Brier ↓ / Acc ↑)

Five committed `results/newsgroups_*.json` files exist. Beyond the pilot table
above (20260626_232910.json), the other four are exploratory runs; numbers below
are read directly from each file's `aggregated` block:

| File | n / seeds / variants | RF-Tabular | RF-Embed | RF-Hybrid | MLP-Embed |
|---|---|---|---|---|---|
| `newsgroups_20260626_232811.json` | 400 / 2 / 4 | .076 / .321 / .765 | .007 / .750 / .247 | .081 / .344 / .753 | .005 / .750 / .247 |
| `newsgroups_20260627_174217.json` | 800 / 3 / 7 | .058 / .540 / .588 | .094 / .073 / .996 | .096 / .074 / .998 | .003 / .006 / .998 |
| `newsgroups_20260627_175253.json` | 500 / 2 / 4 | .077 / .614 / .525 | .184 / .265 / .970 | .190 / .272 / .985 | .010 / .027 / .985 |
| `newsgroups_20260627_181524.json` | 400 / 2 / 6* | .097 / .636 / .457 | .178 / .255 / .963 | .184 / .268 / .951 | .019 / .056 / .963 |

\* the 181524 file also contains calibration variants:
`RF-Hybrid-Isotonic` .113 / .309 / .784 and `MLP-Embed-Isotonic` .061 / .200 / .877.
The 174217 file additionally contains `XGB-Tabular` .126 / .616 / .586,
`XGB-Embed` .014 / .015 / .994 and `XGB-Hybrid` .014 / .014 / .994.

**Cross-run inconsistency is itself the finding to be honest about:** the
embedding-based variants flip between useless (Acc .247, Brier .750) and
near-perfect (Acc .998, Brier .006) across these files. The `data_source`
(real vs synthetic fallback) flag lives in the loader's pkl cache
(`data_loader.py:60-93`) and is **not** recorded in the results JSONs, so the
real-vs-synthetic provenance of each file cannot be resolved after the fact.
This unresolved provenance is a publication blocker listed in DEEP_DIVE.md.

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
- Dependencies: `requirements.txt` (generated from static import analysis —
  see the header comment in the file) covers numpy, scipy, scikit-learn,
  matplotlib, seaborn, torch, xgboost, lightgbm and pytest; the optional
  Hugging Face `datasets` / `requests` / `transformers` imports
  (imdb / ag_news / yelp loaders and GPT-2 embedding extraction) are
  listed as guarded comments in the same file. Install:
  `pip install -r requirements.txt`.
- Seeds: `config.py` sets `RANDOM_SEEDS = list(range(10))` (the planned
  design); the committed pilot runs used `n_seeds: 2` (recorded in each
  results JSON's `config` block). The initial train/val/test split is
  stratified with `seed=42` (`data_loader.py`).

## Offline verification

```bash
python -m pip install pytest
pytest -q
```

These lightweight tests do not rerun model training. They verify that every
committed result has its dataset, seed count, and model-variant metadata;
that reported calibration metrics stay in their valid ranges; and that the
pilot table above matches its named JSON source.

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

## Limits

- **The full 14 x 10 x 4 design was not run.** A 2026-11-10 attempt to run a
  representative subset on the authoring host was blocked twice over:
  (1) `fetch_20newsgroups` download fails there (HTTPError), so the loader
  falls back to synthetic data (`data_loader.py:93`); (2) the BGE / MiniLM
  embedding models are not present in the local HF/ModelScope caches and
  huggingface.co was unreachable, so real-embedding extraction cannot start.
  A tabular-only config (RF-Tabular, 10 seeds, 400 samples) does complete in
  ~14 s wall-clock on that host, but only against the synthetic fallback, so
  its output was deleted rather than committed (`data_source` flag would read
  synthetic). No new numbers were added from that attempt.
- n=2–3 seeds per committed file; paired Wilcoxon tests are uninformative at
  those sample sizes.
- Real-vs-synthetic provenance of the committed result files is unresolvable
  post-hoc (see the cross-run inconsistency note above).
- imdb / ag_news / yelp have zero committed results; the loaders exist but
  were never successfully run end-to-end against the real datasets.

## License

Code is MIT-licensed ([LICENSE](LICENSE)). The calibration study releases the
experiment logs and result files under
[CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/).

GitHub's license detector reports this repository as `NOASSERTION` because it reads a
single SPDX id per repository and this one carries two. The split is deliberate: the
code stays permissively licensed so it can be reused, and the research material stays
attributable so a citation is required.

## U1 results table (recomputed 2026-11-11)

All numbers below are recomputed offline from the five committed result JSONs.
No training was run in this pass.

| Result file | n_samples | n_seeds | model variants | seed_results | Source (path:line) |
|---|---|---|---|---|---|
| `newsgroups_20260626_232811.json` | 400 | 2 | 4 | 2 | `results/newsgroups_20260626_232811.json:4` |
| `newsgroups_20260626_232910.json` | 400 | 2 | 4 | 2 | `results/newsgroups_20260626_232910.json:4` |
| `newsgroups_20260627_174217.json` | 800 | 3 | 7 | 3 | `results/newsgroups_20260627_174217.json:4` |
| `newsgroups_20260627_175253.json` | 500 | 2 | 4 | 2 | `results/newsgroups_20260627_175253.json:4` |
| `newsgroups_20260627_181524.json` | 400 | 2 | 6 | 2 | `results/newsgroups_20260627_181524.json:4` |

Primary pilot file (`newsgroups_20260626_232910.json`), `aggregated` means:

| Model | ECE ↓ | Brier ↓ | Accuracy ↑ | Source (path:line) |
|---|---|---|---|---|
| `RF-Tabular` | 0.074 | 0.273 | 0.796 | `results/newsgroups_20260626_232910.json:9` |
| `RF-Embed` | 0.007 | 0.750 | 0.247 | `results/newsgroups_20260626_232910.json:10` |
| `RF-Hybrid` | 0.087 | 0.280 | 0.790 | `results/newsgroups_20260626_232910.json:11` |
| `MLP-Embed` | 0.006 | 0.750 | 0.247 | `results/newsgroups_20260626_232910.json:12` |

**Recompute command actually executed for this pass** (exit 0):

```bash
# 5 result files
# newsgroups_20260626_232811.json 400 2 4
# newsgroups_20260626_232910.json 400 2 4
# newsgroups_20260627_174217.json 800 3 7
# newsgroups_20260627_175253.json 500 2 4
# newsgroups_20260627_181524.json 400 2 6
```

**NOT YET MEASURED**: the full 14 variants x 10 seeds x 4 datasets design
(`README.md:23`, `README.md:127`), and any paired significance claim at n=2–3
seeds (`README.md:43`). The cross-run inconsistency is documented in the
existing README section above: `RF-Embed` accuracy moves from
0.247 to
0.996
across committed files. Offline reproduction (no network, no model downloads):

```bash
python3 -c "import json,glob; [print(p, json.load(open(p))['aggregated']) for p in sorted(glob.glob('results/newsgroups_*.json'))]"
```

<!-- figure: pending real measurement -->
