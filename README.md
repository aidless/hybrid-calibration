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

## License

Code is MIT-licensed ([LICENSE](LICENSE)). The calibration study releases the
experiment logs and result files under
[CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/).

GitHub's license detector reports this repository as `NOASSERTION` because it reads a
single SPDX id per repository and this one carries two. The split is deliberate: the
code stays permissively licensed so it can be reused, and the research material stays
attributable so a citation is required.
