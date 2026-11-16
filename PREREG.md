# PREREG — hybrid-calibration

This file is the original plan. It is not a report of what the committed files show.

**STATUS: WORK-IN-PROGRESS — not A yet.** The plan below was 14 variants × 10 seeds × 4 datasets, with paired Wilcoxon tests and a Bonferroni correction. That design is **NOT YET MEASURED**. The committed historical files are pilots (one dataset name, 2–3 recorded seeds, 4–7 variants) whose probability vectors have width 4 while `config.num_classes` says 8. They do not jointly support the hypotheses below. See [README.md](README.md).

## Planned question

Does adding a frozen text embedding to 14 surface features change ECE and Brier, or only accuracy?

## Planned hypotheses (not established)

- H1. Tabular features: higher accuracy, middling ECE.
- H2. Embedding-only models: very small ECE and near-chance accuracy.
- H3. Hybrid models: tabular accuracy, worse ECE than tabular alone.

H2 describes the June 26 embedding rows (single confidence bin, Brier near `1 - 1/4`). It does not describe the later files, where embedding accuracy is near 1. H1 and H3 are not stable across the five files. None of the three is a finding.

## What the plan said the features were

The plan text used to say "768-d GPT-2". The code and the committed matrices do not. The default encoder name in `config.py` is `BAAI/bge-small-zh-v1.5` (512-d). The other committed matrices are MiniLM (384-d). There is no GPT-2 weight and no 768-d matrix in the tree.

The plan text used to say `--skip-embeddings` was on and the embeddings were synthetic. That is not recorded in the JSON. Zeros are what `--skip-embeddings` does. Some historical embedding rows look like a constant predictor. Others do not. The flag was not saved.

## What a confirmatory run would have to record

`data_source`, empirical class count, encoder name, cache filename, the split seed, and `y_true` / `y_prob` on the test split. Pairwise tests are not interpretable below 10 seeds. A shuffled-label control has to be reported next to any near-perfect embedding accuracy, on the same rows.

## Reproduction of the numbers that do exist

```bash
python3 -m pip install -r requirements-check.txt
make check
```

That does not rerun the planned design.
