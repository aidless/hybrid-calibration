# Narrow JSON check

`bench_offline_verify.py` checks the five historical `results/newsgroups_*.json`
files for metadata and finite aggregates, and checks one hardcoded round-3
tuple against `newsgroups_20260626_232910.json`. It does not read the README
and it does not recompute ECE from probabilities.

The offline contract is `make check` (see the repository README). That command
recomputes ECE and Brier from `results/leakage/` and fails if the observed
blocks drift.

```bash
python benchmarks/bench_offline_verify.py
```
