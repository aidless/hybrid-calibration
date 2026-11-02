# Benchmark: offline verification of the committed pilot tables

`bench_offline_verify.py` verifies — **without rerunning any model training** —
that the committed pilot numbers are real and recomputable:

| Check | What it proves |
|---|---|
| result files present | `results/*.json` exist (5 newsgroups pilot files) |
| reproducibility metadata | each file carries dataset, n_seeds (>0), model_variants |
| finite probabilities | ECE/MCE/Brier/accuracy means & stds in valid ranges |
| pilot table byte-match | README pilot table equals `newsgroups_20260626_232910.json` `aggregated` (round 3) |

```bash
python benchmarks/bench_offline_verify.py   # RESULT: PASS
```

This is the honest claim the pilot can make: the reported ECE/MCE/Brier/accuracy
numbers are recomputable from the committed JSON with no model call. It does **not**
claim the pilot is statistically meaningful — 2 seeds cannot separate a real effect
from noise (see `DEEP_DIVE.md` blocking issues and `PREREG.md` §5 boundaries).
