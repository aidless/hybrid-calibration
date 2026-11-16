# Failure modes

Numbers cited from runs live in the observed block of [README.md](README.md) and [DEEP_DIVE.md](DEEP_DIVE.md). `make check` rebuilds that block. This page names the mechanisms.

## Leakage

Label text, duplicate rows, a scaler fit on the wrong split, and a constant embedding each produce a different signature.

- **Label inside the text.** The current synthetic generator's class is the style template that wrote the document, so a hashing model can be perfect without any split bug. The `shuffled_hash` arm permutes labels before the split; accuracy falls to the chance band. On the real 8-category 20 Newsgroups subset, a fraction of documents contain class cue words. Deleting a fixed cue list (including ordinary words such as `car`, `gun`, and `windows`) lowers TF-IDF accuracy and does not remove it. That ablation is aggressive and label-independent; it is not a pure causal estimate.
- **Duplicate rows across splits.** `train_test_split` does not drop repeated texts. The diagnostic records, per seed, how many test strings occur verbatim in train, and it scores the complement. The committed embedding matrices contain exact float32 duplicates between the inferred train file and the inferred test file. A ceiling that gives every overlapping test row a free correct prediction, and every other row only 4-class chance, is still below the historical near-perfect RF-Embed accuracy on the one JSON with the same test length. Duplicates are real. They do not, by themselves, explain that accuracy. The cache-to-JSON link is a test-size match, not a stored filename.
- **Scaler before the resplit.** The historical path scaled tabular features on the seed-42 training split and then moved those scaled rows into other seeds' test sets. Tree splits are invariant to per-feature affine rescaling, so this does not change RF, XGB, or LGB predictions. It can change the MLP. `experiment.py` now refits the scaler inside each seed. Historical MLP numbers were not rerun.
- **Embedding fit on the full labeled set.** The extractor does not do this. It applies a frozen encoder. Substituting zeros (`--skip-embeddings`, and the `zero_embed` arm) yields Brier `1 - 1/K` and accuracy near `1/K`, with ECE near 0, because the forest emits the class prior. Several historical embedding rows have that shape at K=4 (Brier near 0.750, one occupied confidence bin). The files do not record whether zeros were used.
- **Cache key.** A hash of only the first and last 100 texts can reuse another split's matrix. The key now covers every text. That does not rewrite the committed matrices.

The historical near-perfect rows (accuracy near 1 with non-trivial Brier, probability width 4) are not reproduced by TF-IDF on the loader's 8-class newsgroups slice, and they are not reproduced by the current 6-class synthetic generator's class count. They stay unresolved.

## ECE binning

`compute_ece` is a macro average of one-vs-rest equal-width bins. `compute_mce` and `reliability_curve` use the top-label confidence. Those are different functionals. The diagnostic recomputes TF-IDF ECE at 5, 10, 15, and 20 bins from the same stored probabilities; the means move. Historical reliability arrays contain bare `NaN` for empty bins, which is not strict JSON. Python's `json` loader accepts them; `make check` uses that loader.

## Small-n

Historical test sets have 81, 101, or 161 rows. With 10 bins, the bin that realizes MCE often has a handful of rows (minimum count 1 in the committed curves). MCE at that n is not a stable calibration summary. Recorded seed counts are 2 or 3. One file's two seed blocks match to floating-point noise, so its independent-seed count is 1. Sample standard deviations are reported in the observed table. No significance claim is made from them. `experiment.py` still computes Wilcoxon and Bonferroni when asked, and it sets `pairwise_tests_interpretable` false when `n_seeds < 10`.

## Calibration on the test split

`CalibratedModel.fit` raises if validation labels are missing, and the historical isotonic variants were fit on the validation split. On the small historical test sets, isotonic still moves accuracy a lot relative to the uncalibrated model; that is small-validation instability, and it is visible in `newsgroups_20260627_181524.json` (see the observed block's source file for the uncalibrated vs isotonic aggregates — the README table shows the embedding rows; the isotonic rows are in that same JSON's `aggregated` object).

The diagnostic adds an explicit contamination probe on one seed: the same per-class isotonic regression is fit on the test labels and then scored on those labels. ECE and Brier on that row are not valid. The validation-fit row, scored on test, is the one that matches the library's rule. One seed is not a variance estimate.
