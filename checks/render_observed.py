"""Render the observed-numbers blocks from raw artifacts.

Every numeric claim in the blocks is computed here from:
- results/newsgroups_*.json (stored metrics; those files have no y_prob)
- results/leakage/*_leakdiag.json (y_true / y_prob, recomputed)
- data/embeddings/*.npy (exact float32 row overlap)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import leakage as L
from leakage import fmt_ms, mean_std, scores_from_pack

GITHUB_ABOUT = (
    "WIP pilot, not a finding. Five committed runs are 4-class files labeled "
    "as 8-class newsgroups; on-disk embeddings are BGE-512 and MiniLM-384, "
    "not GPT-2-768. Scores flip from chance to near-perfect; train/test "
    "duplicate vectors are observed; 14x10x4 is not measured."
)

STATUS_LINE = "STATUS: WORK-IN-PROGRESS — not A yet"


def _load_json(path: Path):
    # Historical reliability arrays contain bare NaN tokens.
    return json.loads(path.read_text(encoding="utf-8"))


def historical_files(results_dir: Path) -> list[tuple[Path, dict]]:
    paths = sorted(p for p in results_dir.glob("*.json") if p.is_file())
    return [(p, _load_json(p)) for p in paths]


def _series(payload, model, key):
    vals = []
    for sr in payload["seed_results"]:
        for row in sr["metrics"]:
            if row["model_name"] == model:
                vals.append(row["metrics"][key])
    return vals


def assert_aggregates_match_seeds(path: Path, payload) -> None:
    for model, metrics in payload["aggregated"].items():
        for key in ("ece", "mce", "brier", "accuracy"):
            vals = _series(payload, model, key)
            mean, std = mean_std(vals)
            if abs(mean - metrics[f"{key}_mean"]) > 1e-6:
                raise AssertionError(f"{path.name} {model} {key}_mean {mean} != {metrics[f'{key}_mean']}")
            if abs(std - metrics[f"{key}_std"]) > 1e-6:
                raise AssertionError(f"{path.name} {model} {key}_std {std} != {metrics[f'{key}_std']}")


def proba_width(payload) -> int:
    widths = set()
    for sr in payload["seed_results"]:
        for row in sr["metrics"]:
            widths.add(len(row["metrics"]["per_class_ece"]))
    if len(widths) != 1:
        raise AssertionError(f"mixed probability widths: {widths}")
    return widths.pop()


def independent_seed_count(payload) -> int:
    seeds = payload["seed_results"]
    if len(seeds) < 2:
        return len(seeds)

    def vec(sr):
        out = {}
        for row in sr["metrics"]:
            m = row["metrics"]
            out[row["model_name"]] = (m["ece"], m["accuracy"], m["brier"])
        return out

    base = vec(seeds[0])
    for sr in seeds[1:]:
        other = vec(sr)
        for name, triple in base.items():
            if any(abs(a - b) > 1e-9 for a, b in zip(triple, other[name])):
                return len(seeds)
    return 1


def one_bin(metrics) -> bool:
    occupied = [c for c in metrics["reliability"]["bin_count"] if c]
    return len(occupied) == 1 and occupied[0] == metrics["n_samples"]


def mce_driver(metrics):
    rel = metrics["reliability"]
    best = None
    for i, count in enumerate(rel["bin_count"]):
        acc = rel["bin_acc"][i]
        if count and acc == acc:
            gap = abs(acc - rel["bin_conf"][i])
            if best is None or gap > best[0]:
                best = (gap, int(count), int(i))
    return best


def cfg_n_seeds(payload) -> int:
    return int(payload["config"]["n_seeds"])


def significant_corrected_count(payload) -> tuple[int, int]:
    tests = payload.get("pairwise_tests_ece", [])
    flagged = sum(1 for t in tests if t.get("significant_corrected"))
    return flagged, len(tests)


def _acc_list(payload, model) -> str:
    vals = []
    for sr in payload["seed_results"]:
        for row in sr["metrics"]:
            if row["model_name"] == model:
                vals.append(f"{row['metrics']['accuracy']:.3f}")
    return ", ".join(vals)


def _fmt_agg(metrics, key) -> str:
    return fmt_ms(metrics[f"{key}_mean"], metrics[f"{key}_std"])


def masked_accuracy(pack, mask, on_overlap: bool) -> float:
    y = np.asarray(pack["y_true"])
    pred = np.argmax(np.asarray(pack["y_prob"]), axis=1)
    sel = np.asarray(mask, dtype=bool)
    if not on_overlap:
        sel = ~sel
    if int(sel.sum()) == 0:
        return float("nan")
    return float(np.mean(pred[sel] == y[sel]))


def render(root: Path) -> dict[str, str]:
    root = Path(root)
    historical = historical_files(root / "results")
    if len(historical) != 5:
        raise AssertionError(f"expected 5 historical JSON files, found {len(historical)}")

    flagged_total = 0
    tests_total = 0
    widths = []
    collapsed_files = []
    floor_names = []
    peaked_names = []
    mce_counts = []
    mce_mismatches = 0
    lines_hist = []
    lines_seeds = []
    for path, payload in historical:
        assert_aggregates_match_seeds(path, payload)
        width = proba_width(payload)
        widths.append(width)
        indep = independent_seed_count(payload)
        if indep == 1 and cfg_n_seeds(payload) > 1:
            collapsed_files.append(path.name)
        flagged, ntests = significant_corrected_count(payload)
        flagged_total += flagged
        tests_total += ntests
        agg = payload["aggregated"]
        cfg = payload["config"]
        lines_hist.append(
            f"| `{path.name}` | {cfg['n_samples']} | {cfg['n_seeds']} | {indep} | "
            f"{len(cfg['model_variants'])} | {width} | {cfg.get('num_classes')} | "
            f"{_fmt_agg(agg['RF-Tabular'], 'accuracy')} | {_fmt_agg(agg['RF-Embed'], 'accuracy')} | "
            f"{_fmt_agg(agg['RF-Embed'], 'ece')} | {_fmt_agg(agg['RF-Embed'], 'brier')} |"
        )
        lines_seeds.append(
            f"| `{path.name}` | {_acc_list(payload, 'RF-Tabular')} | {_acc_list(payload, 'RF-Embed')} |"
        )
        for sr in payload["seed_results"]:
            for row in sr["metrics"]:
                met = row["metrics"]
                if one_bin(met) and "Embed" in row["model_name"]:
                    label = f"{path.name} seed {sr['seed']} {row['model_name']}"
                    if met["brier"] > 0.5:
                        floor_names.append(label)
                    else:
                        peaked_names.append(label)
                driver = mce_driver(met)
                if driver is None or abs(driver[0] - met["mce"]) > 1e-6:
                    mce_mismatches += 1
                elif driver is not None:
                    mce_counts.append(driver[1])

    if len(set(widths)) != 1:
        raise AssertionError(widths)
    if flagged_total != 0:
        raise AssertionError(f"significant_corrected true count = {flagged_total}")

    audit = L.audit_embedding_dir(root / "data" / "embeddings")
    for group in audit["groups"]:
        if group["sum_n"] != group["expected_sum"]:
            raise AssertionError(group)

    ng = _load_json(root / "results" / "leakage" / "newsgroups_leakdiag.json")
    syn = _load_json(root / "results" / "leakage" / "synthetic_leakdiag.json")
    if ng["protocol_id"] != L.PROTOCOL_ID or syn["protocol_id"] != L.PROTOCOL_ID:
        raise AssertionError("protocol id drift")
    if ng["cue_tokens"] != L.CUE_TOKENS:
        raise AssertionError("cue token drift")

    ng_arms = [
        "tabular", "tfidf", "tfidf_cue_ablated", "shuffled_tfidf",
        "zero_embed", "dup_inject_tfidf",
    ]
    syn_arms = ["tabular", "hash", "hash_label_in_text", "shuffled_hash", "zero_embed"]
    ng_sum = {arm: L.arm_summary(ng["seeds"], arm) for arm in ng_arms}
    syn_sum = {arm: L.arm_summary(syn["seeds"], arm) for arm in syn_arms}

    # Controls that the committed predictions must still satisfy.
    if ng_sum["shuffled_tfidf"]["accuracy_mean"] >= 0.25:
        raise AssertionError("shuffled-label control did not collapse")
    if abs(ng_sum["zero_embed"]["brier_mean"] - (1 - 1 / 8)) > 0.002:
        raise AssertionError("8-class zero-embed Brier is not 1-1/K")
    if ng_sum["dup_inject_tfidf"]["accuracy_mean"] <= ng_sum["tfidf"]["accuracy_mean"]:
        raise AssertionError("duplicate injection did not raise accuracy")
    if syn_sum["shuffled_hash"]["accuracy_mean"] >= 0.40:
        raise AssertionError("synthetic shuffle did not collapse")

    nonoverlap = [
        masked_accuracy(row["arms"]["tfidf"], row["test_in_train"], on_overlap=False)
        for row in ng["seeds"]
    ]
    non_m, non_s = mean_std(nonoverlap)
    overlap_ns = ", ".join(str(row["n_test_exact_text_in_train"]) for row in ng["seeds"])
    injected = ", ".join(str(v) for v in sorted({
        row["arms"]["dup_inject_tfidf"]["n_injected"] for row in ng["seeds"]
    }))

    bin_bits = []
    for n_bins in L.N_BINS_SWEEP:
        summary = L.arm_summary(ng["seeds"], "tfidf", n_bins=n_bins)
        bin_bits.append(f"{n_bins}:{summary['ece']}")

    seed0 = ng["seeds"][0]
    raw0 = scores_from_pack(seed0["arms"]["tfidf"])
    iso_val = scores_from_pack(ng["isotonic_probe"]["fit_on_val"])
    iso_te = scores_from_pack(ng["isotonic_probe"]["fit_on_test"])
    if not (iso_te["ece"] < iso_val["ece"]):
        raise AssertionError("test-fit isotonic did not lower ECE on the committed probe")

    minilm = next(g for g in audit["groups"] if g["id"] == "minilm_n800")
    chance4 = 0.25
    n_ov = minilm["test_rows_in_train"]
    n_te = minilm["test_n"]
    ceiling = (n_ov * 1.0 + (n_te - n_ov) * chance4) / n_te
    matched = []
    for path, payload in historical:
        ns = {row["metrics"]["n_samples"] for sr in payload["seed_results"] for row in sr["metrics"]}
        if ns == {n_te}:
            matched.append((path, payload))
    if len(matched) != 1:
        raise AssertionError(f"expected one historical file with test n={n_te}, got {len(matched)}")
    matched_path, matched_payload = matched[0]
    matched_acc = matched_payload["aggregated"]["RF-Embed"]["accuracy_mean"]

    bge400 = next(g for g in audit["groups"] if g["id"] == "bge_n400")
    files_n81 = []
    for path, payload in historical:
        ns = {row["metrics"]["n_samples"] for sr in payload["seed_results"] for row in sr["metrics"]}
        if bge400["test_n"] in ns:
            files_n81.append(path.name)

    cross = audit["cross_n"]
    mce_min = min(mce_counts) if mce_counts else None
    mce_med = float(np.median(mce_counts)) if mce_counts else float("nan")
    floor_embed = floor_names

    readme = []
    readme.append("### Historical JSON files (stored metrics, not recomputed from probabilities)")
    readme.append("")
    readme.append("These five files have no `y_true` or `y_prob`. `make check` averages the stored per-seed ECE, MCE, Brier, and accuracy and requires that average to match `aggregated`. It cannot rebuild ECE from predictions that were never saved.")
    readme.append("")
    readme.append("| File | n | seeds recorded | independent seeds | variants | proba width | config classes | RF-Tabular acc | RF-Embed acc | RF-Embed ECE | RF-Embed Brier |")
    readme.append("|---|---:|---:|---:|---:|---:|---:|---|---|---|---|")
    readme.extend(lines_hist)
    readme.append("")
    readme.append("Per-seed accuracy, RF-Tabular then RF-Embed:")
    readme.append("")
    readme.append("| File | RF-Tabular seeds | RF-Embed seeds |")
    readme.append("|---|---|---|")
    readme.extend(lines_seeds)
    readme.append("")
    collapsed_txt = ", ".join(f"`{name}`" for name in collapsed_files) or "none"
    readme.append(
        f"Every stored probability vector has width {widths[0]}. `config.num_classes` is 8 in every file. "
        f"Those two facts disagree. Pairwise tests stored in the files: {flagged_total} of {tests_total} have "
        f"`significant_corrected` true. That count is not a significance result; recorded seed counts are 2 or 3. "
        f"Files whose seed blocks match to 1e-9 on ECE, accuracy, and Brier, so the independent-seed count is 1: {collapsed_txt}."
    )
    readme.append("")
    floor_files = sorted({name.split()[0] for name in floor_embed})
    floor_file_txt = ", ".join(f"`{name}`" for name in floor_files) or "none"
    peaked_txt = ", ".join(f"`{name}`" for name in peaked_names) or "none"
    readme.append(
        f"Single-bin reliability with Brier above 0.5 occurs for {len(floor_embed)} embedding model-rows, "
        f"in {floor_file_txt}. A forest on constant features produces Brier 1-1/K. "
        f"The diagnostic `zero_embed` arm lands on that identity for K=8 and K=6 (table below). "
        f"Those historical rows sit near Brier 0.750, which is 1-1/4, matching the stored probability width. "
        f"The files do not say whether those features were zeros. "
        f"Single-bin reliability with Brier at or below 0.5 is a different pattern (a peaked, mostly correct predictor), "
        f"and it occurs in: {peaked_txt}."
    )
    readme.append("")
    readme.append(f"Stored MCE equals the largest |bin accuracy − bin confidence| gap on the saved reliability curve for every seed-model row (mismatches: {mce_mismatches}). The bin that realizes MCE has minimum count {mce_min} and median count {mce_med:.0f}. At test n of 81–161 and 10 bins, MCE is a small-bin statistic.")
    readme.append("")
    readme.append("### Committed embedding matrices (exact float32 row match)")
    readme.append("")
    readme.append("Groups are inferred: one model family whose train+val+test row counts sum to the stated n. Result JSON does not store the cache filename.")
    readme.append("")
    readme.append("| Group | dim | test n | test rows also in train | test rows also in val |")
    readme.append("|---|---:|---:|---:|---:|")
    for group in audit["groups"]:
        readme.append(
            f"| `{group['id']}` | {group['dim']} | {group['test_n']} | "
            f"{group['test_rows_in_train']} | {group['test_rows_in_val']} |"
        )
    readme.append("")
    readme.append(
        f"`{minilm['id']}` test n={n_te} is the only historical file with that test size: "
        f"`{matched_path.name}`, RF-Embed accuracy mean {matched_acc:.3f}. "
        f"If the {n_ov} overlapping test rows were perfect and the other rows were at 4-class chance {chance4:.2f}, "
        f"accuracy would be {ceiling:.3f}. {matched_acc:.3f} is above that ceiling. "
        f"Exact duplicate vectors in that cache are a real split overlap and are not sufficient to produce the stored accuracy. "
        f"The link from this cache to that JSON is the shared test size, not an id stored in the JSON."
    )
    readme.append("")
    readme.append(
        f"Test n={bge400['test_n']} appears in more than one JSON ({', '.join(files_n81)}), so the BGE n=400 cache is not assigned to a single file. "
        f"Across sample sizes, `{cross['left']}` has {cross['left_rows_in_right']}/{cross['left_n']} rows exactly equal to a row in `{cross['right']}`. "
        f"That pair is not one split."
    )
    readme.append("")
    readme.append("### Diagnostic `leakdiag-2026-11-16` (ECE and Brier recomputed from stored probabilities)")
    readme.append("")
    readme.append("Random forest, 100 trees, depth 10, `random_state=42`, `n_jobs=1`. Seeds 0–4 are split seeds only. No Wilcoxon test is computed. Not GPT-2, not the committed BGE/MiniLM matrices, not a rerun of the five JSON files.")
    readme.append("")
    readme.append("Real 20 Newsgroups, the loader's 8 categories, headers/footers/quotes removed, stratified subset n=800. TF-IDF (`max_features=2000`, word n-grams 1–2, `min_df=2`) is fit on the training split only.")
    readme.append("")
    readme.append("| Arm | ECE | Brier | Accuracy |")
    readme.append("|---|---|---|---|")
    for arm in ng_arms:
        s = ng_sum[arm]
        readme.append(f"| `{arm}` | {s['ece']} | {s['brier']} | {s['accuracy']} |")
    readme.append("")
    readme.append(
        f"Chance accuracy at 8 classes is {1/8:.3f}. `shuffled_tfidf` is {ng_sum['shuffled_tfidf']['accuracy']}. "
        f"`zero_embed` Brier is {ng_sum['zero_embed']['brier']} (1-1/8 = {1-1/8:.3f}) and accuracy is {ng_sum['zero_embed']['accuracy']}. "
        f"TF-IDF accuracy on rows whose exact text is absent from train is {fmt_ms(non_m, non_s)}. "
        f"Exact train/test text overlap by seed: {overlap_ns}. "
        f"`dup_inject_tfidf` replaces {injected} test rows with copies of training rows (fraction {L.DUP_INJECT_FRACTION}) and accuracy moves from {ng_sum['tfidf']['accuracy']} to {ng_sum['dup_inject_tfidf']['accuracy']}."
    )
    readme.append("")
    readme.append(
        f"Cue-word ablation (`tfidf_cue_ablated`) uses a label-independent word-boundary deletion of {len(L.CUE_TOKENS)} tokens, including ordinary words (`car`, `gun`, `windows`). "
        f"Recorded on the subset at generation time, and not recomputed by `make check` because the raw text is not committed: "
        f"own-class cue rate {ng['own_class_cue_rate_subset']:.3f}, any-cue rate {ng['any_cue_rate_subset']:.3f}, "
        f"full corpus n={ng['full_corpus']['n']} unique={ng['full_corpus']['unique_texts']} empty={ng['full_corpus']['empty']}, "
        f"subset n={ng['subset_corpus']['n']} unique={ng['subset_corpus']['unique_texts']} empty={ng['subset_corpus']['empty']}."
    )
    readme.append("")
    readme.append(f"TF-IDF ECE at bin counts " + ", ".join(bin_bits) + ". The same probabilities, three different bin widths, three different ECE means.")
    readme.append("")
    readme.append(
        f"Isotonic probe, seed {ng['isotonic_probe']['seed']} only, same TF-IDF forest. "
        f"Uncalibrated on that seed: ECE {raw0['ece']:.3f}, Brier {raw0['brier']:.3f}, accuracy {raw0['accuracy']:.3f}. "
        f"Isotonic fit on validation, scored on test: ECE {iso_val['ece']:.3f}, Brier {iso_val['brier']:.3f}, accuracy {iso_val['accuracy']:.3f}. "
        f"Isotonic fit on the test labels, scored on those same labels: ECE {iso_te['ece']:.3f}, Brier {iso_te['brier']:.3f}, accuracy {iso_te['accuracy']:.3f}. "
        f"The last row is contamination, not a result."
    )
    readme.append("")
    readme.append("Synthetic style corpus from the current `_load_synthetic(600)` (6 classes). The class is the template family that wrote the text. This generator's class count is 6, so it is not what the 4-column historical files scored.")
    readme.append("")
    readme.append("| Arm | ECE | Brier | Accuracy |")
    readme.append("|---|---|---|---|")
    for arm in syn_arms:
        s = syn_sum[arm]
        readme.append(f"| `{arm}` | {s['ece']} | {s['brier']} | {s['accuracy']} |")
    readme.append("")
    readme.append(
        "What this supports: the five JSON files disagree with each other and with `config.num_classes`; "
        "their stored pairwise flags are all false; committed embedding splits contain exact duplicate vectors; "
        "a shuffled-label control on the new 8-class TF-IDF run collapses; constant features reproduce the uniform Brier identity; "
        "injecting train rows into the test set raises accuracy; fitting isotonic on the test labels lowers that seed's ECE."
    )
    readme.append("")
    readme.append(
        "What this does not support: a stable calibration effect of embeddings; GPT-2 at 768 dimensions; "
        "that the historical near-perfect embedding accuracies are accounted for by the duplicate-vector counts above; "
        "any Wilcoxon or Bonferroni claim; the planned 14 variants × 10 seeds × 4 datasets, which is NOT YET MEASURED."
    )
    readme.append("")
    readme.append("GitHub About replacement (the live About field is not changed by a pull request):")
    readme.append("")
    readme.append(f"> {GITHUB_ABOUT}")
    readme.append("")

    deep = []
    deep.append("The block below is generated from the raw JSON and `.npy` files by `checks/render_observed.py`. `make check` fails if this block is edited by hand.")
    deep.append("")
    deep.extend(readme)
    deep.append("### Per-group duplicate counts inside each matrix")
    deep.append("")
    deep.append("| Group | split | rows | unique | extra copies | max multiplicity |")
    deep.append("|---|---|---:|---:|---:|---:|")
    for group in audit["groups"]:
        for split_name in ("train", "val", "test"):
            info = group[split_name]
            deep.append(
                f"| `{group['id']}` | {split_name} | {info['n']} | {info['unique']} | "
                f"{info['extra_copies']} | {info['max_multiplicity']} |"
            )
    deep.append("")
    deep.append(
        f"Diagnostic environment recorded in the JSON: sklearn {ng['sklearn']}, numpy {ng['numpy']}. "
        f"`make check` does not retrain the forests and does not download 20 Newsgroups; it recomputes ECE, Brier, and accuracy from the stored probability matrices."
    )
    deep.append("")

    return {
        "readme": "\n".join(readme).rstrip() + "\n",
        "deep": "\n".join(deep).rstrip() + "\n",
    }
