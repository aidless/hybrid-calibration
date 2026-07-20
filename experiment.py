"""
Main Experiment Runner: Hybrid Tree-Ensemble + LLM Calibration Study.

Pipeline:
  1. Load dataset & extract tabular features
  2. Extract GPT-2 embeddings (cached to disk)
  3. For each random seed:
     a. Split data
     b. Train all model variants
     c. Evaluate calibration (ECE, MCE, Brier, reliability)
  4. Aggregate results across seeds
  5. Statistical significance tests
  6. Save results to JSON
"""

import time
import copy
import numpy as np
from typing import Dict, List, Optional

from config import (
    MODEL_VARIANTS,
    RANDOM_SEEDS,
    N_BINS,
    DATASET_CONFIGS,
    RESULTS_DIR,
    EMBEDDING_DIM,
    EMBEDDING_MODEL_NAME,
)
from data_loader import prepare_data
from embedding_extractor import EmbeddingExtractor, build_feature_matrix
from models import create_model, BaseModel
from calibration import evaluate_calibration
from utils import save_results, build_summary_table, compute_pairwise_tests


def run_experiment(
    dataset_name: str = "imdb",
    n_samples: Optional[int] = None,
    model_filter: Optional[List[str]] = None,
    skip_embeddings: bool = False,
    verbose: bool = True,
) -> Dict:
    """
    Run the full hybrid calibration experiment on one dataset.

    Args:
        dataset_name: 'imdb' | 'ag_news' | 'yelp_polarity'.
        n_samples: Subset size (None = use config default).
        model_filter: If provided, only run these model names.
        skip_embeddings: Skip embedding extraction (for debugging).
        verbose: Print progress.

    Returns:
        Full results dict.
    """
    cfg = DATASET_CONFIGS[dataset_name]
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    if verbose:
        print("=" * 70)
        print(f"EXPERIMENT: Hybrid Calibration Study")
        print(f"  Dataset: {dataset_name} ({cfg['task']}, {cfg['num_classes']} classes)")
        print(f"  Seeds: {len(RANDOM_SEEDS)}")
        print(f"  Timestamp: {timestamp}")
        print("=" * 70)

    # ============================================================
    # Phase 1: Data Preparation
    # ============================================================
    if verbose:
        print("\n[Phase 1] Loading data ...")
    t0 = time.time()

    data = prepare_data(
        dataset_name=dataset_name,
        n_samples=n_samples or cfg.get("n_samples"),
        seed=42  # Fixed seed for initial split; per-seed splits will vary
    )
    if verbose:
        print(f"  Train: {len(data['texts_train'])}, "
              f"Val: {len(data['texts_val'])}, "
              f"Test: {len(data['texts_test'])}")
        print(f"  Tabular features: {data['X_train_tab'].shape[1]} dims")
        print(f"  Data source: {data.get('data_source', 'unknown')}")
        if data.get('data_source') == 'synthetic':
            print(f"  ⚠ WARNING: Using synthetic data! Results may not reflect real-world performance.")
        print(f"  Time: {time.time() - t0:.1f}s")

    # ============================================================
    # Phase 2: Embedding Extraction (cached)
    # ============================================================
    if not skip_embeddings:
        if verbose:
            print("\n[Phase 2] Extracting LLM embeddings ...")
        t0 = time.time()

        extractor = EmbeddingExtractor()

        emb_train = extractor.extract_with_cache(
            data["texts_train"], dataset_name, "train"
        )
        emb_val = extractor.extract_with_cache(
            data["texts_val"], dataset_name, "val"
        )
        emb_test = extractor.extract_with_cache(
            data["texts_test"], dataset_name, "test"
        )

        if verbose:
            print(f"  Embedding dim: {emb_train.shape[1]}")
            print(f"  Time: {time.time() - t0:.1f}s")
    else:
        if verbose:
            print("\n[Phase 2] Skipping embeddings (dummy zeros) ...")
        emb_dim = EMBEDDING_DIM  # Use config dimension, not hardcoded
        emb_train = np.zeros((len(data["texts_train"]), emb_dim), dtype=np.float32)
        emb_val = np.zeros((len(data["texts_val"]), emb_dim), dtype=np.float32)
        emb_test = np.zeros((len(data["texts_test"]), emb_dim), dtype=np.float32)

    # ============================================================
    # Phase 3: Per-Seed Experiments
    # ============================================================
    # Filter models
    variants = MODEL_VARIANTS
    if model_filter:
        variants = [v for v in variants if v.name in model_filter]
        if verbose:
            print(f"\n[Phase 3] Running {len(variants)}/{len(MODEL_VARIANTS)} model variants "
                  f"(filtered by: {model_filter})")
    else:
        if verbose:
            print(f"\n[Phase 3] Running {len(variants)} model variants across {len(RANDOM_SEEDS)} seeds")

    all_seed_results = []

    for seed_idx, seed in enumerate(RANDOM_SEEDS):
        if verbose:
            print(f"\n{'─' * 50}")
            print(f"  Seed {seed_idx + 1}/{len(RANDOM_SEEDS)} (seed={seed})")
            print(f"{'─' * 50}")

        seed_result = _run_single_seed(
            data=data,
            emb_train=emb_train,
            emb_val=emb_val,
            emb_test=emb_test,
            variants=variants,
            seed=seed,
            verbose=verbose,
        )
        all_seed_results.append(seed_result)

    # ============================================================
    # Phase 4: Aggregate & Statistics
    # ============================================================
    if verbose:
        print("\n" + "=" * 70)
        print("Phase 4: Aggregate Results")
        print("=" * 70)

    # Aggregate metrics across seeds
    aggregated = _aggregate_results(all_seed_results, variants)

    # Summary table
    if verbose:
        print("\n" + build_summary_table(all_seed_results))

    # Pairwise statistical tests for ECE
    if verbose:
        print("\n[Statistical Tests] ECE pairwise (Wilcoxon, Bonferroni corrected)")
    ece_scores = {}
    for v in variants:
        name = v.name
        vals = []
        for sr in all_seed_results:
            for r in sr["metrics"]:
                if r["model_name"] == name:
                    vals.append(r["metrics"]["ece"])
        ece_scores[name] = vals

    pairwise_tests = compute_pairwise_tests(ece_scores, metric_name="ece")

    if verbose:
        sig_count = sum(1 for pt in pairwise_tests if pt["significant_corrected"])
        print(f"  Significant pairs (corrected): {sig_count}/{len(pairwise_tests)}")
        for pt in pairwise_tests:
            if pt["significant_corrected"]:
                direction = "↓" if pt["mean_diff"] < 0 else "↑"
                print(f"    {pt['model_a']} vs {pt['model_b']}: "
                      f"Δ={pt['mean_diff']:.4f} "
                      f"p={pt['p_value_corrected']:.4f} {direction}")

    # ============================================================
    # Phase 5: ACL Diagnostics (Embedding Quality + Version)
    # ============================================================
    final_results = {
        "dataset": dataset_name,
        "timestamp": timestamp,
        "config": {
            "n_samples": n_samples or cfg.get("n_samples"),
            "n_seeds": len(RANDOM_SEEDS),
            "n_bins": N_BINS,
            "model_variants": [v.name for v in variants],
            "num_classes": cfg["num_classes"],
        },
        "seed_results": all_seed_results,
        "aggregated": aggregated,
        "pairwise_tests_ece": pairwise_tests,
    }

    # Inject embedding model version metadata (ACL 2027 Gene 3)
    from config import EMBEDDING_MODELS
    for key, ecfg in EMBEDDING_MODELS.items():
        if ecfg["name"] == EMBEDDING_MODEL_NAME:
            final_results = inject_version_metadata(final_results, key)
            break

    # Embedding quality diagnosis (ACL 2027 Gene 2: floor effect)
    if verbose:
        diagnosis = diagnose_embedding_quality(final_results, verbose=True)
        final_results["embedding_diagnosis"] = diagnosis

    save_results(final_results, dataset_name, timestamp)

    if verbose:
        print(f"\nExperiment complete. Results saved to {RESULTS_DIR}/")
        print("=" * 70)

    return final_results


def _run_single_seed(
    data: Dict,
    emb_train: np.ndarray,
    emb_val: np.ndarray,
    emb_test: np.ndarray,
    variants: List,
    seed: int,
    verbose: bool = True,
) -> Dict:
    """Run all model variants for a single random seed."""

    from sklearn.model_selection import train_test_split

    # Re-split data with this seed to get per-seed variation
    # Combine all data and re-split
    all_texts = data["texts_train"] + data["texts_val"] + data["texts_test"]
    all_labels = np.concatenate([data["y_train"], data["y_val"], data["y_test"]])
    all_emb = np.concatenate([emb_train, emb_val, emb_test], axis=0)
    all_tab = np.concatenate([data["X_train_tab"], data["X_val_tab"], data["X_test_tab"]], axis=0)

    # Split: train / val / test
    from config import TEST_SIZE, VAL_SIZE
    texts_tr, texts_temp, y_tr, y_temp, emb_tr, emb_temp, tab_tr, tab_temp = train_test_split(
        all_texts, all_labels, all_emb, all_tab,
        test_size=TEST_SIZE + VAL_SIZE,
        random_state=seed, stratify=all_labels,
    )
    val_ratio = VAL_SIZE / (TEST_SIZE + VAL_SIZE)
    texts_va, texts_te, y_va, y_te, emb_va, emb_te, tab_va, tab_te = train_test_split(
        texts_temp, y_temp, emb_temp, tab_temp,
        test_size=1 - val_ratio,
        random_state=seed, stratify=y_temp,
    )

    # Build feature matrices for each feature_set
    feature_matrices = {
        "train": {
            "tabular": tab_tr,
            "embeddings": emb_tr,
            "hybrid": build_feature_matrix(emb_tr, tab_tr, "hybrid"),
        },
        "val": {
            "tabular": tab_va,
            "embeddings": emb_va,
            "hybrid": build_feature_matrix(emb_va, tab_va, "hybrid"),
        },
        "test": {
            "tabular": tab_te,
            "embeddings": emb_te,
            "hybrid": build_feature_matrix(emb_te, tab_te, "hybrid"),
        },
    }

    metrics_list = []
    base_models = {}

    for variant in variants:
        t0 = time.time()

        # Get feature matrix
        X_tr = feature_matrices["train"][variant.feature_set]
        X_va = feature_matrices["val"][variant.feature_set]
        X_te = feature_matrices["test"][variant.feature_set]

        # Create model
        model: BaseModel
        if variant.model_type == "calibrated":
            model = create_model(variant, base_models=base_models)
        else:
            model = create_model(variant)

        # Train
        model.fit(X_tr, y_tr, X_val=X_va, y_val=y_va)

        # Evaluate
        y_prob = model.predict_proba(X_te)
        # Ensure shape is (n, n_classes)
        if y_prob.ndim == 1:
            n_class = len(np.unique(y_te))
            y_prob_2d = np.zeros((len(y_prob), n_class))
            y_prob_2d[:, 1] = y_prob
            y_prob_2d[:, 0] = 1 - y_prob
            y_prob = y_prob_2d

        eval_result = evaluate_calibration(
            y_te, y_prob, n_bins=N_BINS, model_name=variant.name
        )
        eval_result["train_time_s"] = round(time.time() - t0, 2)
        eval_result["feature_set"] = variant.feature_set
        eval_result["model_type"] = variant.model_type

        metrics_list.append({
            "model_name": variant.name,
            "feature_set": variant.feature_set,
            "model_type": variant.model_type,
            "metrics": eval_result,
        })

        # Store for calibrated variants
        if variant.model_type != "calibrated":
            base_models[variant.name] = model

        if verbose:
            ece = eval_result["ece"]
            acc = eval_result["accuracy"]
            print(f"    {variant.name:<25} ECE={ece:.4f}  ACC={acc:.4f}  "
                  f"({eval_result['train_time_s']}s)")

    return {
        "seed": seed,
        "metrics": metrics_list,
    }


def _aggregate_results(all_seed_results: List[Dict], variants: List) -> Dict:
    """Aggregate metrics across seeds: mean ± std."""
    aggregated = {}

    for variant in variants:
        name = variant.name
        ece_vals, acc_vals, auroc_vals, f1_vals, time_vals = [], [], [], [], []

        for sr in all_seed_results:
            for r in sr["metrics"]:
                if r["model_name"] == name:
                    m = r["metrics"]
                    ece_vals.append(m["ece"])
                    acc_vals.append(m["accuracy"])
                    auroc_vals.append(m.get("auroc", np.nan))
                    f1_vals.append(m.get("f1", np.nan))
                    time_vals.append(r["metrics"]["train_time_s"])

        def _ms(arr):
            m = np.nanmean(arr) if len(arr) > 0 else np.nan
            s = np.nanstd(arr, ddof=1) if len(arr) > 1 else 0.0
            return float(m), float(s)

        ecm, ecs = _ms(ece_vals)
        acm, acs = _ms(acc_vals)
        aum, aus = _ms(auroc_vals)
        f1m, f1s = _ms(f1_vals)

        aggregated[name] = {
            "ece_mean": ecm, "ece_std": ecs,
            "accuracy_mean": acm, "accuracy_std": acs,
            "auroc_mean": aum, "auroc_std": aus,
            "f1_mean": f1m, "f1_std": f1s,
            "train_time_mean_s": float(np.mean(time_vals)),
        }

    return aggregated


# ============================================================
# ACL 2027 Gene 1: N-Sensitivity Analysis
# ============================================================
# From EPC paper: small-N batches show strong coupling, larger-N
# batches collapse. We test the same pattern for calibration:
# does ECE stabilize, grow, or shrink with sample size?

def run_n_sensitivity(
    dataset_name: str = "newsgroups",
    embedding_model: str = "bge-zh",
    sample_sizes: List[int] = None,
    n_seeds: int = 5,
    verbose: bool = True,
) -> Dict:
    """
    Test whether calibration metrics are N-dependent.

    Returns per-sample-size ECE/accuracy patterns for N-sensitivity
    diagnosis. Parallels ACL 2027 Tables 5-6.
    """
    if sample_sizes is None:
        sample_sizes = [200, 500, 1000, 2000, 5000]

    import config
    from config import EMBEDDING_MODELS

    ecfg = EMBEDDING_MODELS[embedding_model]
    config.EMBEDDING_MODEL_NAME = ecfg["name"]
    config.EMBEDDING_DIM = ecfg["dim"]

    # Save original seeds, use subset
    orig_seeds = list(config.RANDOM_SEEDS)
    config.RANDOM_SEEDS.clear()
    config.RANDOM_SEEDS.extend(list(range(n_seeds)))

    results = {}
    for n in sample_sizes:
        if verbose:
            print(f"\n[N-SENSITIVITY] n={n}, seeds={n_seeds}")
        r = run_experiment(
            dataset_name=dataset_name,
            n_samples=n,
            model_filter=["RF-Embed", "RF-Hybrid", "MLP-Embed"],
            verbose=verbose,
        )
        results[str(n)] = r

    # Restore
    config.RANDOM_SEEDS.clear()
    config.RANDOM_SEEDS.extend(orig_seeds)

    # Compute N-sensitivity summary
    summary = {}
    for n_str, r in results.items():
        agg = r.get("aggregated", {})
        summary[n_str] = {
            name: {"ece": v["ece_mean"], "acc": v["accuracy_mean"]}
            for name, v in agg.items()
        }

    from utils import save_results
    save_results({
        "analysis": "n_sensitivity",
        "embedding_model": embedding_model,
        "embedding_version": ecfg.get("version", "unknown"),
        "dataset": dataset_name,
        "sample_sizes": sample_sizes,
        "n_seeds": n_seeds,
        "results": results,
        "summary": summary,
    }, f"n_sensitivity_{dataset_name}")

    if verbose:
        print("\n[N-SENSITIVITY] ECE vs N:")
        for name in ["MLP-Embed", "RF-Hybrid", "RF-Embed"]:
            ece_vals = [summary[str(n)].get(name, {}).get("ece", np.nan) for n in sample_sizes]
            print(f"  {name}: " + " → ".join(f"{v:.4f}" for v in ece_vals))

    return results


# ============================================================
# ACL 2027 Gene 2: Embedding Quality Diagnostic (Floor Effect)
# ============================================================
# From EPC paper: DeepSeek self-eval 97% zero-coupling looked
# stable, but ECE=0.31 revealed it was incapable (floor effect).
# Similarly, if an embedding model produces low ECE but also
# near-random accuracy, it's not "well-calibrated" — it's incapable.

def diagnose_embedding_quality(
    results: Dict,
    embedding_model: str = "bge-zh",
    verbose: bool = True,
) -> Dict:
    """
    Detect floor effects in embedding quality using ECE+accuracy.

    A model with ECE≈0 AND accuracy≈random is NOT well-calibrated —
    it simply hasn't learned anything to be overconfident about.
    This parallels ACL 2027's finding that DeepSeek self-eval 97%
    zero-coupling was a floor effect, not genuine stability.
    """
    aggregated = results.get("aggregated", {})
    diagnosis = {}

    n_classes = results.get("config", {}).get("num_classes", 4)
    random_acc = 1.0 / n_classes

    for name, metrics in aggregated.items():
        ece = metrics.get("ece_mean", np.nan)
        acc = metrics.get("accuracy_mean", np.nan)

        # Floor effect: ECE < 0.02 AND accuracy < 2× random
        floor_effect = (ece < 0.02 and acc < 2 * random_acc + 0.05)
        # Overconfidence: ECE > 0.08 AND accuracy > 2× random
        overconfident = (ece > 0.08 and acc > 2 * random_acc)

        diagnosis[name] = {
            "ece": round(ece, 4),
            "accuracy": round(acc, 4),
            "random_baseline": round(random_acc, 4),
            "floor_effect": floor_effect,
            "overconfident": overconfident,
            "interpretation": (
                "Floor effect: model cannot discriminate, ECE artificially low"
                if floor_effect
                else "Overconfident: model discriminates but probability is miscalibrated"
                if overconfident
                else "Moderate: further analysis needed"
            ),
        }

    if verbose:
        print("\n[EMBEDDING QUALITY DIAGNOSIS]")
        for name, d in diagnosis.items():
            flag = "⚠ FLOOR" if d["floor_effect"] else ("⚠ OVERCONFIDENT" if d["overconfident"] else "  OK")
            print(f"  {flag} {name}: ECE={d['ece']:.4f} ACC={d['accuracy']:.4f} "
                  f"(random={d['random_baseline']:.2f}) — {d['interpretation']}")

    return diagnosis


# ============================================================
# ACL 2027 Gene 3: Version Drift Detection
# ============================================================
# From EPC paper: GPT-4o May→June drift inverted results.
# For embedding models, we track: model name, version tag,
# download date, and source (ModelScope vs HuggingFace).

def inject_version_metadata(results: Dict, embedding_model_key: str = "bge-zh") -> Dict:
    """Add embedding model version metadata to results."""
    from config import EMBEDDING_MODELS
    ecfg = EMBEDDING_MODELS.get(embedding_model_key, {})
    if "version_metadata" not in results:
        results["version_metadata"] = {}
    results["version_metadata"]["embedding_model"] = {
        "key": embedding_model_key,
        "name": ecfg.get("name", "unknown"),
        "version": ecfg.get("version", "unknown"),
        "source": ecfg.get("source", "unknown"),
        "language": ecfg.get("language", "unknown"),
        "dim": ecfg.get("dim", 0),
    }
    return results


# ============================================================
# Full Study: Multi-Model × Multi-Size × Cross-Dataset
# ============================================================

def run_full_study(
    dataset_names=None,
    embedding_models=None,
    sample_sizes=None,
    model_filter=None,
    verbose=True,
):
    """
    Run the complete study matrix matching the EPC/BOUNDARY_SYNC design:
      Datasets × Embedding Models × Sample Sizes × Tree Models

    Each factor isolated, measured, and statistically compared.
    """
    from config import EMBEDDING_MODELS, SAMPLE_SIZE_SWEEP

    if dataset_names is None:
        dataset_names = ["newsgroups"]
    if embedding_models is None:
        embedding_models = list(EMBEDDING_MODELS.keys())
    if sample_sizes is None:
        sample_sizes = [1000]
    if model_filter is None:
        model_filter = [
            "RF-Tabular", "RF-Embed", "RF-Hybrid",
            "LGB-Hybrid", "MLP-Embed",
            "RF-Hybrid-Isotonic", "MLP-Embed-Isotonic",
        ]

    all_results = {}
    total_runs = len(dataset_names) * len(embedding_models) * len(sample_sizes)

    for dataset_name in dataset_names:
        all_results[dataset_name] = {}
        for emb_key in embedding_models:
            emb_cfg = EMBEDDING_MODELS[emb_key]
            all_results[dataset_name][emb_key] = {}

            # Override embedding config
            import config
            config.EMBEDDING_MODEL_NAME = emb_cfg["name"]
            config.EMBEDDING_DIM = emb_cfg["dim"]

            for n_samples in sample_sizes:
                r = run_experiment(
                    dataset_name=dataset_name,
                    n_samples=n_samples,
                    model_filter=model_filter,
                    verbose=verbose,
                )
                all_results[dataset_name][emb_key][str(n_samples)] = r

    from utils import save_results
    save_results({
        "study_type": "full_study",
        "factors": {
            "datasets": dataset_names,
            "embedding_models": embedding_models,
            "sample_sizes": sample_sizes,
        },
        "results": all_results,
    }, "full_study")
    return all_results


# ============================================================
# CLI Entry Point
# ============================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Hybrid Tree-Ensemble + LLM Calibration Experiment"
    )
    parser.add_argument(
        "--dataset", type=str, default="newsgroups",
        choices=["imdb", "ag_news", "yelp_polarity", "newsgroups"],
        help="Dataset to use"
    )
    parser.add_argument(
        "--n_samples", type=int, default=None,
        help="Number of samples (default: use config value)"
    )
    parser.add_argument(
        "--models", type=str, nargs="*", default=None,
        help="Specific model names to run (default: all)"
    )
    parser.add_argument(
        "--skip-embeddings", action="store_true",
        help="Skip LLM embedding extraction (use zeros for testing)"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick test: 2 seeds, 400 samples, BGE embeddings"
    )
    parser.add_argument(
        "--full", action="store_true",
        help="Full experiment: 10 seeds, 1000 samples, all models"
    )
    parser.add_argument(
        "--study", action="store_true",
        help="Run full multi-factor study (embedding models × sample sizes)"
    )
    parser.add_argument(
        "--n-sensitivity", action="store_true",
        help="ACL 2027: N-sensitivity analysis (ECE vs sample size sweep)"
    )
    parser.add_argument(
        "--emb-model", type=str, default="bge-zh",
        choices=["bge-zh", "minilm-en"],
        help="Embedding model to use"
    )

    args = parser.parse_args()

    if args.quick:
        import config
        config.RANDOM_SEEDS.clear()
        config.RANDOM_SEEDS.extend(list(range(10)))
        from config import EMBEDDING_MODELS
        ecfg = EMBEDDING_MODELS.get(args.emb_model, EMBEDDING_MODELS["bge-zh"])
        config.EMBEDDING_MODEL_NAME = ecfg["name"]
        config.EMBEDDING_DIM = ecfg["dim"]
        run_experiment(
            dataset_name=args.dataset,
            n_samples=1000,
            model_filter=["RF-Tabular", "RF-Embed", "RF-Hybrid",
                          "MLP-Embed", "RF-Hybrid-Isotonic", "MLP-Embed-Isotonic"],
            skip_embeddings=False,
        )
    elif args.study:
        run_full_study(
            dataset_names=[args.dataset],
            embedding_models=[args.emb_model],
            sample_sizes=[500, 1000, 2000],
        )
    elif args.n_sensitivity:
        run_n_sensitivity(
            dataset_name=args.dataset,
            embedding_model=args.emb_model,
            sample_sizes=[200, 500, 1000, 2000, 5000],
            n_seeds=5,
        )
    else:
        import config
        n_seeds = len(config.RANDOM_SEEDS)
        print(f"Running with {n_seeds} seeds. Use --quick for a fast test.")
        # Use selected embedding model
        from config import EMBEDDING_MODELS
        ecfg = EMBEDDING_MODELS.get(args.emb_model, EMBEDDING_MODELS["bge-zh"])
        config.EMBEDDING_MODEL_NAME = ecfg["name"]
        config.EMBEDDING_DIM = ecfg["dim"]
        run_experiment(
            dataset_name=args.dataset,
            n_samples=args.n_samples,
            model_filter=args.models,
            skip_embeddings=args.skip_embeddings,
        )
