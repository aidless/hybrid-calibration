"""
Utilities: statistical tests, result serialization, helpers.
"""

import json
import os
import time
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy import stats
from config import RESULTS_DIR


# ============================================================
# Statistical Tests
# ============================================================

def paired_wilcoxon(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    alpha: float = 0.05,
) -> Dict:
    """
    Paired Wilcoxon signed-rank test.

    Args:
        scores_a: Array of scores for method A (across seeds/splits).
        scores_b: Array of scores for method B.
        alpha: Significance level.

    Returns:
        Dict with statistic, p_value, significant, method, etc.
    """
    diff = scores_a - scores_b

    # If all differences are zero, no test needed
    if np.allclose(diff, 0):
        return {
            "statistic": 0.0,
            "p_value": 1.0,
            "significant": False,
            "method": "wilcoxon",
            "mean_diff": 0.0,
            "std_diff": 0.0,
        }

    try:
        stat, p = stats.wilcoxon(scores_a, scores_b, zero_method="zsplit")
    except ValueError:
        # Fallback if wilcoxon fails
        stat, p = stats.ttest_rel(scores_a, scores_b)

    return {
        "statistic": float(stat),
        "p_value": float(p),
        "significant": bool(p < alpha),
        "method": "wilcoxon",
        "mean_diff": float(np.mean(scores_a) - np.mean(scores_b)),
        "std_diff": float(np.std(scores_a - scores_b, ddof=1)),
    }


def bonferroni_correction(p_values: List[float], alpha: float = 0.05) -> List[float]:
    """Apply Bonferroni correction to a list of p-values."""
    n = len(p_values)
    corrected = [min(p * n, 1.0) for p in p_values]
    return corrected


def compute_pairwise_tests(
    results_dict: Dict[str, List[float]],
    metric_name: str = "ece",
    alpha: float = 0.05,
) -> List[Dict]:
    """
    Compute all pairwise statistical comparisons.

    Args:
        results_dict: {model_name: [score_seed1, score_seed2, ...]}
        metric_name: Name of the metric being compared.
        alpha: Significance level.

    Returns:
        List of pairwise comparison dicts.
    """
    model_names = sorted(results_dict.keys())
    comparisons = []

    for i, name_a in enumerate(model_names):
        for j, name_b in enumerate(model_names):
            if i >= j:
                continue
            scores_a = np.array(results_dict[name_a])
            scores_b = np.array(results_dict[name_b])
            test_result = paired_wilcoxon(scores_a, scores_b, alpha=alpha)
            comparisons.append({
                "model_a": name_a,
                "model_b": name_b,
                "metric": metric_name,
                **test_result,
            })

    # Apply Bonferroni correction
    all_p = [c["p_value"] for c in comparisons]
    corrected_p = bonferroni_correction(all_p, alpha=alpha)
    for c, cp in zip(comparisons, corrected_p):
        c["p_value_corrected"] = cp
        c["significant_corrected"] = cp < alpha

    return comparisons


# ============================================================
# Result Persistence
# ============================================================

def save_results(
    results: Dict,
    dataset_name: str,
    timestamp: Optional[str] = None,
) -> str:
    """Save experiment results to JSON."""
    if timestamp is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"{dataset_name}_{timestamp}.json"
    filepath = os.path.join(RESULTS_DIR, filename)

    # Convert numpy types for JSON serialization
    serializable = _make_serializable(results)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)
    print(f"[Results] Saved to {filepath}")
    return filepath


def load_results(filepath: str) -> Dict:
    """Load experiment results from JSON."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def _make_serializable(obj):
    """Recursively convert numpy types to native Python types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_make_serializable(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(_make_serializable(v) for v in obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    else:
        return str(obj)


# ============================================================
# Summary Table
# ============================================================

def build_summary_table(
    all_seed_results: List[Dict],
    metric_keys: List[str] = ("ece", "mce", "brier", "accuracy"),
) -> str:
    """
    Build a formatted summary table from results across seeds.

    Args:
        all_seed_results: List of per-seed result dicts.
        metric_keys: Which metrics to include.

    Returns:
        Formatted string table.
    """
    # Aggregate across seeds
    model_names = sorted(set(
        r["model_name"]
        for seed_results in all_seed_results
        for r in seed_results["metrics"]
    ))

    rows = []
    header = f"{'Model':<25}" + "".join(
        f"{m.upper():>10}  " for m in metric_keys
    )
    rows.append(header)
    rows.append("-" * len(header))

    for name in model_names:
        metric_values = {}
        for mk in metric_keys:
            vals = []
            for seed_results in all_seed_results:
                for r in seed_results["metrics"]:
                    if r["model_name"] == name:
                        vals.append(r["metrics"][mk])
            if vals:
                mean = np.mean(vals)
                std = np.std(vals, ddof=1)
                metric_values[mk] = f"{mean:.4f}±{std:.4f}"
            else:
                metric_values[mk] = "N/A"

        row = f"{name:<25}" + "".join(
            f"{metric_values[mk]:>12}  " for mk in metric_keys
        )
        rows.append(row)

    return "\n".join(rows)
