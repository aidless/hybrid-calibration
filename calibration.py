"""
Probability Calibration Evaluation Suite.

Metrics:
  - ECE (Expected Calibration Error) — equal-width binning
  - MCE (Maximum Calibration Error)
  - Brier Score
  - Reliability diagram data
  - Per-class ECE (for multiclass)

All metrics work for both binary and multiclass settings.
"""

import numpy as np
from typing import Dict, Tuple, Optional
from sklearn.metrics import brier_score_loss


# ============================================================
# Core Metrics
# ============================================================

def compute_ece(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    class_idx: Optional[int] = None,
) -> float:
    """
    Expected Calibration Error (ECE) with equal-width binning.

    Args:
        y_true: True labels (int, shape (n,)).
        y_prob: Predicted probabilities. For binary: (n,) or (n, 2).
                For multiclass: (n, n_classes).
        n_bins: Number of equal-width bins.
        class_idx: For multiclass, compute ECE for a specific class.
                   If None and binary, uses positive class (class 1).
                   If None and multiclass, returns macro-averaged ECE.

    Returns:
        ECE value (lower = better calibrated).
    """
    # Normalize to (n,) array of confidence scores
    confidences, accuracies = _get_confidence_accuracy(y_true, y_prob, class_idx)

    if class_idx is not None:
        # Single-class ECE
        return _ece_single(confidences, accuracies, n_bins)
    else:
        # Average over all classes
        n_classes = y_prob.shape[1]
        ece_total = 0.0
        for c in range(n_classes):
            conf_c = y_prob[:, c]
            # Binary accuracy for class c
            acc_c = (y_true == c).astype(np.float64)
            ece_total += _ece_single(conf_c, acc_c, n_bins)
        return ece_total / n_classes


def compute_mce(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Maximum Calibration Error (MCE).
    """
    confidences, accuracies = _get_confidence_accuracy(y_true, y_prob, None)
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    max_error = 0.0

    for i in range(n_bins):
        mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        if np.sum(mask) > 0:
            bin_conf = np.mean(confidences[mask])
            bin_acc = np.mean(accuracies[mask])
            error = abs(bin_acc - bin_conf)
            max_error = max(max_error, error)

    return float(max_error)


def compute_brier(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Multi-class Brier score.

    Args:
        y_true: (n,) int labels.
        y_prob: (n, n_classes) probability matrix.
    """
    n_classes = y_prob.shape[1]
    y_onehot = np.eye(n_classes, dtype=np.float64)[y_true]
    return float(np.mean(np.sum((y_prob - y_onehot) ** 2, axis=1)))


def compute_jsd_calibration(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Jensen-Shannon Divergence between predicted and empirical distributions
    across calibration bins. Bounded in [0, ln 2].

    Standard approach: build histograms of predicted confidence (how many
    predictions fall into each confidence bin) and observed accuracy
    (what fraction are correct in each bin), then compute JSD between
    these two discrete distributions.

    Args:
        y_true: (n,) int labels.
        y_prob: (n, n_classes) probability matrix.
        n_bins: Number of calibration bins.

    Returns:
        JSD value (lower = better calibrated).
    """
    from scipy.spatial.distance import jensenshannon

    # Get per-sample confidence and accuracy
    confidences = np.max(y_prob, axis=1)
    correct = (np.argmax(y_prob, axis=1) == y_true).astype(np.float64)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)

    # Build histograms: predicted mass per bin vs observed accuracy per bin
    pred_dist = np.zeros(n_bins)
    obs_dist = np.zeros(n_bins)

    for i in range(n_bins):
        mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        bin_count = np.sum(mask)
        if bin_count > 0:
            pred_dist[i] = bin_count / len(confidences)
            obs_dist[i] = np.mean(correct[mask])
        # else: both remain 0 for empty bins

    # Normalize observed distribution to sum to 1 (same as pred mass)
    obs_sum = np.sum(obs_dist)
    if obs_sum > 0:
        obs_dist = obs_dist / obs_sum

    # JSD between predicted mass distribution and observed accuracy distribution
    return float(jensenshannon(pred_dist, obs_dist) ** 2)


# ============================================================
# Reliability Diagram Data
# ============================================================

def reliability_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> Dict[str, np.ndarray]:
    """
    Compute reliability diagram data for the positive class.

    Returns:
        Dict with keys:
          - bin_conf: mean predicted probability per bin
          - bin_acc: observed accuracy per bin
          - bin_count: number of samples per bin
          - bin_edges: bin boundary edges (n_bins + 1)
    """
    confidences, accuracies = _get_confidence_accuracy(y_true, y_prob, None)
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)

    bin_conf = np.zeros(n_bins)
    bin_acc = np.zeros(n_bins)
    bin_count = np.zeros(n_bins, dtype=np.int64)

    for i in range(n_bins):
        mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        bin_count[i] = np.sum(mask)
        if bin_count[i] > 0:
            bin_conf[i] = np.mean(confidences[mask])
            bin_acc[i] = np.mean(accuracies[mask])
        else:
            bin_conf[i] = (bin_boundaries[i] + bin_boundaries[i + 1]) / 2
            bin_acc[i] = np.nan

    return {
        "bin_conf": bin_conf,
        "bin_acc": bin_acc,
        "bin_count": bin_count,
        "bin_edges": bin_boundaries,
    }


# ============================================================
# Full Evaluation
# ============================================================

def evaluate_calibration(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    model_name: str = "",
) -> Dict[str, object]:
    """
    Full calibration evaluation for one model on one dataset split.

    Args:
        y_true: (n,) int labels.
        y_prob: (n, n_classes) probabilities.
        n_bins: ECE bins.

    Returns:
        Dict of metrics.
    """
    results = {
        "model": model_name,
        "ece": compute_ece(y_true, y_prob, n_bins=n_bins),
        "mce": compute_mce(y_true, y_prob, n_bins=n_bins),
        "brier": compute_brier(y_true, y_prob),
        "jsd": compute_jsd_calibration(y_true, y_prob),
        "reliability": reliability_curve(y_true, y_prob, n_bins=n_bins),
        "n_samples": len(y_true),
    }

    # Per-class ECE
    n_classes = y_prob.shape[1]
    per_class_ece = {}
    for c in range(n_classes):
        per_class_ece[f"ece_class_{c}"] = compute_ece(
            y_true, y_prob, n_bins=n_bins, class_idx=c
        )
    results["per_class_ece"] = per_class_ece

    # Accuracy
    results["accuracy"] = float(np.mean(y_true == np.argmax(y_prob, axis=1)))

    # AUC (binary only)
    if n_classes == 2:
        from sklearn.metrics import roc_auc_score
        try:
            results["auc"] = float(roc_auc_score(y_true, y_prob[:, 1]))
        except ValueError:
            results["auc"] = np.nan

    return results


# ============================================================
# Helpers
# ============================================================

def _get_confidence_accuracy(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_idx: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract confidence scores and binary accuracy for calibration evaluation.

    For binary classification with class_idx=None:
      confidence = P(y=1), accuracy = 1 if y_true == 1 else 0

    For multiclass with class_idx=None:
      confidence = max probability per sample
      accuracy = 1 if predicted class matches true label, else 0

    For class_idx specified:
      confidence = P(y=class_idx)
      accuracy = 1 if y_true == class_idx else 0
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_prob = np.asarray(y_prob, dtype=np.float64)

    if y_prob.ndim == 1:
        y_prob = np.column_stack([1 - y_prob, y_prob])

    n_classes = y_prob.shape[1]

    if class_idx is not None:
        confidences = y_prob[:, class_idx]
        accuracies = (y_true == class_idx).astype(np.float64)
    elif n_classes == 2:
        # Binary: use positive class
        confidences = y_prob[:, 1]
        accuracies = (y_true == 1).astype(np.float64)
    else:
        # Multiclass: use argmax
        pred_class = np.argmax(y_prob, axis=1)
        confidences = y_prob[np.arange(len(y_prob)), pred_class]
        accuracies = (y_true == pred_class).astype(np.float64)

    return confidences, accuracies


def _ece_single(
    confidences: np.ndarray,
    accuracies: np.ndarray,
    n_bins: int,
) -> float:
    """Compute ECE for a single (confidence, accuracy) pair series."""
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(confidences)

    for i in range(n_bins):
        mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        bin_size = np.sum(mask)
        if bin_size > 0:
            bin_conf = np.mean(confidences[mask])
            bin_acc = np.mean(accuracies[mask])
            ece += (bin_size / n) * abs(bin_acc - bin_conf)

    return float(ece)
