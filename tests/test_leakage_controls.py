"""Mechanism checks for the leakage controls. They do not use the historical JSON."""

import numpy as np
from sklearn.model_selection import train_test_split

import leakage as L


def test_zero_features_four_class_uniform_signature():
    """Constant features: Brier near 1-1/4 and accuracy near 1/4.

    This is the shape of the June 26 embedding rows. It does not say those
    rows were produced by zeros; it says zeros are sufficient to produce it.
    """
    y = np.tile(np.arange(4), 50)
    X = np.zeros((len(y), 4), dtype=np.float64)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=0, stratify=y
    )
    scores = L.scores_from_pack(L.pack_probs(y_te, L.train_eval(X_tr, y_tr, X_te, y_te)))
    assert abs(scores["brier"] - 0.75) < 0.02
    assert abs(scores["accuracy"] - 0.25) < 0.05
    assert scores["ece"] < 0.05


def test_shuffled_labels_collapse_when_the_feature_is_the_label():
    y = np.tile(np.arange(6), 40)
    X = np.eye(6, dtype=np.float64)[y]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=0, stratify=y
    )
    true_acc = L.scores_from_pack(
        L.pack_probs(y_te, L.train_eval(X_tr, y_tr, X_te, y_te))
    )["accuracy"]
    assert true_acc > 0.95

    y_shuf = L.shuffle_labels(y, seed=0)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y_shuf, test_size=0.25, random_state=1, stratify=y_shuf
    )
    shuf_acc = L.scores_from_pack(
        L.pack_probs(y_te, L.train_eval(X_tr, y_tr, X_te, y_te))
    )["accuracy"]
    assert shuf_acc < 0.40


def test_scaler_is_fit_on_train_only():
    rng = np.random.RandomState(0)
    X_tr = rng.normal(loc=5.0, size=(30, 2))
    X_va = rng.normal(loc=5.0, size=(8, 2))
    X_te = rng.normal(loc=-5.0, size=(12, 2))
    tr, _, te = L.scale_train_only(X_tr, X_va, X_te)
    assert abs(float(tr.mean())) < 0.4
    assert float(te.mean()) < -1.0
