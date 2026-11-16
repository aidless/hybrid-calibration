"""Leakage diagnostics that do not download a language model.

`make check` imports this module. It must stay free of torch, transformers,
and network calls. The diagnostic runner (scripts/run_leakage_diagnostic.py)
is the only entry point that reads the 20 Newsgroups corpus.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from calibration import compute_brier, compute_ece
from data_loader import NEWSGROUP_CATEGORIES, extract_tabular_features

# Frozen protocol id. Change the id if any constant below changes.
PROTOCOL_ID = "leakdiag-2026-11-16"
SEEDS = [0, 1, 2, 3, 4]
TEST_SIZE = 0.2
VAL_SIZE = 0.1
RF_PARAMS = {
    "n_estimators": 100,
    "max_depth": 10,
    "random_state": 42,
    "n_jobs": 1,
}
TFIDF_PARAMS = {
    "max_features": 2000,
    "ngram_range": (1, 2),
    "min_df": 2,
}
HASH_PARAMS = {
    "n_features": 256,
    "ngram_range": (3, 5),
    "analyzer": "char_wb",
    "alternate_sign": False,
    "norm": "l2",
}
N_BINS = 10
N_BINS_SWEEP = [5, 10, 15, 20]
DUP_INJECT_FRACTION = 0.25
SHUFFLE_OFFSET = 1000
ZERO_DIM = 32

# Aggressive, label-independent cue list. "car" / "gun" / "windows" also match
# ordinary English, so an accuracy drop on the ablated arm is not a pure
# estimate of label leakage.
CUE_TOKENS = [
    "graphics",
    "windows",
    "car",
    "auto",
    "automobile",
    "motorcycle",
    "bike",
    "crypt",
    "encryption",
    "crypto",
    "cryptography",
    "electronics",
    "circuit",
    "gun",
    "guns",
    "israel",
    "arab",
    "mideast",
    "palestinian",
]

# Size-complement groups. Inferred: one model family, and the three files'
# row counts sum to the sample size encoded by those counts. The result JSONs
# do not store the cache filename, so this is not an id match.
INFERRED_SPLIT_GROUPS = [
    {
        "id": "bge_n400",
        "sum_n": 400,
        "train": "newsgroups_train_BAAI_bge_small_zh_v1.5_d08ff15b_279.npy",
        "val": "newsgroups_val_BAAI_bge_small_zh_v1.5_04d24594_40.npy",
        "test": "newsgroups_test_BAAI_bge_small_zh_v1.5_39bec81c_81.npy",
    },
    {
        "id": "bge_n500",
        "sum_n": 500,
        "train": "newsgroups_train_BAAI_bge_small_zh_v1.5_8324f553_349.npy",
        "val": "newsgroups_val_BAAI_bge_small_zh_v1.5_9d1be2f7_50.npy",
        "test": "newsgroups_test_BAAI_bge_small_zh_v1.5_ddf7744e_101.npy",
    },
    {
        "id": "bge_n996",
        "sum_n": 996,
        "train": "newsgroups_train_BAAI_bge_small_zh_v1.5_8aa6da46_697.npy",
        "val": "newsgroups_val_BAAI_bge_small_zh_v1.5_eecaaf2e_99.npy",
        "test": "newsgroups_test_BAAI_bge_small_zh_v1.5_b9167976_200.npy",
    },
    {
        "id": "minilm_n800",
        "sum_n": 800,
        "train": "newsgroups_train_sentence_transformers_all_MiniLM_L6_v2_1331ad30_559.npy",
        "val": "newsgroups_val_sentence_transformers_all_MiniLM_L6_v2_186b7141_80.npy",
        "test": "newsgroups_test_sentence_transformers_all_MiniLM_L6_v2_a49eb7f8_161.npy",
    },
]

# Cross-file pair that shares vectors but is NOT one split (different n).
CROSS_N_PAIR = (
    "newsgroups_test_BAAI_bge_small_zh_v1.5_39bec81c_81.npy",
    "newsgroups_train_BAAI_bge_small_zh_v1.5_8324f553_349.npy",
)


def row_keys(matrix: np.ndarray) -> list[bytes]:
    arr = np.ascontiguousarray(matrix)
    return [row.tobytes() for row in arr]


def overlap_rows(left: np.ndarray, right: np.ndarray) -> int:
    """How many rows of `left` have an exact vector match in `right`."""
    right_keys = set(row_keys(right))
    return sum(1 for key in row_keys(left) if key in right_keys)


def within_file_dups(matrix: np.ndarray) -> dict:
    counts = Counter(row_keys(matrix))
    return {
        "n": int(matrix.shape[0]),
        "unique": int(len(counts)),
        "extra_copies": int(sum(v - 1 for v in counts.values() if v > 1)),
        "max_multiplicity": int(max(counts.values()) if counts else 0),
    }


def audit_embedding_dir(directory: Path) -> dict:
    directory = Path(directory)
    groups = []
    for spec in INFERRED_SPLIT_GROUPS:
        train = np.load(directory / spec["train"])
        val = np.load(directory / spec["val"])
        test = np.load(directory / spec["test"])
        summed = train.shape[0] + val.shape[0] + test.shape[0]
        groups.append({
            "id": spec["id"],
            "expected_sum": spec["sum_n"],
            "sum_n": int(summed),
            "train_file": spec["train"],
            "val_file": spec["val"],
            "test_file": spec["test"],
            "train": within_file_dups(train),
            "val": within_file_dups(val),
            "test": within_file_dups(test),
            "test_rows_in_train": int(overlap_rows(test, train)),
            "test_rows_in_val": int(overlap_rows(test, val)),
            "test_n": int(test.shape[0]),
            "dim": int(test.shape[1]),
        })
    left = np.load(directory / CROSS_N_PAIR[0])
    right = np.load(directory / CROSS_N_PAIR[1])
    cross = {
        "left": CROSS_N_PAIR[0],
        "right": CROSS_N_PAIR[1],
        "left_rows_in_right": int(overlap_rows(left, right)),
        "left_n": int(left.shape[0]),
        "note": "Different sample sizes. Shared vectors are a shared corpus, not one train/test split.",
    }
    return {"groups": groups, "cross_n": cross}


def split_three(texts, labels, seed: int):
    """Match experiment.py: val is VAL_SIZE, test is TEST_SIZE, both stratified."""
    labels = np.asarray(labels)
    texts_tr, texts_temp, y_tr, y_temp = train_test_split(
        list(texts), labels,
        test_size=TEST_SIZE + VAL_SIZE,
        random_state=seed,
        stratify=labels,
    )
    val_ratio = VAL_SIZE / (TEST_SIZE + VAL_SIZE)
    texts_va, texts_te, y_va, y_te = train_test_split(
        texts_temp, y_temp,
        test_size=1 - val_ratio,
        random_state=seed,
        stratify=y_temp,
    )
    return {
        "texts_train": list(texts_tr),
        "texts_val": list(texts_va),
        "texts_test": list(texts_te),
        "y_train": np.asarray(y_tr),
        "y_val": np.asarray(y_va),
        "y_test": np.asarray(y_te),
    }


def scale_train_only(X_train, X_val, X_test):
    scaler = StandardScaler()
    return (
        np.nan_to_num(scaler.fit_transform(X_train), nan=0.0),
        np.nan_to_num(scaler.transform(X_val), nan=0.0),
        np.nan_to_num(scaler.transform(X_test), nan=0.0),
    )


def make_forest():
    return RandomForestClassifier(**RF_PARAMS)


def train_eval(X_train, y_train, X_test, y_test) -> np.ndarray:
    """Fit on train, return probabilities on test. Model RNG is RF_PARAMS."""
    model = make_forest()
    model.fit(X_train, y_train)
    return np.asarray(model.predict_proba(X_test), dtype=np.float64)


def text_in_train_mask(train_texts, test_texts) -> np.ndarray:
    pool = set(train_texts)
    return np.array([t in pool for t in test_texts], dtype=bool)


def shuffle_labels(labels, seed: int) -> np.ndarray:
    labels = np.asarray(labels).copy()
    perm = np.random.RandomState(seed + SHUFFLE_OFFSET).permutation(len(labels))
    return labels[perm]


def cue_pattern():
    return re.compile(r"\b(" + "|".join(map(re.escape, CUE_TOKENS)) + r")\b", re.I)


def strip_cues(texts) -> list[str]:
    pat = cue_pattern()
    return [pat.sub(" ", t) for t in texts]


def own_class_cue_rate(texts, labels, class_names) -> float:
    """Fraction of docs that contain a cue token associated with their own class.

    Association is the subset of CUE_TOKENS that appears in the class name
    or is the usual stem of that name. This is a count, not a model.
    """
    stems = {
        "comp.graphics": ["graphics"],
        "comp.os.ms-windows.misc": ["windows"],
        "rec.autos": ["car", "auto", "automobile"],
        "rec.motorcycles": ["motorcycle", "bike"],
        "sci.crypt": ["crypt", "encryption", "crypto", "cryptography"],
        "sci.electronics": ["electronics", "circuit"],
        "talk.politics.guns": ["gun", "guns"],
        "talk.politics.mideast": ["israel", "arab", "mideast", "palestinian"],
    }
    hit = 0
    for text, lab in zip(texts, labels):
        tokens = stems[class_names[int(lab)]]
        pat = re.compile(r"\b(" + "|".join(map(re.escape, tokens)) + r")\b", re.I)
        if pat.search(text):
            hit += 1
    return float(hit / max(len(texts), 1))


def any_cue_rate(texts) -> float:
    pat = cue_pattern()
    return float(sum(1 for t in texts if pat.search(t)) / max(len(texts), 1))


def hashing_matrix(texts) -> np.ndarray:
    vectorizer = HashingVectorizer(**HASH_PARAMS)
    return vectorizer.transform(texts).astype(np.float64).toarray()


def tfidf_matrices(train_texts, other_lists):
    vectorizer = TfidfVectorizer(**TFIDF_PARAMS)
    X_train = vectorizer.fit_transform(train_texts).astype(np.float64).toarray()
    others = [vectorizer.transform(texts).astype(np.float64).toarray() for texts in other_lists]
    return X_train, others


def tabular_matrices(train_texts, val_texts, test_texts):
    X_tr, _ = extract_tabular_features(train_texts)
    X_va, _ = extract_tabular_features(val_texts)
    X_te, _ = extract_tabular_features(test_texts)
    return scale_train_only(X_tr, X_va, X_te)


def isotonic_apply(proba_fit, y_fit, proba_apply) -> np.ndarray:
    """Per-class isotonic, the same shape as models.CalibratedModel isotonic."""
    out = np.zeros_like(proba_apply, dtype=np.float64)
    for k in range(proba_fit.shape[1]):
        iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        iso.fit(proba_fit[:, k], (np.asarray(y_fit) == k).astype(np.float64))
        out[:, k] = iso.predict(proba_apply[:, k])
    out = np.clip(out, 0.0, None)
    totals = out.sum(axis=1, keepdims=True)
    totals[totals == 0] = 1.0
    return out / totals


def inject_train_rows(X_test, y_test, X_train, y_train, seed: int):
    rng = np.random.RandomState(seed + 17)
    n = len(y_test)
    k = int(round(DUP_INJECT_FRACTION * n))
    take = rng.choice(len(y_train), size=k, replace=False)
    X2 = np.array(X_test, copy=True)
    y2 = np.array(y_test, copy=True)
    X2[:k] = X_train[take]
    y2[:k] = y_train[take]
    return X2, y2, int(k)


def pack_probs(y_true, y_prob) -> dict:
    return {
        "y_true": [int(v) for v in np.asarray(y_true).tolist()],
        "y_prob": np.asarray(y_prob, dtype=np.float64).tolist(),
    }


def scores_from_pack(pack, n_bins: int = N_BINS) -> dict:
    y = np.asarray(pack["y_true"], dtype=np.int64)
    proba = np.asarray(pack["y_prob"], dtype=np.float64)
    pred = np.argmax(proba, axis=1)
    return {
        "ece": float(compute_ece(y, proba, n_bins=n_bins)),
        "brier": float(compute_brier(y, proba)),
        "accuracy": float(np.mean(pred == y)),
        "n": int(len(y)),
        "n_classes": int(proba.shape[1]),
    }


def mean_std(values):
    arr = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    return mean, std


def fmt_ms(mean, std) -> str:
    return f"{mean:.3f}±{std:.3f}"


def arm_summary(seed_rows, arm: str, n_bins: int = N_BINS) -> dict:
    ece, brier, acc = [], [], []
    n_classes = None
    n_test = None
    for row in seed_rows:
        sc = scores_from_pack(row["arms"][arm], n_bins=n_bins)
        ece.append(sc["ece"])
        brier.append(sc["brier"])
        acc.append(sc["accuracy"])
        n_classes = sc["n_classes"]
        n_test = sc["n"]
    em, es = mean_std(ece)
    bm, bs = mean_std(brier)
    am, as_ = mean_std(acc)
    return {
        "ece": fmt_ms(em, es),
        "brier": fmt_ms(bm, bs),
        "accuracy": fmt_ms(am, as_),
        "ece_mean": em,
        "brier_mean": bm,
        "accuracy_mean": am,
        "accuracy_std": as_,
        "n_classes": n_classes,
        "n_test": n_test,
        "n_seeds": len(seed_rows),
    }
