#!/usr/bin/env python3
"""Run the frozen leakage diagnostic and write prediction files.

This script may read sklearn's 20 Newsgroups cache (network on a cold cache).
`make check` does not call it. Raw texts are not written; prediction vectors,
overlap masks, and corpus counts are.

Protocol id: leakdiag-2026-11-16. Do not edit the constants in leakage.py
without changing that id.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import leakage as L
from data_loader import NEWSGROUP_CATEGORIES, _load_synthetic, _stratified_subset


def _corpus_counts(texts):
    empty = sum(1 for t in texts if not str(t).strip())
    unique = len(set(texts))
    return {
        "n": len(texts),
        "unique_texts": unique,
        "extra_copies": len(texts) - unique,
        "empty": empty,
    }


def build_newsgroups(texts, labels, class_names):
    if len(class_names) != len(set(int(v) for v in labels)):
        raise SystemExit("class name count does not match empirical labels")
    seed_rows = []
    probe = None
    for seed in L.SEEDS:
        split = L.split_three(texts, labels, seed)
        split["seed"] = seed
        y_tr, y_te = split["y_train"], split["y_test"]
        texts_tr = split["texts_train"]
        texts_va = split["texts_val"]
        texts_te = split["texts_test"]
        mask = L.text_in_train_mask(texts_tr, texts_te)

        tab_tr, _, tab_te = L.tabular_matrices(texts_tr, texts_va, texts_te)
        X_tr, (X_va, X_te) = L.tfidf_matrices(texts_tr, [texts_va, texts_te])
        forest = L.make_forest()
        forest.fit(X_tr, y_tr)
        proba_val = np.asarray(forest.predict_proba(X_va), dtype=np.float64)
        proba_te = np.asarray(forest.predict_proba(X_te), dtype=np.float64)

        stripped_tr = L.strip_cues(texts_tr)
        stripped_va = L.strip_cues(texts_va)
        stripped_te = L.strip_cues(texts_te)
        Xs_tr, (_, Xs_te) = L.tfidf_matrices(stripped_tr, [stripped_va, stripped_te])

        y_shuf = L.shuffle_labels(labels, seed)
        shuf = L.split_three(texts, y_shuf, seed)
        Xs_shuf_tr, (_, Xs_shuf_te) = L.tfidf_matrices(
            shuf["texts_train"], [shuf["texts_val"], shuf["texts_test"]]
        )

        zeros_tr = np.zeros((len(y_tr), L.ZERO_DIM))
        zeros_te = np.zeros((len(y_te), L.ZERO_DIM))
        X_inj, y_inj, k = L.inject_train_rows(X_te, y_te, X_tr, y_tr, seed)
        inj_pack = L.pack_probs(y_inj, forest.predict_proba(X_inj))
        inj_pack["n_injected"] = k

        arms = {
            "tabular": L.pack_probs(y_te, L.train_eval(tab_tr, y_tr, tab_te, y_te)),
            "tfidf": L.pack_probs(y_te, proba_te),
            "tfidf_cue_ablated": L.pack_probs(
                y_te, L.train_eval(Xs_tr, y_tr, Xs_te, y_te)
            ),
            "shuffled_tfidf": L.pack_probs(
                shuf["y_test"],
                L.train_eval(Xs_shuf_tr, shuf["y_train"], Xs_shuf_te, shuf["y_test"]),
            ),
            "zero_embed": L.pack_probs(
                y_te, L.train_eval(zeros_tr, y_tr, zeros_te, y_te)
            ),
            "dup_inject_tfidf": inj_pack,
        }
        seed_rows.append({
            "seed": int(seed),
            "n_train": len(texts_tr),
            "n_val": len(texts_va),
            "n_test": len(texts_te),
            "n_test_exact_text_in_train": int(mask.sum()),
            "n_test_empty": int(sum(1 for t in texts_te if not str(t).strip())),
            "test_in_train": [int(v) for v in mask.tolist()],
            "arms": arms,
        })
        if seed == L.SEEDS[0]:
            probe = {
                "seed": int(seed),
                "base_arm": "tfidf",
                "fit_on_val": L.pack_probs(
                    y_te, L.isotonic_apply(proba_val, split["y_val"], proba_te)
                ),
                "fit_on_test": L.pack_probs(
                    y_te, L.isotonic_apply(proba_te, y_te, proba_te)
                ),
                "note": (
                    "fit_on_test is a contamination control: the calibrator sees "
                    "the same labels that are then scored. It is not a valid ECE."
                ),
            }
        print(f"  newsgroups seed {seed} done", flush=True)
    return seed_rows, probe


def build_synthetic(texts, labels, class_names):
    seed_rows = []
    for seed in L.SEEDS:
        split = L.split_three(texts, labels, seed)
        y_tr, y_te = split["y_train"], split["y_test"]
        texts_tr, texts_va, texts_te = (
            split["texts_train"], split["texts_val"], split["texts_test"]
        )
        mask = L.text_in_train_mask(texts_tr, texts_te)
        tab_tr, _, tab_te = L.tabular_matrices(texts_tr, texts_va, texts_te)
        hash_tr = L.hashing_matrix(texts_tr)
        hash_te = L.hashing_matrix(texts_te)

        leaked_tr = [
            f"{t} style={class_names[int(y)]}" for t, y in zip(texts_tr, y_tr)
        ]
        leaked_te = [
            f"{t} style={class_names[int(y)]}" for t, y in zip(texts_te, y_te)
        ]
        y_shuf = L.shuffle_labels(labels, seed)
        shuf = L.split_three(texts, y_shuf, seed)
        hash_shuf_tr = L.hashing_matrix(shuf["texts_train"])
        hash_shuf_te = L.hashing_matrix(shuf["texts_test"])
        zeros_tr = np.zeros((len(y_tr), L.ZERO_DIM))
        zeros_te = np.zeros((len(y_te), L.ZERO_DIM))
        arms = {
            "tabular": L.pack_probs(y_te, L.train_eval(tab_tr, y_tr, tab_te, y_te)),
            "hash": L.pack_probs(y_te, L.train_eval(hash_tr, y_tr, hash_te, y_te)),
            "hash_label_in_text": L.pack_probs(
                y_te,
                L.train_eval(
                    L.hashing_matrix(leaked_tr), y_tr, L.hashing_matrix(leaked_te), y_te
                ),
            ),
            "shuffled_hash": L.pack_probs(
                shuf["y_test"],
                L.train_eval(hash_shuf_tr, shuf["y_train"], hash_shuf_te, shuf["y_test"]),
            ),
            "zero_embed": L.pack_probs(
                y_te, L.train_eval(zeros_tr, y_tr, zeros_te, y_te)
            ),
        }
        seed_rows.append({
            "seed": int(seed),
            "n_train": len(texts_tr),
            "n_val": len(texts_va),
            "n_test": len(texts_te),
            "n_test_exact_text_in_train": int(mask.sum()),
            "n_test_empty": 0,
            "test_in_train": [int(v) for v in mask.tolist()],
            "arms": arms,
        })
        print(f"  synthetic seed {seed} done", flush=True)
    return seed_rows


def main():
    import sklearn

    out_dir = ROOT / "results" / "leakage"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("loading 20 newsgroups (sklearn cache or download)")
    from sklearn.datasets import fetch_20newsgroups

    raw = fetch_20newsgroups(
        subset="all",
        categories=list(NEWSGROUP_CATEGORIES),
        remove=("headers", "footers", "quotes"),
        random_state=42,
    )
    full_texts = list(raw.data)
    full_labels = np.asarray(raw.target)
    class_names = [raw.target_names[i] for i in range(len(raw.target_names))]
    if class_names != list(NEWSGROUP_CATEGORIES):
        raise SystemExit(f"class name order mismatch: {class_names}")
    subset_texts, subset_labels = _stratified_subset(full_texts, full_labels, 800)
    print(f"  full={len(full_texts)} subset={len(subset_texts)} classes={len(set(subset_labels))}")

    ng_seeds, probe = build_newsgroups(subset_texts, subset_labels, class_names)
    ng = {
        "protocol_id": L.PROTOCOL_ID,
        "dataset": "newsgroups",
        "data_source": "real",
        "loader": "sklearn.datasets.fetch_20newsgroups",
        "remove": ["headers", "footers", "quotes"],
        "categories": list(NEWSGROUP_CATEGORIES),
        "empirical_num_classes": int(len(set(int(v) for v in subset_labels))),
        "subset_rule": "data_loader._stratified_subset with RandomState(42), n=800",
        "full_corpus": _corpus_counts(full_texts),
        "subset_corpus": _corpus_counts(subset_texts),
        "own_class_cue_rate_subset": L.own_class_cue_rate(
            subset_texts, subset_labels, class_names
        ),
        "any_cue_rate_subset": L.any_cue_rate(subset_texts),
        "cue_tokens": list(L.CUE_TOKENS),
        "cue_ablation": (
            "Label-independent deletion of CUE_TOKENS with word boundaries "
            "before the TF-IDF fit. Common words (car, gun, windows) are in "
            "the list, so this is an aggressive scrub, not a pure causal effect."
        ),
        "sklearn": sklearn.__version__,
        "numpy": np.__version__,
        "seeds": ng_seeds,
        "isotonic_probe": probe,
        "model": {
            "type": "RandomForestClassifier",
            **L.RF_PARAMS,
            "note": "random_state is fixed. The seed axis is the split only.",
        },
        "tfidf": {
            "max_features": L.TFIDF_PARAMS["max_features"],
            "ngram_range": list(L.TFIDF_PARAMS["ngram_range"]),
            "min_df": L.TFIDF_PARAMS["min_df"],
            "fit": "training split only",
        },
        "not_this_file": [
            "Not GPT-2 and not the committed MiniLM/BGE matrices.",
            "Not a re-estimate of results/newsgroups_*.json.",
            "Not a Wilcoxon or Bonferroni result.",
            "The 14 x 10 x 4 design is not run here.",
        ],
    }
    ng_path = out_dir / "newsgroups_leakdiag.json"
    ng_path.write_text(json.dumps(ng), encoding="utf-8")
    print(f"wrote {ng_path} ({ng_path.stat().st_size} bytes)")

    print("synthetic fallback from data_loader._load_synthetic")
    syn_texts, syn_labels, syn_names, syn_source = _load_synthetic(600)
    if syn_source != "synthetic":
        raise SystemExit(syn_source)
    syn_seeds = build_synthetic(syn_texts, syn_labels, syn_names)
    syn = {
        "protocol_id": L.PROTOCOL_ID,
        "dataset": "synthetic_style",
        "data_source": "synthetic",
        "loader": "data_loader._load_synthetic(600)",
        "empirical_num_classes": int(len(set(int(v) for v in syn_labels))),
        "class_names": list(syn_names),
        "n": len(syn_texts),
        "label_definition": (
            "The class is the style template family that generated the text. "
            "The label is a function of the wording. hash_label_in_text also "
            "appends ' style=<class name>'."
        ),
        "sklearn": sklearn.__version__,
        "seeds": syn_seeds,
        "not_this_file": [
            "Current generator has 6 styles. Historical result files have 4 "
            "probability columns, so this generator is not what those files scored.",
        ],
    }
    syn_path = out_dir / "synthetic_leakdiag.json"
    syn_path.write_text(json.dumps(syn), encoding="utf-8")
    print(f"wrote {syn_path} ({syn_path.stat().st_size} bytes)")

    def show(title, payload, arms):
        print(f"\n== {title} ==")
        for arm in arms:
            summary = L.arm_summary(payload["seeds"], arm)
            print(
                f"  {arm:<22} ECE {summary['ece']}  Brier {summary['brier']}  "
                f"Acc {summary['accuracy']}  K={summary['n_classes']} n_test={summary['n_test']}"
            )

    show("newsgroups", ng, [
        "tabular", "tfidf", "tfidf_cue_ablated", "shuffled_tfidf",
        "zero_embed", "dup_inject_tfidf",
    ])
    if probe:
        for key in ("fit_on_val", "fit_on_test"):
            sc = L.scores_from_pack(probe[key])
            print(f"  isotonic {key:<12} ECE {sc['ece']:.3f}  Brier {sc['brier']:.3f}  Acc {sc['accuracy']:.3f}")
    overlaps = [row["n_test_exact_text_in_train"] for row in ng_seeds]
    print("  test rows whose text is in train, by seed:", overlaps)
    print(
        f"  own-class cue rate {ng['own_class_cue_rate_subset']:.3f}  "
        f"any-cue rate {ng['any_cue_rate_subset']:.3f}"
    )
    print("  full corpus", ng["full_corpus"], "subset", ng["subset_corpus"])
    show("synthetic", syn, [
        "tabular", "hash", "hash_label_in_text", "shuffled_hash", "zero_embed",
    ])


if __name__ == "__main__":
    main()
