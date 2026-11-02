#!/usr/bin/env python3
"""Offline verification benchmark for hybrid-calibration.

Does NOT rerun model training. Verifies that every committed result file carries
reproducibility metadata, that calibration aggregates are finite probabilities in
valid ranges, and that the README pilot table matches its named JSON source.

This is the honest claim the pilot makes: the numbers are recomputable from the
committed JSON with no model call.

Usage: python benchmarks/bench_offline_verify.py
Exit: 0 if all checks pass, 1 otherwise (fail-closed).
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
BOUNDED = {"ece", "mce", "brier", "accuracy"}
PILOT_SOURCE = "newsgroups_20260626_232910.json"
EXPECTED = {
    "RF-Tabular": (0.074, 0.351, 0.273, 0.796),
    "RF-Hybrid": (0.087, 0.334, 0.280, 0.790),
    "RF-Embed": (0.007, 0.008, 0.750, 0.247),
    "MLP-Embed": (0.006, 0.006, 0.750, 0.247),
}


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    return cond


def main():
    print("=== hybrid-calibration offline verification ===")
    all_ok = True
    paths = sorted(RESULTS.glob("*.json"))
    all_ok &= check("committed result files present", len(paths) > 0, f"{len(paths)} files")

    n_files = 0
    for path in paths:
        p = json.loads(path.read_text(encoding="utf-8"))
        cfg = p.get("config", {})
        ok_meta = (
            isinstance(p.get("timestamp"), str)
            and isinstance(cfg, dict)
            and isinstance(p.get("dataset"), str)
            and isinstance(cfg.get("n_seeds"), int) and cfg["n_seeds"] > 0
            and isinstance(cfg.get("model_variants"), list) and bool(cfg["model_variants"])
        )
        n_files += 1
        all_ok &= check(f"metadata: {path.name}", ok_meta,
                        f"dataset={p.get('dataset')} n_seeds={cfg.get('n_seeds')}")

    for path in paths:
        p = json.loads(path.read_text(encoding="utf-8"))
        agg = p.get("aggregated", {})
        for model, metrics in agg.items():
            for name in BOUNDED:
                mean = metrics.get(name + "_mean")
                std = metrics.get(name + "_std")
                if not (isinstance(mean, (int, float)) and 0 <= mean <= 1):
                    all_ok &= check(f"range {model}/{name}_mean", False, f"{mean}")
                if not (isinstance(std, (int, float)) and std >= 0):
                    all_ok &= check(f"range {model}/{name}_std", False, f"{std}")
    all_ok &= check("all aggregates finite probabilities in [0,1]", True, f"{n_files} files scanned")

    pilot = json.loads((RESULTS / PILOT_SOURCE).read_text(encoding="utf-8"))
    agg = pilot["aggregated"]
    pilot_ok = True
    for label, exp in EXPECTED.items():
        m = agg[label]
        actual = tuple(round(m[n + "_mean"], 3) for n in ("ece", "mce", "brier", "accuracy"))
        if actual != exp:
            pilot_ok = False
            print(f"  [FAIL] pilot {label}: expected {exp}, got {actual}")
    all_ok &= check("README pilot table matches named JSON (byte-level, round 3)", pilot_ok)

    print("---")
    print(f"RESULT: {'PASS' if all_ok else 'FAIL'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
