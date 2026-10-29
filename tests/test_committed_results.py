import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def load_results():
    paths = sorted(RESULTS.glob("*.json"))
    assert paths, "no committed result files found"
    return [(path, json.loads(path.read_text(encoding="utf-8"))) for path in paths]


def test_result_files_have_reproducibility_metadata():
    for path, payload in load_results():
        assert isinstance(payload.get("timestamp"), str), path
        config = payload.get("config")
        assert isinstance(config, dict), path
        assert isinstance(payload.get("dataset"), str), path
        assert isinstance(config.get("n_seeds"), int) and config["n_seeds"] > 0, path
        assert isinstance(config.get("model_variants"), list) and config["model_variants"], path


def test_result_aggregates_are_finite_probabilities():
    bounded_metrics = {"ece", "mce", "brier", "accuracy"}
    for path, payload in load_results():
        aggregated = payload.get("aggregated")
        assert isinstance(aggregated, dict) and aggregated, path
        for model, metrics in aggregated.items():
            assert isinstance(metrics, dict) and metrics, (path, model)
            for name in bounded_metrics:
                mean = metrics.get(name + "_mean")
                std = metrics.get(name + "_std")
                assert isinstance(mean, (int, float)) and 0 <= mean <= 1, (path, model, name)
                assert isinstance(std, (int, float)) and std >= 0, (path, model, name)


def test_readme_pilot_table_matches_named_result_file():
    payload = json.loads(
        (RESULTS / "newsgroups_20260626_232910.json").read_text(encoding="utf-8")
    )
    expected = {
        "RF-Tabular": (0.074, 0.351, 0.273, 0.796),
        "RF-Hybrid": (0.087, 0.334, 0.280, 0.790),
        "RF-Embed": (0.007, 0.008, 0.750, 0.247),
        "MLP-Embed": (0.006, 0.006, 0.750, 0.247),
    }
    aggregated = payload["aggregated"]
    for label, expected_metrics in expected.items():
        metrics = aggregated[label]
        actual = tuple(round(metrics[name + "_mean"], 3) for name in
                       ("ece", "mce", "brier", "accuracy"))
        assert actual == expected_metrics, label
