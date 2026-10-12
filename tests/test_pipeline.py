"""End-to-end: audit, two short experiments and the report on synthetic data."""

import importlib.util
from pathlib import Path

from cxrshortcut.utils import load_json

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_full_pipeline(synth_root, tmp_path):
    results = tmp_path / "results"
    data = str(synth_root)

    audit = load_script("audit_dataset").main(["--data-root", data, "--out", str(results / "audit"), "--workers", "2"])
    assert audit["duplicates"]["exact_groups_spanning_train_and_test"] == 1

    run = load_script("run_experiment").main
    common = ["--data-root", data, "--out-root", str(results), "--img-size", "48", "--batch-size", "16",
              "--n-boot", "50", "--gradcam-per-class", "3", "--verbose", "0"]
    normal = run(common + ["--model", "small_cnn", "--epochs", "2", "--eval-occluded"])
    hidden = run(common + ["--model", "small_cnn", "--name", "small_cnn_occluded", "--epochs", "1", "--occlude"])

    for exp in (normal, hidden):
        t = exp["test"]
        assert t["n"] == 25 and len(t["confusion_matrix"]) == 3
        assert {"macro_auc", "balanced_accuracy", "COVID19_sensitivity"} <= set(t["ci95"])
    assert "test_lungs_occluded" in normal and "test_lungs_occluded" not in hidden
    assert hidden["trained_on_occluded"] is True

    out = results / "small_cnn"
    for f in ("model.keras", "history.csv", "splits.csv", "predictions_test.csv",
              "predictions_test_lungs_occluded.csv", "gradcam_examples.png", "metrics.json", "run_info.json"):
        assert (out / f).exists(), f
    assert load_json(out / "metrics.json")["history"]["epochs_run"] == 2

    summary = load_script("make_report").main(["--results", str(results)]).read_text()
    for heading in ("Audit 1", "Audit 2", "Models on the test set", "lungs blacked out", "Grad-CAM"):
        assert heading in summary
