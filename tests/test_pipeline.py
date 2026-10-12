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

    # Re-running with --reuse-model evaluates the saved model without training again
    history_before = (out / "history.csv").read_text()
    again = run(common + ["--model", "small_cnn", "--epochs", "5", "--reuse-model"])
    assert (out / "history.csv").read_text() == history_before
    assert again["history"]["epochs_run"] == 2


def test_odd_test_files_do_not_crash_the_run(synth_root, tmp_path):
    """A BMP in the test set is read; a file TF cannot decode is left out and reported."""
    import shutil

    import numpy as np
    from PIL import Image

    data = tmp_path / "Data"
    shutil.copytree(synth_root, data)
    a = (np.random.default_rng(1).random((64, 64)) * 255).astype(np.uint8)
    Image.fromarray(a).save(data / "test" / "NORMAL" / "gray8.bmp")
    Image.fromarray(a).convert("1").save(data / "test" / "NORMAL" / "one_bit.bmp")

    res = load_script("run_experiment").main([
        "--data-root", str(data), "--out-root", str(tmp_path / "results"), "--model", "small_cnn",
        "--img-size", "48", "--batch-size", "16", "--epochs", "1", "--n-boot", "20",
        "--gradcam-per-class", "2", "--verbose", "0",
    ])
    assert [p.endswith("one_bit.bmp") for p in res["excluded_unreadable_files"]] == [True]
    assert res["test"]["n"] == 26  # 25 synthetic test images + the readable BMP
    assert (tmp_path / "results" / "small_cnn" / "unreadable_files.txt").exists()
