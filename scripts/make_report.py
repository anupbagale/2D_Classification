"""Collect results/audit/audit.json and results/*/metrics.json into results/summary.md.

    python scripts/make_report.py --results results
"""

from __future__ import annotations

import argparse
from pathlib import Path

from cxrshortcut.utils import load_json


def _f(x, digits=3) -> str:
    return "–" if x is None else f"{x:.{digits}f}"


def _ci(value, ci) -> str:
    if value is None:
        return "–"
    if not ci or ci[0] is None:
        return _f(value)
    return f"{_f(value)} ({_f(ci[0])}–{_f(ci[1])})"


def _table(header: list[str], rows: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return lines + [""]


def audit_section(audit: dict) -> list[str]:
    out = ["## Dataset", ""]
    counts = audit["counts"]
    splits = list(next(iter(counts.values())).keys())
    out += _table(["Class"] + splits, [[c] + [counts[c][s] for s in splits] for c in counts])

    st = audit["metadata_shortcut_test"]
    m = st["metrics"]
    out += ["## Audit 1: can file metadata alone predict the label?", "",
            "A random forest that sees only image size, file format, colour mode and overall "
            "brightness, never the pixels' content. 0.5 AUC = no information; 1.0 = the label "
            "can be read off the file.", ""]
    out += _table(
        ["", "Macro AUC", "Balanced accuracy", "Majority baseline accuracy"] + [f"AUC {c}" for c in m["classes"]],
        [["Metadata only (test set)", _f(m["macro_auc"]), _f(m["balanced_accuracy"]),
          _f(m["majority_baseline_accuracy"])] + [_f(v["auc_one_vs_rest"]) for v in m["per_class"].values()]],
    )
    out += ["Most informative metadata: " + ", ".join(f"`{k}` ({v:.2f})" for k, v in list(st["feature_importance"].items())[:5]), ""]
    rows = []
    for c, s in st["per_class_summary"].items():
        fmt = ", ".join(f"{k} {v:.0%}" for k, v in s["format_share"].items())
        mode = ", ".join(f"{k} {v:.0%}" for k, v in s["mode_share"].items())
        rows.append([c, s["n"], f"{s['median_width']:.0f}×{s['median_height']:.0f}", s["median_file_kb"],
                     s["mean_intensity"], fmt, mode])
    out += _table(["Class", "Images", "Median size (w×h)", "Median KB", "Mean intensity", "Format", "Colour mode"], rows)

    d = audit["duplicates"]
    out += ["## Audit 2: duplicates between training and test images", ""]
    out += _table(["Check", "Count"], [
        ["Groups of byte-identical files", d["exact_duplicate_groups"]],
        ["…of which span train/val and test", d["exact_groups_spanning_train_and_test"]],
        ["…of which have conflicting labels", d["exact_groups_with_conflicting_labels"]],
        [f"Test images with a near-duplicate in train/val (≤{d['near_duplicate_max_distance_bits']} of "
         f"{d['near_duplicate_hash_bits']} hash bits)",
         f"{d['test_images_with_near_duplicate_in_train']} of {d['test_images_total']}"],
        ["Near-duplicate pairs with different labels", d["near_duplicate_pairs_label_mismatch"]],
    ])
    out += ["Near-duplicates are candidates, not proof; see `audit/near_duplicates_examples.png`.", ""]
    return out


def experiments_section(exps: list[dict]) -> list[str]:
    out = ["## Models on the test set", "",
           "Best epoch chosen on the validation split; the test set was used once, at the end. "
           "Brackets are 95% bootstrap confidence intervals.", ""]
    rows = []
    for e in exps:
        t = e["test"]
        ci = t.get("ci95", {})
        pc = t.get("positive_class")
        pcv = t["per_class"].get(pc, {}) if pc else {}
        rows.append([
            e["name"], "yes" if e["trained_on_occluded"] else "no",
            _f(t["accuracy"]), _f(t["majority_baseline_accuracy"]),
            _ci(t["balanced_accuracy"], ci.get("balanced_accuracy")),
            _ci(t["macro_auc"], ci.get("macro_auc")),
            _ci(pcv.get("sensitivity"), ci.get(f"{pc}_sensitivity")),
            _ci(pcv.get("specificity"), ci.get(f"{pc}_specificity")),
        ])
    pc_name = exps[0]["test"].get("positive_class") or "positive"
    out += _table(["Experiment", "Lungs hidden?", "Accuracy", "Majority baseline", "Balanced accuracy",
                   "Macro AUC", f"{pc_name} sensitivity", f"{pc_name} specificity"], rows)

    occ = [e for e in exps if "test_lungs_occluded" in e]
    if occ:
        out += ["### Normally trained models, tested with the lungs blacked out", "",
                "If a model relies on the lungs, hiding them should push AUC toward 0.5.", ""]
        rows = []
        for e in occ:
            a, b = e["test"], e["test_lungs_occluded"]
            rows.append([e["name"], _f(a["macro_auc"]), _ci(b["macro_auc"], b.get("ci95", {}).get("macro_auc")),
                         _f(b["balanced_accuracy"])])
        out += _table(["Experiment", "Macro AUC (normal)", "Macro AUC (lungs hidden)", "Balanced accuracy (lungs hidden)"], rows)

    # For models trained with the lungs hidden the box is black, so this share is meaningless;
    # their gradcam_examples.png still shows where they look instead.
    cams = [e for e in exps if "gradcam" in e and not e["trained_on_occluded"]]
    if cams:
        classes = exps[0]["classes"]
        out += ["## Grad-CAM: where does each model look?", "",
                "Share of heatmap mass inside the lung box. Attention spread evenly over the "
                f"image would give {cams[0]['gradcam']['box_area_fraction']:.2f}. Models trained "
                "with the lungs hidden are left out (their box is black); see their example images.", ""]
        rows = [[e["name"], _f(e["gradcam"]["mean_lung_fraction"], 2)]
                + [_f(e["gradcam"]["per_class"].get(c, {}).get("mean_lung_fraction"), 2) for c in classes]
                for e in cams]
        out += _table(["Experiment", "All"] + classes, rows)
        out += ["Example heatmaps: `<experiment>/gradcam_examples.png`.", ""]

    out += ["## Confusion matrices (rows = true, columns = predicted)", ""]
    for e in exps:
        t = e["test"]
        out += [f"**{e['name']}**", ""]
        out += _table(["true \\ pred"] + t["classes"], [[c] + row for c, row in zip(t["classes"], t["confusion_matrix"])])
    return out


def main(argv=None) -> Path:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default="results")
    args = ap.parse_args(argv)
    root = Path(args.results)

    lines = ["# Results", "", "Generated by `scripts/make_report.py`. Do not edit by hand.", ""]
    audit_path = root / "audit" / "audit.json"
    if audit_path.exists():
        lines += audit_section(load_json(audit_path))
    exps = [load_json(p) for p in sorted(root.glob("*/metrics.json"))]
    if exps:
        lines += experiments_section(exps)
    if not audit_path.exists() and not exps:
        raise SystemExit(f"No results found under {root}")

    out = root / "summary.md"
    out.write_text("\n".join(lines))
    print(f"Wrote {out}")
    return out


if __name__ == "__main__":
    main()
