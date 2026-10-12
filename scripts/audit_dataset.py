"""Audit the dataset before training: class counts, metadata shortcut test,
duplicate check. CPU only.

    python scripts/audit_dataset.py --data-root /path/to/Data --out results/audit
"""

from __future__ import annotations

import argparse
from pathlib import Path

from cxrshortcut.audit import collect_metadata, duplicate_report, metadata_shortcut_test, save_pair_sheet
from cxrshortcut.data import add_validation_split, class_names, counts_table, index_dataset
from cxrshortcut.utils import run_info, save_json


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True, help="Folder containing train/ and test/ (searched recursively)")
    ap.add_argument("--out", default="results/audit")
    ap.add_argument("--classes", nargs="+", help="Keep only these classes, e.g. COVID19 NORMAL")
    ap.add_argument("--val-fraction", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-distance", type=int, default=6, help="Near-duplicate threshold in dhash bits (of 256)")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    df = add_validation_split(index_dataset(args.data_root, args.classes), args.val_fraction, args.seed)
    classes = class_names(df)
    counts = counts_table(df)
    print("Images per class and split:\n", counts.to_string(), "\n", sep="")

    print("Reading image metadata ...")
    meta = collect_metadata(df, workers=args.workers)
    meta.to_csv(out / "metadata.csv", index=False)
    shortcut = metadata_shortcut_test(meta, classes, seed=args.seed)
    m = shortcut["metrics"]
    print(f"Metadata-only classifier on test: macro AUC {m['macro_auc']:.3f}, "
          f"balanced accuracy {m['balanced_accuracy']:.3f} "
          f"(majority baseline accuracy {m['majority_baseline_accuracy']:.3f})")
    for c, v in m["per_class"].items():
        print(f"  {c}: one-vs-rest AUC {v['auc_one_vs_rest']:.3f}")

    print("Checking duplicates ...")
    dup_summary, pairs = duplicate_report(df, max_distance=args.max_distance, workers=args.workers)
    pairs.to_csv(out / "near_duplicates_test_vs_train.csv", index=False)
    save_pair_sheet(pairs, out / "near_duplicates_examples.png")
    for k, v in dup_summary.items():
        print(f"  {k}: {v}")

    result = {
        "classes": classes,
        "counts": counts.to_dict(orient="index"),
        "metadata_shortcut_test": shortcut,
        "duplicates": dup_summary,
    }
    save_json(result, out / "audit.json")
    save_json(run_info(vars(args)), out / "run_info.json")
    print(f"\nSaved to {out}/")
    return result


if __name__ == "__main__":
    main()
