"""Train one model, evaluate it on the untouched test set, and run Grad-CAM.

Examples:
    # Pretrained DenseNet121 on normal images
    python scripts/run_experiment.py --data-root DATA --model densenet121 --name densenet121 \
        --fine-tune-epochs 10 --eval-occluded

    # Same model, trained and tested with the lung region blacked out
    python scripts/run_experiment.py --data-root DATA --model densenet121 --name densenet121_occluded \
        --occlude
"""

from __future__ import annotations

import argparse
from pathlib import Path

from cxrshortcut.data import add_validation_split, class_names, counts_table, index_dataset
from cxrshortcut.evaluate import evaluate_model
from cxrshortcut.gradcam import gradcam_report
import keras
import pandas as pd

from cxrshortcut.images import DEFAULT_LUNG_BOX, find_unreadable
from cxrshortcut.models import MODEL_BUILDERS
from cxrshortcut.train import history_summary, train_model
from cxrshortcut.utils import run_info, save_json


def _ci(ci) -> str:
    return "n/a" if not ci or ci[0] is None else f"{ci[0]:.3f}-{ci[1]:.3f}"


def _print_metrics(title: str, m: dict) -> None:
    ci = m.get("ci95", {})
    pc = m.get("positive_class")
    print(f"\n{title}")
    print(f"  accuracy {m['accuracy']:.3f} (majority baseline {m['majority_baseline_accuracy']:.3f})")
    print(f"  balanced accuracy {m['balanced_accuracy']:.3f}  (95% CI {_ci(ci.get('balanced_accuracy'))})")
    print(f"  macro AUC {m['macro_auc']:.3f}  (95% CI {_ci(ci.get('macro_auc'))})")
    if pc:
        v = m["per_class"][pc]
        print(f"  {pc} sensitivity {v['sensitivity']:.3f}, specificity {v['specificity']:.3f}")
    print(f"  confusion matrix (rows true, cols predicted; {m['classes']}): {m['confusion_matrix']}")


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--model", required=True, choices=sorted(MODEL_BUILDERS))
    ap.add_argument("--name", help="Experiment folder name (default: model name)")
    ap.add_argument("--out-root", default="results")
    ap.add_argument("--classes", nargs="+", help="Keep only these classes, e.g. COVID19 NORMAL")
    ap.add_argument("--positive-class", default="COVID19")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--fine-tune-epochs", type=int, default=0, help="densenet121 only")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--fine-tune-lr", type=float, default=1e-5)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--occlude", action="store_true", help="Train and test with the lung box blacked out")
    ap.add_argument("--eval-occluded", action="store_true",
                    help="Also test a normally trained model on lung-occluded test images")
    ap.add_argument("--gradcam-per-class", type=int, default=40, help="0 to skip Grad-CAM")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--val-fraction", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--weights", default="imagenet", help="'imagenet' or 'none' (densenet121)")
    ap.add_argument("--reuse-model", action="store_true",
                    help="Skip training if this experiment folder already has model.keras and "
                         "history.csv (e.g. to redo evaluation after a crash)")
    ap.add_argument("--verbose", type=int, default=2)
    args = ap.parse_args(argv)

    name = args.name or args.model
    out = Path(args.out_root) / name
    out.mkdir(parents=True, exist_ok=True)
    weights = None if args.weights.lower() == "none" else args.weights
    box = DEFAULT_LUNG_BOX

    df = index_dataset(args.data_root, args.classes)
    unreadable = find_unreadable(df)
    if unreadable:
        print(f"WARNING: {len(unreadable)} file(s) cannot be decoded and are left out "
              f"(listed in {out / 'unreadable_files.txt'}):")
        for p in unreadable:
            print("   ", p)
        (out / "unreadable_files.txt").write_text("\n".join(unreadable) + "\n")
        df = df[~df.path.isin(unreadable)].reset_index(drop=True)
    df = add_validation_split(df, args.val_fraction, args.seed)
    classes = class_names(df)
    df.to_csv(out / "splits.csv", index=False)
    print("Images per class and split:\n", counts_table(df).to_string(), sep="")
    save_json(run_info(vars(args)), out / "run_info.json")

    ckpt, log = out / "model.keras", out / "history.csv"
    if args.reuse_model and ckpt.exists() and log.exists():
        print(f"Reusing the trained model in {out} (no training).")
        model, history = keras.models.load_model(ckpt), pd.read_csv(log)
    else:
        model, history = train_model(
            args.model, df, classes, out,
            img_size=args.img_size, batch_size=args.batch_size, epochs=args.epochs,
            fine_tune_epochs=args.fine_tune_epochs, lr=args.lr, fine_tune_lr=args.fine_tune_lr,
            patience=args.patience, occlude=args.occlude, box=box, seed=args.seed,
            weights=weights, verbose=args.verbose,
        )

    test = df[df.split == "test"]
    common = dict(img_size=args.img_size, batch_size=args.batch_size, box=box,
                  positive_class=args.positive_class, n_boot=args.n_boot, seed=args.seed)
    result = {
        "name": name,
        "model": args.model,
        "trained_on_occluded": args.occlude,
        "classes": classes,
        "excluded_unreadable_files": unreadable,
        "history": history_summary(history),
        "test": evaluate_model(model, test, classes, out, tag="test", occlude=args.occlude, **common),
    }
    _print_metrics("Test set", result["test"])

    if args.eval_occluded and not args.occlude:
        result["test_lungs_occluded"] = evaluate_model(model, test, classes, out, tag="test_lungs_occluded",
                                                       occlude=True, **common)
        _print_metrics("Test set with lungs blacked out", result["test_lungs_occluded"])

    if args.gradcam_per_class > 0:
        result["gradcam"] = gradcam_report(model, test, classes, out, img_size=args.img_size,
                                           n_per_class=args.gradcam_per_class, occlude=args.occlude,
                                           box=box, seed=args.seed)
        g = result["gradcam"]
        print(f"\nGrad-CAM share inside lung box: {g['mean_lung_fraction']:.2f} "
              f"(uniform attention would give {g['box_area_fraction']:.2f})")

    save_json(result, out / "metrics.json")
    print(f"\nSaved to {out}/")
    return result


if __name__ == "__main__":
    main()
