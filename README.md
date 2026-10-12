# Does a COVID-19 X-ray classifier look at the lungs?

[![CI](https://github.com/anupbagale/2D_Classification/actions/workflows/ci.yml/badge.svg)](https://github.com/anupbagale/2D_Classification/actions/workflows/ci.yml)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/anupbagale/2D_Classification/blob/main/notebooks/run_all_colab.ipynb)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

My first version of this project trained a small CNN on chest X-rays and reported
**98% accuracy** for COVID-19 vs normal and **93%** for COVID-19 vs normal vs pneumonia.
Those numbers came from a test set that was also used to monitor training, and
accuracy hid a class imbalance. More importantly, they could not tell whether the
model had learned anything about the lungs at all.

This version asks that question directly: **when a model classifies these X-rays,
is it reading the lungs, or is it recognising where each image came from?**

## Why this matters

Public COVID-19 X-ray datasets were assembled quickly during the pandemic, often by
combining COVID images from one set of sources with normal and pneumonia images from
others. Then the class label is correlated with the hospital, scanner, patient age,
image format or text markers burned into the image. A CNN can learn those
**shortcuts** and still score near-perfectly on a test set drawn from the same mix,
while failing on images from a new hospital. This has been shown for widely used
COVID X-ray datasets ([DeGrave et al., 2021](https://doi.org/10.1038/s42256-021-00338-7)),
and a review of COVID imaging models found methodological problems that made none of
them ready for clinical use ([Roberts et al., 2021](https://doi.org/10.1038/s42256-021-00307-0)).

## How the question is tested

| Test | What it does | If the model reads the lungs | If it relies on shortcuts |
|---|---|---|---|
| **Metadata-only classifier** | Random forest on file size, format, colour mode and brightness; never sees image content | AUC ≈ 0.5 | AUC well above 0.5: the label can be read off the file |
| **Duplicate check** | Byte-identical and near-identical (perceptual hash) images shared by train and test | none | test scores are inflated by memorised images |
| **Train with lungs hidden** | DenseNet121 trained *and* tested with the lung region blacked out | AUC ≈ 0.5 | AUC well above 0.5: information outside the lungs predicts the label |
| **Hide lungs at test time** | Normally trained models tested on lung-occluded images | AUC drops toward 0.5 | AUC stays high |
| **Grad-CAM** | Share of each heatmap inside the lung region | well above the box's share of the image | near or below it |

## Results

> **Pending the first full run.** Open the Colab notebook above and run all cells
> (about an hour on a free T4 GPU). Every number is generated into
> [`results/summary.md`](results/summary.md) by `scripts/make_report.py`; this section
> will summarise them.

## Methods

**Data.** [Chest X-ray (Covid-19 & Pneumonia)](https://www.kaggle.com/datasets/prashant268/chest-xray-covid19-pneumonia)
on Kaggle, pre-split into train and test folders. The classes are imbalanced:

| Class | Train | Test |
|---|---|---|
| COVID-19 | 460 | 116 |
| Normal | 1,266 | 317 |
| Pneumonia | 3,418 | 855 |
| **Total** | **5,144** | **1,288** |

Always predicting "pneumonia" already gives 66% test accuracy. (The copy used for v1
had 1,292 test images; the current Kaggle download has 1,288.)

**Splits.** 15% of the training images (stratified by class, fixed seed) form a
validation set. The best epoch is chosen on validation AUC. The test set is used once,
at the end. The split for each run is saved in `results/<run>/splits.csv`.

**Preprocessing.** Every image is converted to grayscale (so RGB vs grayscale files
look the same), resized to 224×224 and repeated to three channels. Augmentation is
small rotations, zoom, shifts and contrast changes. **No flips**: an X-ray is never
upside down, and a left-right flip puts the heart on the wrong side.

**Models.**
- `original_cnn`: the v1 architecture, unchanged, as a baseline. About 98% of its
  parameters sit in one Flatten → Dense layer.
- `densenet121`: ImageNet-pretrained DenseNet121 (the CheXNet backbone). The new
  head is trained first, then the last dense block is fine-tuned at a low learning
  rate. Batch-norm layers stay frozen.

Training uses Adam, class weights to balance the classes, early stopping on
validation AUC and the best checkpoint.

**Metrics.** Accuracy next to the majority-class baseline; balanced accuracy;
per-class sensitivity, specificity and PPV; one-vs-rest and macro AUC; confusion
matrices. 95% confidence intervals come from 1,000 bootstrap resamples of the test set.

**Lung region.** A fixed box covering 10–85% of the image height and 12–88% of its
width (`DEFAULT_LUNG_BOX` in `src/cxrshortcut/images.py`), about 57% of the image.

## Reproduce

On Colab: open [`notebooks/run_all_colab.ipynb`](notebooks/run_all_colab.ipynb) and run all cells.

Locally (a GPU is strongly recommended for training):

```bash
pip install -r requirements.txt && pip install -e .
python scripts/audit_dataset.py --data-root PATH/TO/Data
python scripts/run_experiment.py --data-root PATH/TO/Data --model original_cnn --epochs 25 --eval-occluded
python scripts/run_experiment.py --data-root PATH/TO/Data --model densenet121 --epochs 10 --fine-tune-epochs 10 --eval-occluded
python scripts/run_experiment.py --data-root PATH/TO/Data --model densenet121 --name densenet121_occluded --epochs 10 --fine-tune-epochs 10 --occlude
python scripts/make_report.py
pytest   # 45 tests on a synthetic dataset, CPU only
```

`--classes COVID19 NORMAL` repeats any step as the binary task from v1.

To try the pipeline without downloading anything, generate a small synthetic dataset.
Its COVID class has a planted shortcut (different file format, size and a corner marker),
and the audit and the lungs-hidden model both catch it:

```python
from cxrshortcut.synthetic import make_synthetic_dataset
make_synthetic_dataset("synthetic")   # then use --data-root synthetic
```

Each run saves its configuration and library versions (`run_info.json`), training
history, per-image predictions, metrics and Grad-CAM examples.

## Repository layout

```
src/cxrshortcut/
  data.py        index train/test folders, stratified validation split
  images.py      loading, augmentation, lung-box occlusion (tf.data)
  audit.py       metadata shortcut test, exact and near-duplicate check
  models.py      original_cnn, small_cnn, densenet121
  train.py       training with validation-based model selection
  metrics.py     imbalance-aware metrics and bootstrap CIs
  evaluate.py    test-set predictions and metrics
  gradcam.py     Grad-CAM and share of attention inside the lung box
  synthetic.py   fake dataset for tests
scripts/         audit_dataset.py, run_experiment.py, make_report.py
notebooks/       run_all_colab.ipynb; original/ holds the v1 notebooks
tests/           pytest suite, runs in CI on every push
results/         generated metrics, predictions and figures
```

## Limitations

- **The lung region is a rectangle, not a segmentation.** It includes the heart and
  mediastinum and can miss lung tissue in unusually positioned images. Real lung masks
  from a segmentation model would make the occlusion tests sharper.
- **No patient IDs.** The dataset does not identify patients, so images of the same
  patient may sit in both train and test. The near-duplicate check catches only very
  similar images.
- **Grad-CAM is coarse** (a 7×7 grid for DenseNet121 at 224 px) and is not a reliable
  explanation on its own. It is supporting evidence next to the occlusion tests.
- **One dataset, one seed.** The results describe what this dataset allows a model to
  learn. They say nothing about clinical performance, and none of these models should
  be used for diagnosis.

## Version history

- **v2 (October 2026)**: rebuilt as a shortcut-learning study: dataset audit, held-out
  validation split, imbalance-aware metrics with confidence intervals, pretrained
  DenseNet121, lung-occlusion tests, Grad-CAM, tests and CI.
- **v1 (June 2024)**: two Colab notebooks, kept in [`notebooks/original/`](notebooks/original/).
  What was wrong with them, and what v2 changes:
  - The test folder was passed as `validation_data`, so it guided training. There was no untouched test set.
  - Only accuracy was reported. The binary model's first-epoch accuracy, 73.2%, is exactly
    317/433: it was predicting "normal" for every image.
  - Vertical flips were used as augmentation.
  - 7.4M of the CNN's 7.5M parameters were in a single Flatten → Dense layer.
  - Hard-coded Google Drive paths, no seeds, no README, and a placeholder CI workflow.

## License

Code: [MIT](LICENSE). The images are not included; they are subject to the terms of the
Kaggle dataset and its original sources.
