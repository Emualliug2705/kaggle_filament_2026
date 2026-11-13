# kaggle_filament_2026
Code for final submission - LB 0.41

## Solar filament instance segmentation

Solar filaments are dark, elongated clouds of cool plasma seen against the Sun's disk in H-alpha. The task
([filament-segmentation-2026](https://www.kaggle.com/competitions/filament-segmentation-2026)) is to outline every filament in 2048 x 2048
full-disk images as its own mask. Entries are scored by panoptic quality (PQ) against each annotator's drawing separately: a predicted mask
counts only if it overlaps that annotator's filament by more than half (IoU > 0.5).

**Final submission: out-of-fold PQ 0.4770 on the 707 training photos** (Kaggle submission ref 58304440).

![Out-of-fold PQ at each stage](reference/figures/score_progression.png)

## Approach

```
 photos ──► Mask2Former, grayscale input (5 folds) ─────────► 8 views ──┐
        ──► Mask2Former, limb-corrected input (5 folds) ────► 8 views ──┴─► one fused 16-view candidate pool per photo
        ──► Mask R-CNN + 3 U-Nets (5 folds) ──► detector candidates, probability + spine maps ──┐
        ──► ROI detector + refiner (5 folds) ──► ROI candidates ────────────────────────────────┤
                                                                                                  ▼
   76 numbers per candidate ──► LightGBM scorer q (expected matched IoU) ──► exact non-overlapping selection
   ──► 5 fold models fused on test ──► boundaries redrawn within 3 px from the U-Net maps ──► submission.csv
```

1. **Candidates.** Two Swin-S Mask2Former families, trained at native resolution on the 5 folds of `data/folds.json` (grouped by date).
   One reads the grayscale photo; the other reads the photo, its limb-darkening-corrected disk and a CLAHE copy. Each fold model predicts
   8 rotated / mirrored views, and the 16 views of a fold's two models are fused into one candidate pool.
2. **Evidence.** Independent models trained on the same folds check each candidate: a Mask R-CNN, three U-Nets (filament probability and
   centre-line maps), and a tiled ROI detector with a mask refiner.
3. **Learned scorer.** LightGBM predicts, for each candidate, its expected IoU with the annotators' filaments. It uses 76 numbers: shape,
   scores, image contrast, competing candidates and agreement with every evidence model. It is cross-fitted: every training photo is scored
   by a model that never saw its fold, on candidates from models that never saw it.
4. **Selection.** Among a photo's candidates, the set maximising the sum of (q - tau) with no two masks overlapping by more than 10 % of
   the smaller one, solved exactly (integer programming). On test, the 5 fold models' candidates are fused first.
5. **Full-resolution boundaries.** Mask2Former masks come out at 1/4 resolution. Each kept mask keeps its extent, and its boundary is redrawn
   within 3 px from the full-resolution U-Net probability maps.

The models, epochs, scorer settings and selection threshold are chosen on out-of-fold PQ. For the boundary redraw, 1 px and 3 px are
equivalent out of fold (0.4774 vs 0.4770, difference -0.0003, 95% interval -0.0012 to +0.0006); the final uses 3 px.

![Out-of-fold predictions on training photos](reference/figures/oof_predictions.png)

<img src="reference/figures/redraw_closeup.png" alt="Boundary redraw close up" width="420">

## Reproducing the result

`PIPELINE.ipynb` runs the whole pipeline, from the photos to `outputs/submission.csv`, with every step explained. It rebuilds the submitted
file **byte for byte**; its last check compares SHA-256 hashes.

* **Integrity.** The data, the 40 checkpoints and the shipped inference outputs are checked against `reference/CHECKSUMS.json`. A missing or
  changed model is listed with its training time, and retrained only after you confirm.
* **Training.** The training code was checked against the original runs: same losses at the first steps (to the digit for the ROI models,
  within GPU rounding noise for Mask2Former).
* **Time.** With the downloaded models, a run takes about 45 min on 4 x 12 GB GPUs (U-Net maps, then ~15 min of CPU for the scorer and
  redraw). Retraining everything takes about 30 GPU-hours.

### Setup

```bash
git clone git@github.com:Emualliug2705/kaggle_filament_2026.git && cd kaggle_filament_2026
python3.12 -m venv venv && . venv/bin/activate
pip install --extra-index-url https://download.pytorch.org/whl/cu128 \
    -r env/requirements-venv.txt -r env/requirements-extra.txt jupyterlab
jupyter lab PIPELINE.ipynb          # run all cells; or headless: python run_headless.py
```

* **Kaggle.** Put your API token in `~/.kaggle/kaggle.json` and accept the competition rules. The photos and annotations are downloaded into
  `data/`.
* **Models.** `filament_models.tar` (7.4 GB: the trained models and the inference outputs) is downloaded from Google Drive (the link is in
  `reference/DOWNLOADS.json`). You can also place it in `downloads/`. Its SHA-256 is checked before extraction.
* **Hardware.** Linux, NVIDIA GPUs with 12 GB or more, a driver supporting CUDA 12.8, and ~35 GB free for `work/scratch`. Keep the
  repository on a normal disk, not a `noexec` RAM disk. To move a folder, set `FILAMENT_DATA`, `FILAMENT_MODELS`, `FILAMENT_WORK`,
  `FILAMENT_SCRATCH`, `FILAMENT_OUTPUTS` or `FILAMENT_DOWNLOADS`.

## Layout

```
src/
  paths.py          every folder the pipeline uses (environment overrides)
  data/             folds and splits, photo utilities
  models/           mask2former/ (model, data, training loop, evaluation), maskrcnn.py, unet.py, roi/ (detector + refiner)
  training/         launchers and trainers: mask2former.py, evidence_models.py, train_unet.py, train_maskrcnn.py, train_roi.py
  inference/        Mask2Former caches and views, two-family pools, ROI banks, view fusion
  scorer/           features + LightGBM + selection (selector.py), exact selection (select.py), boundary redraw
  evaluation/       out-of-fold matching and PQ, validation sweeps
  utils/            masks, events
configs/            model and data settings
data/folds.json     the 5 folds (photos and annotations are downloaded)
reference/          checksums, download sources, the submitted file's SHA-256, epoch table, figures
models/  work/  outputs/  downloads/     created by the notebook (git-ignored)
```

## Licence

AGPL-3.0 (`LICENSE`). Parts of `src/` are adapted from open-source projects; see `THIRD_PARTY_NOTICES.md`.
Keep this repository **private until the competition ends**: Kaggle rules generally forbid sharing code outside Kaggle during a competition.
