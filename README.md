<div align="center">
  <h1>Kaggle Filament Challenge - 2026</h1>
  <p>Our solution to the Kaggle competition <i>filament-segmentation-2026</i>:<br>training, inference and the final submission, reproducible from one notebook.</p>
</div>

<div align="center">
  <p align="center">
    <a href="https://www.kaggle.com/competitions/filament-segmentation-2026/writeups/multi-family-mask2former-approach" target="_blank">
      <img src="https://img.shields.io/badge/Kaggle-%2310%20Leaderboard%20Solution-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white" alt="Kaggle — #10 Leaderboard Solution" height="40"/>
    </a>
  </p>
</div>

<div align="center">
  <a href="#folds"><img src="https://img.shields.io/badge/OOF%20PQ-0.4801-brightgreen?style=flat" alt="Out-of-fold PQ 0.4801" /></a>
  <a href="https://www.kaggle.com/competitions/filament-segmentation-2026/leaderboard"><img src="https://img.shields.io/badge/Public%20LB%20PQ-0.42-5865F2?style=flat" alt="Public leaderboard PQ 0.42" /></a>
  <a href="https://www.kaggle.com/competitions/filament-segmentation-2026/leaderboard"><img src="https://img.shields.io/badge/Private%20LB%20PQ-0.39%20%C2%B7%20rank%2010%20of%201708-orange?style=flat" alt="Private leaderboard PQ 0.39, rank 10 of 1,708" /></a>
  <img src="https://img.shields.io/badge/Inter--annotator%20PQ-0.340-lightgrey?style=flat" alt="Inter-annotator PQ 0.340" />
  <br>
  <img src="https://img.shields.io/badge/Python-3.12-3776AB?style=flat&logo=python&logoColor=white" alt="Python 3.12" />
  <img src="https://img.shields.io/badge/PyTorch-2.8-EE4C2C?style=flat&logo=pytorch&logoColor=white" alt="PyTorch 2.8" />
  <img src="https://img.shields.io/badge/CUDA-12.8-76B900?style=flat&logo=nvidia&logoColor=white" alt="CUDA 12.8" />
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-AGPL--3.0-blue.svg?style=flat" alt="License AGPL-3.0" /></a>
</div>

<p align="center">
  <a href="https://www.kaggle.com/competitions/filament-segmentation-2026">Competition</a>
  ·
  <a href="report/main.pdf">Report (PDF)</a>
  ·
  <a href="PIPELINE.ipynb">Pipeline notebook</a>
  ·
  <a href="#reproducing-the-result">Reproduce</a>
  ·
  <a href="https://drive.google.com/file/d/1OV9XdvO7Py1zkZEmGnV6gbC_ZS2vmQlQ/view?usp=sharing">Trained models (7.4 GB)</a>
</p>


## TL;DR

* **Task.** Outline every solar filament as its own mask in 2048 x 2048 full-disk H-alpha photos. Entries are scored by panoptic
  quality (PQ, IoU > 0.5) against each annotator's drawing separately, and the annotators often disagree.
* **Idea.** Finding filaments is not the bottleneck: our first model already proposes a candidate for 85-89 % of the annotated ones.
  We therefore spent our effort on *choosing* which candidates to submit.
* **What we built.** Two Mask2Former families over 16 test-time views produce the candidates; a Mask R-CNN, three U-Nets and a tiled
  ROI detector check them; a LightGBM scorer predicts each candidate's IoU with the annotators (129 numbers per candidate); an exact
  integer program keeps the best non-overlapping set; and each kept boundary is redrawn at full resolution from the U-Net maps.
* **Results.** Out-of-fold PQ **0.4801** on the 707 training photos (95 % interval **0.4714-0.4882**), against **0.4629** for one
  Mask2Former with a fixed threshold and **0.340** between two annotators of the same photo. Final **private leaderboard PQ 0.39,
  rank 10 of 1,708 teams** (public leaderboard: 0.42, rank 36).
* **Reproducible.** `PIPELINE.ipynb` rebuilds the submitted file, **byte for byte** from the same U-Net maps: about **50 min** on
  4 x 12 GB GPUs with the 40 downloaded models, or about **115 GPU-hours** to retrain them all.

## Monitored with WeightsLab

The experiment was fully monitored with [WeightsLab](https://github.com/GrayboxTech/weightslab): the training curves of every run side by
side, and the 707 training photos browsed with their annotations and predicted filaments overlaid. WeightsLab was used for monitoring only;
it is not needed to reproduce the result.

![The Mask2Former runs monitored in WeightsLab](reference/figures/weightslab_monitoring.png)

## Solar filament instance segmentation

Solar filaments are dark, elongated clouds of cool plasma seen against the Sun's disk in H-alpha. The task
([filament-segmentation-2026](https://www.kaggle.com/competitions/filament-segmentation-2026)) is to outline every filament in 2048 x 2048
full-disk images as its own mask. Entries are scored by panoptic quality (PQ) against each annotator's drawing separately: a predicted mask
counts only if it overlaps that annotator's filament by more than half (IoU > 0.5).

![A training photo and its two annotators' filaments](reference/figures/task_example.png)

The training set has 707 photos with 1,154 annotator records (one to three annotators per photo, 8,199 filaments). Annotators often
disagree, as above: 11 filaments for one, 7 for the other, and only some of them in common.

**Result: out-of-fold PQ 0.4801** on the 707 training photos (95% bootstrap interval 0.4714-0.4882; the five folds range from 0.4733 to
0.4909). Each photo is predicted by models that never saw it during training (see [Folds](#folds)). On the public leaderboard the
submission (ref 58424919) scores 0.42, like the previous one (ref 58390515, out-of-fold 0.4783). On the **private leaderboard** (final
standing, competition closed on 2026-11-16) it scores **0.39**, **rank 10 of 1,708 teams** (the
winning private score is 0.64).

![Out-of-fold PQ at each stage](reference/figures/score_progression.png)

## Approach

```
 photos ──► Mask2Former, grayscale input (5 folds) ─────────► 8 views ──┐
        ──► Mask2Former, limb-corrected input (5 folds) ────► 8 views ──┴─► one fused 16-view candidate pool per photo
        ──► Mask R-CNN + 3 U-Nets (5 folds) ──► detector candidates, probability + spine maps ──┐
        ──► ROI detector + refiner (5 folds) ──► ROI candidates ────────────────────────────────┤
                                                                                                  ▼
  129 numbers per candidate ──► LightGBM scorer q (expected matched IoU) ──► on test, 5 fold models fused
   ──► exact non-overlapping selection ──► boundaries redrawn within 2 px from the U-Net maps ──► submission.csv
```

1. **Candidates.** Two Swin-S Mask2Former families, trained at native resolution on the 5 folds of `data/folds.json` (grouped by date).
   One reads the grayscale photo; the other reads the photo, its limb-darkening-corrected disk and a CLAHE copy. Each fold model predicts
   8 rotated / mirrored views, and the 16 views of a fold's two models are fused into one candidate pool.
2. **Evidence.** Independent models trained on the same folds check each candidate: a Mask R-CNN, three U-Nets (filament probability and
   centre-line maps), and a tiled ROI detector with a mask refiner.
3. **Learned scorer.** LightGBM predicts, for each candidate, its expected IoU with the annotators' filaments. It uses 129 numbers: shape,
   scores, image contrast, competing candidates, agreement with every evidence model, how stable the candidate's outline is across the
   16 raw Mask2Former views, and what each U-Net's probability and spine maps say inside and around it. It is averaged over 10 seeds and
   cross-fitted: every training photo is scored by a model that never saw its fold, on candidates from models that never saw it.
4. **Selection.** Among a photo's candidates, the set maximising the sum of (q - tau) with no two masks overlapping by more than 10 % of
   the smaller one, solved exactly (integer programming). On test, the 5 fold models' candidates are fused first.
5. **Full-resolution boundaries.** Mask2Former masks come out at 1/4 resolution. Each kept mask keeps its extent, and its boundary is redrawn
   within 2 px from the full-resolution U-Net probability maps, in two steps:
   * **Sub-pixel shift.** The probability map is moved 0.25 px towards the top of the photo, where the annotators' outlines sit on
     average.
   * **Area blend.** The new mask's area is the geometric mean of the Mask2Former area and the U-Net area (the U-Net blob above 0.5
     inside the 2 px band, ratio limited to 0.8-1.25); the band pixels with the highest probability are kept up to that area.

The models, epochs, scorer settings, selection threshold (tau 0.25) and redraw rule are chosen on out-of-fold PQ. The boundary redraw adds
+0.0041 to the masks as emitted (0.4760 to 0.4801, 95% interval +0.0026 to +0.0055).

**Scorer update (after submission 58390515).** Two blocks were added to the scorer, both read from outputs the models already produce
(nothing was retrained): the agreement of each candidate with the 16 raw views (17 numbers: in how many views it appears, how much its
outline moves from view to view, the spread of the views' scores and areas) and the 3 U-Nets' filament-probability maps (12 numbers each,
as for their spine maps). With 10 seeds instead of 5, out-of-fold PQ goes from 0.4783 to 0.4801 (+0.0018, paired 95% interval -0.0004 to
+0.0039; better in 3 of 5 folds). The scorer now recognises more of its weak candidates, so the chosen tau rises from 0.225 to 0.25: out of
fold, 265 fewer false positives (1,969) for 117 fewer matches (5,409). Changing only the scorer's seeds moves out-of-fold PQ by about
0.0005, so the gain was checked on two independent sets of 10 seeds (+0.0010 and +0.0013 on the masks as emitted, averaged over tau
0.2-0.275). Not kept, as they showed no gain out of fold: candidates from the Mask R-CNN and ROI banks added to the pool, a threshold or
boundary shift per year, site or hour, crescent-shape features, a second-stage scorer, and the ROI refiner's own mask statistics.

![Out-of-fold predictions on training photos](reference/figures/oof_predictions.png)

<img src="reference/figures/redraw_closeup.png" alt="Boundary redraw close up" width="420">

## Folds

Every model uses the same 5 folds, fixed in `data/folds.json` (142, 142, 141, 141 and 141 photos).

* **Grouped by day.** The 707 photos were taken on 656 days (UTC), up to 3 per day, mostly from different observatory sites; the date opens the
  file name (`YYYYMMDDhhmmss`, the letter before `h` is the site). Photos of the same day show the same filaments, so a day is never
  split between two folds. The folds were drawn once as a group k-fold over that date: no day is shared between folds, and every fold
  holds photos from at least 11 of the 12 years 2011-2022 and from all six sites.
* **Whole photos.** A fold holds photos, so all annotators of a photo are on the same side.
* **Fixed file.** The file is the reference: `src/data/split.py` reads it for every model (`FILAMENT_FOLDS_JSON`, `FILAMENT_VAL_FOLD`).
  Drawing a new group k-fold gives a different assignment, and then the shipped models would no longer be out of fold.
* **How they are used.** Each model is trained five times, once per fold, on the other four folds (565-566 photos), and predicts the fold
  it did not see. Every training photo therefore has an out-of-fold prediction. The scorer is cross-fitted on the same folds, and every
  choice (models, epochs, scorer settings, thresholds) is made on these out-of-fold predictions. On test, the five fold models are fused.

| Fold | 0 | 1 | 2 | 3 | 4 | All 707 photos |
|---|---|---|---|---|---|---|
| Out-of-fold PQ of the final pipeline | 0.4733 | 0.4909 | 0.4775 | 0.4825 | 0.4755 | 0.4801 |

The spread between folds (sd 0.0070) comes from the photos each fold holds, not from better or worse models: the five fold models are
trained the same way.

## Reproducing the result

`PIPELINE.ipynb` runs the whole pipeline, from the photos to `outputs/submission.csv`, with every step explained. Its last check compares
the result with the submitted file (Kaggle submission ref 58424919, kept in `reference/submission.csv`): by SHA-256, then mask by mask.

* **What to expect.** From the same U-Net maps, the selection and the redraw rebuild the submitted file **byte for byte**. The U-Net maps
  themselves are recomputed by the notebook, and they were not bit-identical between our two machines: for an earlier version of the
  pipeline, 8 of the 1,247 test masks changed from one machine to the other (the other 1,239 matched at a mean IoU of 0.999). On another
  machine, expect the same kind of difference (a few masks) rather than the same hash.
* **Integrity.** The data, the 40 checkpoints and the shipped inference outputs are checked against `reference/CHECKSUMS.json`. A missing or
  changed model is listed with its training time, and retrained only after you confirm.
* **Training.** The training code was checked against the original runs: same losses at the first steps (to the digit for the ROI models,
  within GPU rounding noise for Mask2Former).
* **Time.** With the downloaded models, a run takes about 50 min on 4 x 12 GB GPUs (U-Net maps, then ~20 min of CPU for the scorer and
  redraw). Retraining all 40 models takes about 115 GPU-hours, about 28 h on 4 x 12 GB GPUs (details below).

### Hardware used

The models were trained on one rented cloud GPU server:

| | |
|---|---|
| GPUs | 4 x NVIDIA GeForce RTX 3080 Ti, 12 GB each (PCIe 4.0 x16), driver 595.99 |
| CPU | 2 x AMD EPYC 7543 (32 cores each); the container was limited to about 27 cores |
| Memory | 377 GB RAM, including a 41 GB RAM disk (`/dev/shm`) for the large intermediate files |
| Disk | 32 GB |
| Software | Ubuntu 24.04, Python 3.12.3, PyTorch 2.8.0 with CUDA 12.8 |

The submissions (58424919, and 58390515 before it) were produced on a second machine, where the notebook rebuilds the latest byte for byte, from the same 40 downloaded models (inference, scorer,
selection and redraw only; nothing retrained): 2 x NVIDIA GeForce RTX 3080 (10 GB each), 4 CPU cores, 76 GB RAM, same PyTorch and CUDA
versions. There a run takes about 1 h 40 min, most of it in the CPU steps.

### Training time, and why it is long

Retraining all 40 models takes about 115 GPU-hours, about 28 h on this server. Inference adds about 1.5 h.

| Models | Runs | Time per run | GPU-hours | Wall time on the 4 GPUs |
|---|---|---|---|---|
| Mask2Former, 2 families x 5 folds | 10 | ~5.5 h | ~55 | ~16.5 h (one run per GPU, 3 rounds) |
| Mask R-CNN + 3 U-Nets, 5 folds | 20 | 1.2-2.7 h | ~35 | ~4.5 h (two runs per GPU) |
| ROI detector + refiner, 5 folds | 10 | 2-3 h | ~25 | ~5 h (up to 4 runs per GPU) |

* **Every model is trained 5 times**, once per fold. The scorer learns from out-of-fold predictions, so every training photo must be
  predicted by models that never saw it.
* **Mask2Former trains at native resolution**: 1024-pixel crops and whole 2048-pixel frames, 50 epochs of 231 optimiser steps (batch 8).
  On a 12 GB GPU the batch of 8 is accumulated one image at a time, and the backbone is recomputed in the backward pass (gradient
  checkpointing), so a step takes about 1.6 s. The held-out evaluations every 5 epochs add the rest.
* **12 GB limits sharing**: only one Mask2Former training fits on a GPU. A GPU with 40-80 GB of memory would avoid the accumulation and
  the recomputation, so it should be faster per step (not measured here).
* **A cheaper variant**: in the report's ablations, the ROI detector + refiner and the second Mask2Former family each add nothing
  measurable out of fold. Without them, retraining would take about 62 GPU-hours (about 15 h on 4 GPUs). Removing both at once was not
  measured.

### Setup

```bash
git clone git@github.com:Emualliug2705/kaggle_filament_2026.git && cd kaggle_filament_2026
python3.12 -m venv venv && . venv/bin/activate
pip install -r requirements.txt jupyterlab   # torch 2.8 for CUDA 12.8 comes from the PyTorch index set in the file
jupyter lab PIPELINE.ipynb          # run all cells; or headless: python run_headless.py
```

* **Data** (photos and annotations, 0.7 GB, into `data/`). **With a Kaggle token** (`~/.kaggle/access_token` or `kaggle.json`, or
  `KAGGLE_API_TOKEN`; competition rules accepted) they are downloaded from Kaggle and replace the data already there. **Without a token, or
  if Kaggle fails**, the data already in `data/` are kept, else `data.zip` is downloaded from Google Drive (link and SHA-256 in
  `reference/DOWNLOADS.json`; you can also place the file in `downloads/`), its SHA-256 checked before extraction. Whatever the source,
  every photo is checked by SHA-256 in section 0.3.
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
  evaluation/       out-of-fold matching and PQ, validation sweeps, comparison of two submission files
  utils/            masks, events
configs/            model and data settings
data/folds.json     the 5 folds (photos and annotations are downloaded)
reference/          checksums, download sources, the submitted file and its SHA-256, epoch table, figures
report/             the competition report (LaTeX, first draft)
submission/         the final submitted file (Kaggle ref 58424919) and the report PDF
kaggle_writeup/      the competition writeup: title, subtitle, URL, thumbnail cards, content
models/  work/  outputs/  downloads/     created by the notebook (git-ignored)
```

## Licence

AGPL-3.0 (`LICENSE`). Parts of `src/` are adapted from open-source public kaggle submission to this challenge; see `THIRD_PARTY_NOTICES.md`.
