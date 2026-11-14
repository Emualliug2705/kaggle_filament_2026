# Third-party notices

This repository is distributed under the GNU Affero General Public License v3.0 (`LICENSE`).

Parts of `src/` are adapted from two open-source projects, used under their licences:

* **Sol** (https://github.com/Nuwdawg/Sol), AGPL-3.0, by Ismail Kasozi: the Mask2Former training and inference code
  (`src/models/mask2former/`), the candidate fusion and sweep utilities (`src/inference/tta.py`, `src/evaluation/sweep.py`),
  `src/data/split.py`, `src/utils/masks.py`, `src/utils/events.py` and `src/utils/kaggle_sync.py`.
* **BarrelAged** filament-segmentation-2026 entry, Apache License 2.0 (text in `LICENSES/Apache-2.0.txt`): the Mask R-CNN,
  U-Net and ROI detector + refiner models and trainers (`src/models/maskrcnn.py`, `src/models/unet.py`, `src/models/roi/`,
  `src/training/prep_cache.py`, `src/training/train_unet.py`, `src/training/train_maskrcnn.py`), the photo utilities
  (`src/data/photos.py`), the exact selection (`src/scorer/select.py`) and the configuration files in `configs/`.

The adapted files were moved, renamed and had their paths and imports changed; their algorithms are unchanged.

Also cited, for inspiration and as a baseline for comparison:

* **NSO-Solar-Filament-Challenge** (https://github.com/pragyaangaur/NSO-Solar-Filament-Challenge).
