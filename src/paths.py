"""Every folder the pipeline reads or writes. Override any of them with an environment variable."""
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _folder(variable: str, default: Path) -> Path:
    return Path(os.environ.get(variable, default)).resolve()


DATA = _folder("FILAMENT_DATA", REPO / "data")             # folds.json; train/ and test/ from Kaggle
MODELS = _folder("FILAMENT_MODELS", REPO / "models")       # one folder per trained model: {name}-f{fold}
WORK = _folder("FILAMENT_WORK", REPO / "work")             # inference outputs and scorer tables
SCRATCH = _folder("FILAMENT_SCRATCH", WORK / "scratch")    # U-Net maps (~0.3 GB, compressed); training cache (~8 GB, retraining only)
OUTPUTS = _folder("FILAMENT_OUTPUTS", REPO / "outputs")    # submission, figures, logs
DOWNLOADS = _folder("FILAMENT_DOWNLOADS", REPO / "downloads")  # downloaded archives (7.4 GB while extracting)
CONFIGS = REPO / "configs"

FOLDS_JSON = Path(os.environ.get("FILAMENT_FOLDS_JSON", REPO / "data" / "folds.json"))   # tracked in the repository
ANNOTATIONS = DATA / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
TRAIN_IMAGES = DATA / "train" / "train_images"
TEST_IMAGES = DATA / "test" / "test_images"


def model_dir(name: str, fold) -> Path:
    """models/{name}-f{fold} (fold 0-4) or models/{name}-full."""
    return MODELS / (f"{name}-full" if fold in ("full", -1) else f"{name}-f{fold}")


TRAIN_ANNOTATIONS = ANNOTATIONS                            # alias used by the evidence-model code


def work(*parts: str) -> Path:
    """A path inside the work folder; its parent folders are created."""
    p = WORK.joinpath(*parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p
