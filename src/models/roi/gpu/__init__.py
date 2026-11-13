"""Training-side modules of the ROI detector + refiner. ROOT is the ROI workspace (work/roi): configs, manifests, the
photo links, the target cache and the starting weights live there (prepared by src.training.train_roi)."""
from src import paths

ROOT = paths.WORK / "roi"
HOST_IDENTITY = None
