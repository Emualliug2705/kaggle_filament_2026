"""Training of one Mask2Former fold model (with resume and an out-of-memory fallback).

    python -m src.training.mask2former FAMILY K [--workers 12]      FAMILY mask2former | mask2former_limb

Runs src.models.mask2former.train on fold K's training photos: batch 8 by gradient accumulation (micro-batch 1),
backbone checkpointing on whole frames, bf16, the deformable-attention kernel from the hub, 50 epochs, held-out
evaluation every 5 epochs. Writes models/FAMILY-fK/ (model_eXXX.pt every 5 epochs, e050_cache.json, sweeps, logs).
Restarts from last.pt if the process dies; after an out-of-memory crash the retries checkpoint the backbone on crops
too (identical mathematics, slower). last.pt is deleted when training succeeds.
"""
import os
import subprocess
import sys

from src import paths

FAMILIES = {"mask2former": {"FILAMENT_PREP": "raw", "FILAMENT_SCALE_JITTER": "1"},
            "mask2former_limb": {"FILAMENT_PREP": "limb3", "FILAMENT_SCALE_JITTER": "1.25"}}


def main():
    family, fold = sys.argv[1], int(sys.argv[2])
    workers = sys.argv[sys.argv.index("--workers") + 1] if "--workers" in sys.argv else "12"
    out = paths.model_dir(family, fold)
    out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, FILAMENT_FOLDS_JSON=str(paths.FOLDS_JSON), FILAMENT_VAL_FOLD=str(fold), PYTHONUNBUFFERED="1",
               PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True", HF_HUB_ENABLE_HF_TRANSFER="0",
               PYTHONPATH=os.pathsep.join(p for p in (str(paths.REPO), os.environ.get("PYTHONPATH")) if p),
               **FAMILIES[family])
    log = out / "train_stdout.log"
    checkpoint = "whole"
    for attempt in range(1, 5):
        resume = ["--resume"] if (out / "last.pt").is_file() else []
        with open(log, "a") as fh:
            fh.write(f"ATTEMPT {attempt} family={family} fold={fold} checkpoint={checkpoint} {' '.join(resume)}\n")
        with open(log, "a") as fh:
            rc = subprocess.run([sys.executable, "-u", "-m", "src.models.mask2former.train",
                                 "--annotations", str(paths.ANNOTATIONS), "--images", str(paths.TRAIN_IMAGES), "--output", str(out),
                                 "--batch", "8", "--micro-batch", "1", "--checkpoint", checkpoint, "--bf16", "--msda", "hub",
                                 "--workers", workers, "--epochs", "50", "--epoch-policy", "fixed", "--eval-every", "5",
                                 "--no-final-d4", *resume], cwd=paths.REPO, env=env, stdout=fh, stderr=subprocess.STDOUT).returncode
        with open(log, "a") as fh:
            fh.write(f"EXIT {rc}\n")
        if rc == 0:
            (out / "last.pt").unlink(missing_ok=True)
            return 0
        if "OutOfMemoryError" in "".join(log.read_text(errors="ignore").splitlines(keepends=True)[-60:]):
            checkpoint = "all"
            with open(log, "a") as fh:
                fh.write("OOM_FALLBACK checkpoint=all (exact recomputation, slower)\n")
    return 1


if __name__ == "__main__":
    sys.exit(main())
