"""Compressed storage of the U-Net maps (lossless).

A map set is what one U-Net predicts for a list of photos: per photo, 2 channels (filament probability, spine) of
2048 x 2048 uint8. It is stored as one zip file, ``<name>.zip``: one deflate-compressed member per photo and channel
(``<j>_<c>.u8``, raw bytes) and ``files.json`` (the photo names, in order). The maps are sparse (1-3 % of the pixels
are non-zero), so a photo takes ~0.1 MB instead of 8.4 MB, and a single photo / channel is read without touching the
rest. Values are exactly those predicted: every result computed from them is unchanged.

    with MapWriter(path, names) as w:          # written to <path>.part, renamed to <path> when complete
        w.write(j, maps)                       # maps: (2, 2048, 2048) uint8, photo j of names
    m = open_maps(path)                        # m.files, len(m), m[j] -> (2, H, W), m.channel(j, c) -> (H, W)
"""
from __future__ import annotations

import json
import os
import zipfile
from pathlib import Path

import numpy as np

H = W = 2048
CHANNELS = 2
LEVEL = 6                                    # zlib level: ~0.15 s per photo, 50-130x smaller than the raw maps


def map_path(folder, name: str) -> Path:
    """<folder>/<name>.zip, e.g. map_path(scratch / 'maps/train_f0', 'testcnxL')."""
    return Path(folder) / f"{name}.zip"


class MapWriter:
    """Writes a map set photo by photo; the file appears under its final name only once every photo is written."""

    def __init__(self, path, files: list[str]):
        self.path, self.files = Path(path), list(files)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.tmp = self.path.with_name(self.path.name + ".part")
        self.zf = zipfile.ZipFile(self.tmp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=LEVEL)
        self.done = set()

    def write(self, j: int, maps: np.ndarray) -> None:
        maps = np.ascontiguousarray(maps, dtype=np.uint8)
        assert maps.shape == (CHANNELS, H, W), maps.shape
        for c in range(CHANNELS):
            self.zf.writestr(f"{j}_{c}.u8", maps[c].tobytes())
        self.done.add(j)

    def close(self) -> None:
        missing = set(range(len(self.files))) - self.done
        if missing:
            self.zf.close(); self.tmp.unlink(missing_ok=True)
            raise RuntimeError(f"{self.path}: {len(missing)} photos not written")
        self.zf.writestr("files.json", json.dumps({"files": self.files}))
        self.zf.close()
        os.replace(self.tmp, self.path)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            self.close()
        else:
            self.zf.close(); self.tmp.unlink(missing_ok=True)


class MapSet:
    """Read access to a map set written by MapWriter."""

    def __init__(self, path):
        self.path = Path(path)
        self.zf = zipfile.ZipFile(self.path, "r")
        self.files = json.loads(self.zf.read("files.json"))["files"]

    def __len__(self) -> int:
        return len(self.files)

    def channel(self, j: int, c: int) -> np.ndarray:
        """Channel c (0: filament probability, 1: spine) of photo j, (H, W) uint8."""
        return np.frombuffer(self.zf.read(f"{j}_{c}.u8"), np.uint8).reshape(H, W)

    def __getitem__(self, j: int) -> np.ndarray:
        return np.stack([self.channel(j, c) for c in range(CHANNELS)])

    def close(self) -> None:
        self.zf.close()


def open_maps(path) -> MapSet:
    return MapSet(path)


def map_files(path) -> list[str]:
    """The photo names of a map set, in order."""
    with zipfile.ZipFile(path, "r") as zf:
        return json.loads(zf.read("files.json"))["files"]


def convert(maps_root) -> None:
    """Convert raw map sets (<unit>/test<src>.npy + .npy.json, the earlier format) under maps_root into compressed
    ones; each raw pair is deleted once its zip is complete. Values are copied unchanged.
        python -m src.utils.mapstore convert <scratch>/maps"""
    for raw in sorted(Path(maps_root).rglob("*.npy")):
        meta = Path(str(raw) + ".json")
        if not meta.is_file():
            continue
        out = raw.with_suffix(".zip")
        if not out.is_file():
            arr = np.load(raw, mmap_mode="r")
            files = json.loads(meta.read_text())["files"]
            with MapWriter(out, files) as w:
                for j in range(len(files)):
                    w.write(j, arr[j])
            del arr
        print(f"{raw} ({raw.stat().st_size / 1e9:.1f} GB) -> {out} ({out.stat().st_size / 1e9:.2f} GB)", flush=True)
        raw.unlink(); meta.unlink()


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 3 and sys.argv[1] == "convert":
        convert(sys.argv[2])
    else:
        raise SystemExit("usage: python -m src.utils.mapstore convert <scratch>/maps")
