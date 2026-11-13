"""Mask2Former training data: the user's folds, native-resolution crops and whole frames, photometric and geometric
augmentation, and the input preparation of each model family.

FILAMENT_PREP selects the input: "raw" (the grayscale photo, family mask2former) or "limb3" (raw, the limb-darkening
corrected disk and its CLAHE, family mask2former_limb). FILAMENT_SCALE_JITTER zooms the training crops by a factor in
[1/j, j] (1 = off)."""
from __future__ import annotations

import json
import os
import math
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from pycocotools import mask as mu
from torch.utils.data import Dataset, Sampler

from src.data.split import assert_disjoint, make_split

NATIVE = 2048
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
EXPECTED = dict(train=974, val=180, all=1154)   # measured 2026-09-17 from make_split


def _to_bytes(rle: dict) -> dict:
    counts = rle["counts"]
    return dict(size=rle["size"], counts=counts.encode("ascii") if isinstance(counts, str) else counts)


def load_records(annotations: Path, fold: str) -> list[dict]:
    """Annotation records of one canonical fold, each with its instance RLEs.

    The split is `make_split`, exactly as `records_for` builds it, so training and
    the PQ sweep can never disagree about which photographs are held out.
    """
    coco = json.loads(Path(annotations).read_text(encoding="utf-8"))
    meta = {record["id"]: record for record in coco["images"]}
    by_image: dict = defaultdict(list)
    for annotation in coco["annotations"]:
        by_image[annotation["image_id"]].append(annotation)
    split = make_split(str(annotations))
    assert_disjoint(split)
    image_ids = dict(train=split.train_image_ids, val=split.val_image_ids,
                     all=split.train_image_ids + split.val_image_ids)[fold]
    if not os.environ.get("FILAMENT_FOLDS_JSON") and len(image_ids) != EXPECTED[fold]:
        raise SystemExit(f"canonical {fold} fold has {len(image_ids)} records, "
                         f"expected {EXPECTED[fold]}; the split has changed")
    records = []
    for image_id in image_ids:
        rles = []
        for annotation in by_image.get(image_id, []):
            segmentation = annotation.get("segmentation")
            if not segmentation:
                continue
            if isinstance(segmentation, list):
                rings = [ring for ring in segmentation if len(ring) >= 6]
                if not rings:
                    continue
                rle = mu.merge(mu.frPyObjects(rings, NATIVE, NATIVE))
            elif isinstance(segmentation["counts"], list):
                rle = mu.frPyObjects(segmentation, NATIVE, NATIVE)
            else:
                rle = segmentation
            rles.append(_to_bytes(rle))
        if rles:
            records.append(dict(image_id=str(image_id),
                                file_name=meta[image_id]["file_name"], rles=rles,
                                boxes=[mu.toBbox(rle).tolist() for rle in rles]))
    return records


def pooled_targets(records: list[dict]) -> list[dict]:
    """One record per photograph, holding a filament whenever any of its annotators drew it (an alternative
    training target, --targets pooled; the pipeline trains on one sample per annotation record).

    Here a photograph's filaments are grouped by
    overlap across annotators, and each group keeps one annotator's version, the one with the highest
    mean IoU with the group's other versions (ties: the larger). Groups never overlap, so kept
    instances overlap only where one annotator's own filaments do (5 of the 974 training records; 2 of
    the 601 pooled photographs keep such a pair, as the per-record targets already did).
    Measured on validation: these targets, used as predictions, score pooled 0.8085 and per record
    0.7719. The training fold goes from 974 records / 6,874 instances to 601 photographs / 4,667.
    """
    by_name: dict = defaultdict(list)
    for record in records:
        by_name[record["file_name"]].append(record)
    out = []
    for name, group in by_name.items():
        if len(group) == 1:
            out.append(group[0])
            continue
        items = [(owner, rle) for owner, record in enumerate(group) for rle in record["rles"]]
        rles = [rle for _, rle in items]
        iou = np.asarray(mu.iou(rles, rles, [0] * len(rles))).reshape(len(rles), len(rles))
        parent = list(range(len(items)))

        def find(i: int) -> int:
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                if items[i][0] != items[j][0] and iou[i, j] > 0:
                    parent[find(i)] = find(j)
        members: dict = defaultdict(list)
        for i in range(len(items)):
            members[find(i)].append(i)

        def agreement(i: int, group_members: list[int]) -> tuple[float, int]:
            others = [iou[i, j] for j in group_members if items[j][0] != items[i][0]]
            return (float(np.mean(others)) if others else 0.0, int(mu.area(rles[i])))

        kept = [rles[max(m, key=lambda i: agreement(i, m))] for m in members.values()]
        out.append(dict(image_id="+".join(record["image_id"] for record in group), file_name=name, rles=kept,
                        boxes=[mu.toBbox(rle).tolist() for rle in kept]))
    return out


PREP = os.environ.get("FILAMENT_PREP", "raw")              # "raw" (grayscale) or "limb3" (3 prepared channels)
SCALE_JITTER = float(os.environ.get("FILAMENT_SCALE_JITTER", "1"))      # crops zoomed by s in [1/j, j]; 1 = off


def read_raw(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None or image.shape != (NATIVE, NATIVE):
        raise FileNotFoundError(f"unreadable or not {NATIVE}x{NATIVE}: {path}")
    return image


def disk_geometry(gray: np.ndarray):
    """(cx, cy, radius) of the solar disk: Hough circle of the limb on a 512 px blurred copy, radius 820-960 px
    (all 887 train + test photos: 887-913 px, each checked on the limb). None when no circle is found."""
    small = cv2.GaussianBlur(cv2.resize(gray, (512, 512), interpolation=cv2.INTER_AREA), (5, 5), 1.5)
    c = cv2.HoughCircles(small, cv2.HOUGH_GRADIENT, dp=1, minDist=400, param1=60, param2=20, minRadius=205, maxRadius=240)
    if c is None:
        return None
    x, y, r = (float(v) * NATIVE / 512 for v in c[0][0])
    return x, y, r


def flatten_disk(gray: np.ndarray, bins: int = 96) -> np.ndarray:
    """Limb-darkening correction: the disk divided by its radial median profile (robust to filaments and plage),
    mapped to uint8 with 1.0 -> 128 and +-0.5 -> 0 / 255; zero off the disk (r > 1.01)."""
    g = disk_geometry(gray)
    if g is None:
        return gray.copy()
    cx, cy, R = g
    yy, xx = np.mgrid[0:NATIVE, 0:NATIVE].astype(np.float32)
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / R
    sub = (slice(None, None, 4), slice(None, None, 4))
    rs, vs = r[sub].ravel(), gray[sub].ravel().astype(np.float32)
    inside = rs <= 1.0
    rs, vs = rs[inside], vs[inside]
    edges = np.linspace(0, 1.0, bins + 1)
    idx = np.clip(np.digitize(rs, edges) - 1, 0, bins - 1)
    centres, prof = [], []
    for b in range(bins):
        v = vs[idx == b]
        if len(v) >= 50:
            centres.append((edges[b] + edges[b + 1]) / 2); prof.append(np.median(v))
    if len(prof) < bins // 2:
        return gray.copy()
    prof = np.convolve(np.pad(prof, 2, mode="edge"), np.ones(5) / 5, mode="valid")
    background = np.interp(r, centres, prof).astype(np.float32)
    flat = gray.astype(np.float32) / np.maximum(background, 1.0)
    out = np.clip(128.0 + (flat - 1.0) * 255.0, 0, 255).astype(np.uint8)
    out[r > 1.01] = 0
    return out


_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))


def prepare(gray: np.ndarray) -> np.ndarray:
    """limb3 input: [raw, limb-flattened disk, CLAHE of the flattened disk] as uint8 (H, W, 3)."""
    if PREP == "raw":
        return gray
    flat = flatten_disk(gray)
    return np.ascontiguousarray(np.stack([gray, flat, _CLAHE.apply(flat)], axis=-1))


def read_gray(path: Path) -> np.ndarray:
    """The model input of a photograph: (H, W) uint8 with FILAMENT_PREP=raw, (H, W, 3) uint8 with limb3."""
    return prepare(read_raw(path))


def to_pixels(gray: np.ndarray) -> torch.Tensor:
    """Grayscale uint8 or float in [0, 255] -> ImageNet-normalised 3-channel tensor (CPU reference)."""
    image = np.ascontiguousarray(gray).astype(np.float32) / 255.0
    image = np.repeat(image[None], 3, axis=0)
    return torch.from_numpy((image - MEAN) / STD)


def gpu_pixels(pixels: torch.Tensor, photometric: list | None, device: torch.device) -> torch.Tensor:
    """uint8 [B, H, W] -> ImageNet-normalised float32 [B, 3, H, W] on `device`.

    The first loader's formula in its operation order: 255 * (x / 255) ** gamma * gain,
    clipped to [0, 255], then / 255 and (x - mean) / std. Gamma and gain are float32,
    as NumPy applied them to a float32 image.
    """
    x = pixels.to(device, non_blocking=True).float()
    channels = x.dim() == 4                                   # limb3: (B, H, W, 3) prepared channels
    if photometric is not None and any(item is not None for item in photometric):
        if any(item is None for item in photometric):
            raise ValueError("a micro-batch mixes augmented and plain samples")
        shape = (-1, 1, 1, 1) if channels else (-1, 1, 1)
        gamma = torch.tensor([item[0] for item in photometric], dtype=torch.float32, device=device).view(*shape)
        gain = torch.tensor([item[1] for item in photometric], dtype=torch.float32, device=device).view(*shape)
        x = torch.clamp(255.0 * (x / 255.0) ** gamma * gain, 0, 255)
    x = (x / 255.0).permute(0, 3, 1, 2) if channels else (x / 255.0)[:, None].expand(-1, 3, -1, -1)
    return (x - torch.as_tensor(MEAN, device=device)) / torch.as_tensor(STD, device=device)


class ExtentBatchSampler(Sampler):
    """Optimiser steps of either `batch` crops or `whole_batch` whole frames, never mixed.

    Both kinds hold the same number of pixels. A step reaches the GPU as micro-batches
    that fit it: `micro` crops, or `whole_micro` frames. The step plan does not depend
    on the micro-batch, so the crop/whole mix, the records drawn and an epoch's cost
    are identical on an 80 GB and a 24 GB card; only the gradient accumulation differs.

    Every index is (record, extent, epoch, slot, last, parts). The epoch travels inside
    it because persistent DataLoader workers hold their own copy of the dataset and
    never see a `set_epoch`. `last` marks a step's final micro-batch and `parts` is the
    number of micro-batches in that step, which the trainer divides the loss by.
    """

    def __init__(self, records: int, crop: int, batch: int, whole_share: float,
                 micro: int | None = None, crops_per_record: int = 2, seed: int = 2026):
        micro = micro or batch
        if batch % micro:
            raise ValueError(f"micro-batch {micro} must divide the step batch {batch}")
        self.records, self.crop, self.batch, self.micro = records, crop, batch, micro
        self.whole_batch = max(1, round(batch * (crop / NATIVE) ** 2))
        self.whole_micro = max(1, micro * self.whole_batch // batch)
        self.whole_share, self.seed, self.epoch = whole_share, seed, 0
        self.steps = (records * crops_per_record) // batch
        self.whole_steps = round(self.steps * whole_share)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        crop_parts = math.ceil(self.batch / self.micro)
        whole_parts = math.ceil(self.whole_batch / self.whole_micro)
        return (self.steps - self.whole_steps) * crop_parts + self.whole_steps * whole_parts

    def __iter__(self):
        rng = np.random.default_rng([self.seed, self.epoch, 7])
        kinds = np.array([True] * self.whole_steps + [False] * (self.steps - self.whole_steps))
        rng.shuffle(kinds)
        stream: list[int] = []
        slot = 0
        for whole in kinds:
            size, micro = (self.whole_batch, self.whole_micro) if whole else (self.batch, self.micro)
            while len(stream) < size:
                stream.extend(rng.permutation(self.records).tolist())
            chosen, stream = stream[:size], stream[size:]
            extent = NATIVE if whole else self.crop
            parts = math.ceil(size / micro)
            for part in range(parts):
                batch = []
                for record in chosen[part * micro:(part + 1) * micro]:
                    batch.append((record, extent, self.epoch, slot, part == parts - 1, parts))
                    slot += 1
                yield batch


class FilamentSamples(Dataset):
    """Native-resolution crops and whole frames with exact dihedral augmentation.

    Faster than the first loader (`tests/reference_data.py`) without changing a sample,
    which the equality test checks key by key:
    - Each photograph is decoded once. With `preload` that happens before DataLoader
      workers fork, so every worker shares one copy of the pixels.
    - Each instance mask is decoded from its RLE once, at full size, and kept as the
      crop of its bounding box. A sample pastes those crops into place instead of
      decoding a 2048x2048 mask per instance per sample.
    - A sample leaves the worker as uint8 pixels plus its gamma and gain. `gpu_pixels`
      applies the photometric step and the normalisation on the GPU.
    Random draws happen in the reference order, so a key gives the same crop, the same
    flips and turns, and the same gamma and gain.
    """

    def __init__(self, records: list[dict], images: Path, min_piece: int = 40,
                 seed: int = 2026, augment: bool = True, rotate90: bool = True,
                 preload: bool = True, cache: int = 64, threads: int = 8):
        from concurrent.futures import ThreadPoolExecutor
        self.records, self.images = records, Path(images)
        self.min_piece, self.seed, self.augment, self.rotate90 = min_piece, seed, augment, rotate90
        self.preload, self.cache_size = preload, cache
        self._photos: dict[str, np.ndarray] = {}
        names = sorted({record["file_name"] for record in records})
        with ThreadPoolExecutor(max_workers=threads) as pool:
            if preload:
                self._photos = dict(zip(names, pool.map(lambda name: read_gray(self.images / name), names)))
            self._pieces = list(pool.map(self._record_pieces, records))

    @staticmethod
    def _record_pieces(record: dict) -> list[tuple[int, int, np.ndarray]]:
        """(x0, y0, bounding-box crop) per instance; an empty mask keeps a 0x0 crop."""
        pieces = []
        for rle in record["rles"]:
            mask = mu.decode(rle)
            rows = np.flatnonzero(mask.any(axis=1))
            columns = np.flatnonzero(mask.any(axis=0))
            if len(rows) == 0:
                pieces.append((0, 0, np.zeros((0, 0), np.uint8)))
                continue
            y0, y1, x0, x1 = int(rows[0]), int(rows[-1]) + 1, int(columns[0]), int(columns[-1]) + 1
            pieces.append((x0, y0, np.ascontiguousarray(mask[y0:y1, x0:x1])))
        return pieces

    def __len__(self) -> int:
        return len(self.records)

    def _gray(self, name: str) -> np.ndarray:
        image = self._photos.get(name)
        if image is None:
            image = read_gray(self.images / name)
            if not self.preload:
                if len(self._photos) >= self.cache_size:
                    self._photos.pop(next(iter(self._photos)))
                self._photos[name] = image
        return image

    @staticmethod
    def _window(box, crop: int, rng, jitter: float) -> tuple[int, int]:
        x, y, w, h = box
        cx = x + w / 2 + rng.uniform(-jitter, jitter) * crop
        cy = y + h / 2 + rng.uniform(-jitter, jitter) * crop
        x0 = int(np.clip(round(cx - crop / 2), 0, NATIVE - crop))
        y0 = int(np.clip(round(cy - crop / 2), 0, NATIVE - crop))
        return x0, y0

    def _masks(self, record_index: int, x0: int, y0: int, crop: int) -> list[np.ndarray]:
        threshold = self.min_piece if crop < NATIVE else 1
        kept = []
        for bx, by, piece in self._pieces[record_index]:
            left, top = max(bx, x0), max(by, y0)
            right, bottom = min(bx + piece.shape[1], x0 + crop), min(by + piece.shape[0], y0 + crop)
            if left >= right or top >= bottom:
                continue
            window = piece[top - by:bottom - by, left - bx:right - bx]
            if int(window.sum()) < threshold:
                continue
            canvas = np.zeros((crop, crop), np.uint8)
            canvas[top - y0:bottom - y0, left - x0:right - x0] = window
            kept.append(canvas)
        return kept

    def __getitem__(self, key):
        record_index, extent, epoch, slot, last, parts = key
        record = self.records[record_index]
        rng = np.random.default_rng([self.seed, epoch, slot, record_index])
        gray = self._gray(record["file_name"])
        if extent >= NATIVE:
            x0 = y0 = 0
            masks = self._masks(record_index, 0, 0, NATIVE)
        else:
            masks = []
            # The anchor filament's centre is always inside its own crop, so a retry
            # happens only when that filament is under `min_piece` pixels.
            zoom = math.exp(rng.uniform(-math.log(SCALE_JITTER), math.log(SCALE_JITTER))) \
                if self.augment and SCALE_JITTER > 1 else 1.0
            window = min(NATIVE, int(round(extent / zoom)))
            for jitter in (0.35, 0.2, 0.0):
                anchor = record["boxes"][int(rng.integers(len(record["boxes"])))]
                x0, y0 = self._window(anchor, window, rng, jitter)
                masks = self._masks(record_index, x0, y0, window)
                if window != extent and masks:
                    masks = [m for m in (cv2.resize(m, (extent, extent), interpolation=cv2.INTER_NEAREST)
                                         for m in masks) if int(m.sum()) >= self.min_piece]
                if masks:
                    break
        if not masks:
            raise RuntimeError(f"no filament in sample of {record['image_id']} at {extent}")
        if extent >= NATIVE:
            patch = gray[y0:y0 + extent, x0:x0 + extent]
        else:
            patch = gray[y0:y0 + window, x0:x0 + window]
            if window != extent:
                patch = cv2.resize(patch, (extent, extent),
                                   interpolation=cv2.INTER_LINEAR if window < extent else cv2.INTER_AREA)
        stack = np.stack(masks)
        photometric = None
        if self.augment:
            if rng.random() < 0.5:
                patch, stack = patch[:, ::-1], stack[:, :, ::-1]
            if rng.random() < 0.5:
                patch, stack = patch[::-1, :], stack[:, ::-1, :]
            if self.rotate90:
                turns = int(rng.integers(4))
                if turns:
                    patch, stack = np.rot90(patch, turns), np.rot90(stack, turns, axes=(1, 2))
            gamma = math.exp(rng.uniform(math.log(0.8), math.log(1.25)))
            gain = rng.uniform(0.85, 1.15)
            photometric = (gamma, gain)
        return (torch.from_numpy(np.ascontiguousarray(patch)),
                torch.from_numpy(np.ascontiguousarray(stack)),
                torch.zeros(len(stack), dtype=torch.long), (bool(last), int(parts), int(extent)),
                photometric, record["file_name"])


def collate(batch):
    """Stack uint8 images; masks and classes stay lists, as Mask2Former's loss expects.

    Returns (pixels uint8 [B, H, W], masks, classes, step metadata, photometric list, photo names).
    The step metadata (last, parts, extent) is shared by every sample of a micro-batch.
    """
    pixels = torch.stack([item[0] for item in batch])
    return (pixels, [item[1] for item in batch], [item[2] for item in batch], batch[0][3],
            [item[4] for item in batch], [item[5] for item in batch])
