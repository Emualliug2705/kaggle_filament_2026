"""Shape statistics of filament masks, for the morphology section of the report.

Compares the masks of the submitted file (test photographs, no labels needed) with the training annotations
(every annotator record). Per mask: area, connected components, mean width, length, elongation, solidity and
distance from the disk centre in solar radii.

    python morphology_stats.py SUBMISSION.csv TEST_IMAGES/ TRAIN_ANNOTATIONS.json TRAIN_IMAGES/ OUT_DIR/

Width and length: for a strip of uniform width w, the distance transform inside it is spread evenly over [0, w/2],
so w = 4 x mean distance; length = area / w; elongation = length / w.
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mu

SIZE = 2048


def disk(image_path: Path) -> tuple[float, float, float]:
    """(cx, cy, radius) of the solar disk: largest bright region of a blurred copy, radius from its area."""
    gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    blur = cv2.GaussianBlur(gray, (0, 0), 4)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return float(centroids[k][0]), float(centroids[k][1]), float(np.sqrt(stats[k, cv2.CC_STAT_AREA] / np.pi))


def shape(binary: np.ndarray, centre: tuple[float, float, float]) -> dict | None:
    ys, xs = np.nonzero(binary)
    if len(ys) == 0:
        return None
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    crop = np.pad(binary[y0:y1, x0:x1].astype(np.uint8), 2)
    area = int(crop.sum())
    components = cv2.connectedComponents(crop, connectivity=8)[0] - 1
    dist = cv2.distanceTransform(crop, cv2.DIST_L2, 5)
    width = max(1.0, 4.0 * float(dist[crop > 0].mean()))
    length = area / width
    contours, _ = cv2.findContours(crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    hull = cv2.contourArea(cv2.convexHull(np.concatenate(contours)))
    cx, cy, radius = centre
    r = float(np.hypot(xs.mean() - cx, ys.mean() - cy) / radius)
    return dict(area=area, components=int(components), width=width, length=length, elongation=length / width,
                solidity=area / max(hull, 1.0), r=r)


def _test_job(args):
    image, rles = args
    centre = disk(image)
    rows = []
    for rle in rles:
        row = shape(mu.decode(dict(size=[SIZE, SIZE], counts=rle.encode("ascii"))), centre)
        if row:
            rows.append(dict(image=image.stem, **row))
    return rows


def _train_job(args):
    image, annotations = args
    centre = disk(image)
    rows = []
    for record, segmentation in annotations:
        rings = [ring for ring in segmentation if len(ring) >= 6]
        if not rings:
            continue
        row = shape(mu.decode(mu.merge(mu.frPyObjects(rings, SIZE, SIZE))), centre)
        if row:
            rows.append(dict(image=image.stem, record=record, **row))
    return rows


def summary(rows: list[dict], per_image: dict[str, int]) -> dict:
    out = dict(masks=len(rows), per_image_median=float(np.median(list(per_image.values()))),
               per_image_mean=float(np.mean(list(per_image.values()))))
    for key in ("area", "width", "length", "elongation", "solidity", "r"):
        values = np.array([row[key] for row in rows])
        out[key] = dict(q25=float(np.percentile(values, 25)), median=float(np.median(values)),
                        q75=float(np.percentile(values, 75)))
    out["one_component"] = float(np.mean([row["components"] == 1 for row in rows]))
    out["beyond_0.9R"] = float(np.mean([row["r"] > 0.9 for row in rows]))
    out["elongation_above_5"] = float(np.mean([row["elongation"] > 5 for row in rows]))
    out["area_below_1000"] = float(np.mean([row["area"] < 1000 for row in rows]))
    return out


def main(submission: Path, test_images: Path, annotations: Path, train_images: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    by_test: dict[str, list[str]] = defaultdict(list)
    with open(submission, newline="") as handle:
        for row in csv.DictReader(handle):
            by_test[row["filament_id"].rsplit("_", 1)[0]].append(row["segmentation_rle"])
    coco = json.loads(annotations.read_text(encoding="utf-8"))
    files = {image["id"]: image["file_name"] for image in coco["images"]}
    by_train: dict[str, list] = defaultdict(list)
    for annotation in coco["annotations"]:
        record = annotation["image_id"]
        by_train[files[record]].append((record, annotation["segmentation"]))

    with ProcessPoolExecutor() as pool:
        test_rows = [row for rows in pool.map(_test_job, [(test_images / f"{stem}.jpeg", rles)
                                                           for stem, rles in sorted(by_test.items())])
                     for row in rows]
        train_rows = [row for rows in pool.map(_train_job, [(train_images / name, items)
                                                             for name, items in sorted(by_train.items())])
                      for row in rows]

    test_counts = {stem: len(rles) for stem, rles in by_test.items()}
    record_counts: dict[str, int] = defaultdict(int)
    for row in train_rows:
        record_counts[row["record"]] += 1
    result = dict(predicted_test=summary(test_rows, test_counts),
                  annotated_train=summary(train_rows, record_counts))
    (out / "morphology_summary.json").write_text(json.dumps(result, indent=1))
    for name, rows in (("predicted_test", test_rows), ("annotated_train", train_rows)):
        with open(out / f"{name}.csv", "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main(*map(Path, sys.argv[1:6]))
