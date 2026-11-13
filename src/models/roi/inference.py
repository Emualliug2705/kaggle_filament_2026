"""ROI detector + refiner inference: a Faster R-CNN v2 detector on 9 tiles of each photo, global NMS, then a ResNet18 U-Net
refiner on a 512 px context crop around each box; each refined mask becomes a candidate (rle, score, area, distance
from the disk centre, 24 features). One JSON per photo plus manifest.json (run). Settings: configs/roi/inference.json."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import random
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as coco

from src import paths

CONFIG = paths.CONFIGS / 'roi' / 'inference.json'
FEATURES = [
    "logit_confidence", "logit_detector", "logit_mask_mean", "log_area",
    "log_width", "log_height", "log_aspect", "fill", "rho", "rho_squared",
    "raw_mean", "raw_std", "ring_contrast", "contrast_z", "mask_std",
    "mask_q10", "stable_fraction", "perimeter_over_sqrt_area",
    "small", "limb", "confidence_x_small", "confidence_x_limb",
    "contrast_x_limb", "area_squared",
]
BANK_DIR = 'banks/roi-test'


def load_config(path=CONFIG) -> dict:
    return json.loads(Path(path).read_text(encoding='utf-8'))


# ------------------------------------------------------------------------------------------ files and digests

def sha256(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, value):
    """JSON with indent 2, sorted keys and a final newline, written through a temporary file (``gpu/runtime.py``)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent, delete=False) as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n')
        temporary = Path(stream.name)
    temporary.replace(path)


def state_digest(value) -> str:
    import torch
    digest = hashlib.sha256()

    def visit(obj):
        if isinstance(obj, torch.Tensor):
            array = obj.detach().cpu().contiguous()
            digest.update(str((array.dtype, tuple(array.shape))).encode())
            digest.update(array.numpy().tobytes())
        elif isinstance(obj, dict):
            for key in sorted(obj, key=str):
                digest.update(str(key).encode())
                visit(obj[key])
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                visit(item)
        else:
            digest.update(repr(obj).encode())
    visit(value)
    return digest.hexdigest()


def load_image(path) -> np.ndarray:
    """A photo as uint8 [2048, 2048], decoded by Pillow."""
    from PIL import Image
    with Image.open(path) as image:
        if image.mode != 'L' or image.size != (2048, 2048):
            raise ValueError(f'expected a 2048 x 2048 grayscale image: {path}')
        return np.asarray(image).copy()


# ------------------------------------------------------------------------------------ ROI geometry (g4_data.py)

def box_iou(a, b):
    a = np.asarray(a, dtype=np.float32).reshape(-1, 4); b = np.asarray(b, dtype=np.float32).reshape(-1, 4)
    intersection = np.maximum(0, np.minimum(a[:, None, 2:], b[None, :, 2:]) - np.maximum(a[:, None, :2], b[None, :, :2])).prod(2)
    return intersection / np.maximum(1e-8, (a[:, 2:] - a[:, :2]).prod(1)[:, None] + (b[:, 2:] - b[:, :2]).prod(1)[None, :] - intersection)


def nms(boxes, scores, threshold=.7, limit=128):
    """Stable CPU NMS; original order breaks exact score ties."""
    order = sorted(range(len(boxes)), key=lambda i: (-float(scores[i]), i)); keep = []
    while order and len(keep) < limit:
        i = order.pop(0); keep.append(i)
        if order:
            overlap = box_iou([boxes[i]], np.asarray(boxes)[order])[0]
            order = [j for j, v in zip(order, overlap) if v <= threshold]
    return keep


def roi_bounds(box, shape, factor=1.5, minimum=128):
    x1, y1, x2, y2 = map(float, box)
    side = min(min(shape), max(minimum, int(math.ceil(max(x2 - x1, y2 - y1) * factor))))
    left = int(np.clip(round((x1 + x2 - side) / 2), 0, shape[1] - side))
    top = int(np.clip(round((y1 + y2 - side) / 2), 0, shape[0] - side))
    return top, left, side


def roi_input(raw, box, size=512, factor=1.5, minimum=128):
    top, left, side = roi_bounds(box, raw.shape, factor, minimum); hint = np.zeros((side, side), np.float32)
    x1, y1, x2, y2 = map(float, box)
    xa, ya = max(0, int(np.floor(x1 - left))), max(0, int(np.floor(y1 - top)))
    xb, yb = min(side, int(np.ceil(x2 - left))), min(side, int(np.ceil(y2 - top)))
    hint[ya:yb, xa:xb] = 1
    image = cv2.resize(raw[top:top + side, left:left + side].astype(np.float32) / 255, (size, size), interpolation=cv2.INTER_LINEAR)
    hint = cv2.resize(hint, (size, size), interpolation=cv2.INTER_NEAREST)
    return np.stack([image, hint]), (top, left, side)


def selected_component(binary, hint):
    """Select most box-supported component, then largest; scan order breaks ties."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8), 8)
    if count == 1:
        return np.zeros(binary.shape, bool)
    support = np.bincount(labels[hint.astype(bool)].ravel(), minlength=count)
    chosen = max(range(1, count), key=lambda i: (int(support[i]), int(stats[i, cv2.CC_STAT_AREA]), -i))
    return labels == chosen


# ------------------------------------------------------------------------- candidate features (g6_quality.py)

def encode_mask(mask):
    assert mask.ndim == 2 and mask.dtype == np.bool_
    value = coco.encode(np.asfortranarray(mask, dtype=np.uint8))
    return {'size': list(value['size']), 'counts': value['counts'].decode('ascii')}


def estimate_disk(image):
    """Estimate a sampling region only. Never use this as a prediction veto."""
    height, width = image.shape
    small = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
    threshold = max(5.0, float(np.percentile(small, 95)) * 0.1)
    binary = (small > threshold).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("Cannot locate solar disk")
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < 0.2 * small.size:
        raise ValueError("Solar-disk estimate too small; inspect image")
    (x, y), radius = cv2.minEnclosingCircle(contour)
    return {"center_x": x * width / 256, "center_y": y * height / 256,
            "radius": radius * min(height, width) / 256}


def logit(p):
    p = np.clip(p, 1e-5, 1 - 1e-5)
    return np.log(p / (1 - p))


def candidates(raw, boxes, scores, probabilities, threshold, factor=1.5, minimum=128):
    """Image-local features use predicted masks only; retain raw pixels."""
    geometry = estimate_disk(raw)
    result = []
    for i, (box, detector, probability) in enumerate(zip(boxes, scores, probabilities)):
        top, left, side = roi_bounds(box, raw.shape, factor, minimum)
        p = cv2.resize(probability.astype(np.float32), (side, side), interpolation=cv2.INTER_LINEAR)
        hint = np.zeros((side, side), np.uint8)
        x1, y1, x2, y2 = box
        hint[max(0, int(np.floor(y1 - top))):min(side, int(np.ceil(y2 - top))),
             max(0, int(np.floor(x1 - left))):min(side, int(np.ceil(x2 - left)))] = 1
        mask = selected_component(p >= threshold, hint)
        area = int(mask.sum())
        if not area:
            continue
        yy, xx = np.nonzero(mask)
        w, h = int(xx.max() - xx.min() + 1), int(yy.max() - yy.min() + 1)
        rho = math.hypot(left + (xx.min() + xx.max() + 1) / 2 - geometry["center_x"],
                         top + (yy.min() + yy.max() + 1) / 2 - geometry["center_y"]) / geometry["radius"]
        crop = raw[top:top + side, left:left + side].astype(np.float32) / 255
        ring = cv2.dilate(mask.astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool) & ~mask
        inside, outside = crop[mask], crop[ring]
        contrast = float(np.median(outside) - inside.mean()) if len(outside) else 0.
        z = contrast / max(float(outside.std()), .02) if len(outside) else 0.
        mp = p[mask]
        confidence = float(detector) * float(mp.mean())
        lc = float(logit(confidence))
        boundary = mask & ~cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        small, limb = float(area < 256), float(rho >= .8)
        la = math.log1p(area)
        features = [lc, float(logit(detector)), float(logit(mp.mean())), la,
                    math.log1p(w), math.log1p(h), math.log(w / h), area / (w * h),
                    rho, rho * rho, float(inside.mean()), float(inside.std()), contrast,
                    float(np.clip(z, -10, 10)), float(mp.std()), float(np.quantile(mp, .1)),
                    float((mp >= min(.95, threshold + .1)).mean()),
                    float(boundary.sum() / math.sqrt(area)), small, limb, lc * small, lc * limb,
                    contrast * limb, la * la]
        full = np.zeros(raw.shape, bool)
        full[top:top + side, left:left + side] = mask
        result.append({"index": i, "score": confidence, "area": area, "rho": rho,
                       "features": features, "rle": encode_mask(full)})
    return result


# ------------------------------------------------------------------------------------------- models (gpu/model.py)

def _modules():
    import torch
    from torch import nn
    from torch.nn import functional as F
    from torchvision.models import resnet18

    class ConvBlock(nn.Sequential):
        def __init__(self, incoming, outgoing):
            super().__init__(nn.Conv2d(incoming, outgoing, 3, padding=1, bias=False), nn.GroupNorm(8, outgoing), nn.ReLU(),
                             nn.Conv2d(outgoing, outgoing, 3, padding=1, bias=False), nn.GroupNorm(8, outgoing), nn.ReLU())

    class ResNetUNet(nn.Module):
        """ResNet18 U-Net. Built without ImageNet weights; load the trained refiner weights after."""

        def __init__(self, channels=1):
            super().__init__()
            if channels < 1:
                raise ValueError("At least one raw channel is required")
            self.encoder = resnet18(weights=None)
            rgb_kernel = self.encoder.conv1.weight.detach().clone()
            self.encoder.conv1 = nn.Conv2d(1, 64, 7, stride=2, padding=3, bias=False)
            with torch.no_grad():
                self.encoder.conv1.weight.zero_()
                self.encoder.conv1.weight[:, :1].copy_(rgb_kernel.sum(dim=1, keepdim=True))
            self.encoder.fc = nn.Identity()
            self.dec4 = ConvBlock(512 + 256, 256)
            self.dec3 = ConvBlock(256 + 128, 128)
            self.dec2 = ConvBlock(128 + 64, 64)
            self.dec1 = ConvBlock(64 + 64, 32)
            self.dec0 = ConvBlock(33, 16)
            self.head = nn.Conv2d(16, 1, 1)
            if channels > 1:
                encoder_input = nn.Conv2d(channels, 64, 7, stride=2, padding=3, bias=False)
                decoder_input = nn.Conv2d(32 + channels, 16, 3, padding=1, bias=False)
                with torch.no_grad():
                    encoder_input.weight.zero_()
                    encoder_input.weight[:, :1].copy_(self.encoder.conv1.weight)
                    decoder_input.weight.zero_()
                    decoder_input.weight[:, :33].copy_(self.dec0[0].weight)
                self.encoder.conv1, self.dec0[0] = encoder_input, decoder_input

        def forward(self, image):
            raw = (image - 0.5) / 0.25
            e = self.encoder
            one = e.relu(e.bn1(e.conv1(raw)))
            two = e.layer1(e.maxpool(one))
            three = e.layer2(two)
            four = e.layer3(three)
            five = e.layer4(four)
            value = five
            for block, skip in ((self.dec4, four), (self.dec3, three), (self.dec2, two), (self.dec1, one), (self.dec0, raw)):
                value = block(torch.cat([F.interpolate(value, size=skip.shape[-2:], mode="nearest"), skip], dim=1))
            return self.head(value)

    return ResNetUNet


def build_refiner(cfg):
    return _modules()(cfg['refiner']['channels'])


def build_detector(cfg):
    from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
    from torchvision.models.detection.anchor_utils import AnchorGenerator
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    s = cfg['detector']
    model = fasterrcnn_resnet50_fpn_v2(weights=None, weights_backbone=None, num_classes=s['num_classes_built'],
                                       min_size=s['input'], max_size=s['input'], box_score_thresh=s['score_floor'],
                                       box_nms_thresh=s['per_tile_nms'], box_detections_per_img=s['detections_per_tile'])
    model.roi_heads.box_predictor = FastRCNNPredictor(model.roi_heads.box_predictor.cls_score.in_features, s['num_classes'])
    model.rpn.anchor_generator = AnchorGenerator(tuple((v,) for v in s['anchors']),
                                                 tuple(tuple(s['aspect_ratios']) for _ in range(5)))
    return model


def configure(device, seed):
    """Global torch settings (``seed_everything``, then non-deterministic algorithms allowed)."""
    if device == 'cuda':
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable')
        torch.cuda.set_device(0)
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(False)
    info = {'device': device, 'torch': torch.__version__, 'torchvision': _version('torchvision'),
            'cuda_runtime': torch.version.cuda, 'cudnn': torch.backends.cudnn.version()}
    if device == 'cuda':
        info['gpu'] = torch.cuda.get_device_name(0)
    return info


def load_models(cfg, device, tag):
    """The detector and refiner of models/roi_detector-<tag> and models/roi_refiner-<tag> (strict loading), in eval
    mode on device; tag is f0-f4 or full."""
    import torch
    models, lineage = {}, {}
    for stage, build in (('detector', build_detector), ('refiner', build_refiner)):
        model = build(cfg)
        payload = torch.load(paths.MODELS / f'roi_{stage}-{tag}' / 'weights.pt', map_location='cpu', weights_only=True)
        model.load_state_dict(payload['model'])                      # strict
        models[stage] = model.to(device).eval()
        lineage[stage] = {'file': f'models/roi_{stage}-{tag}/weights.pt', 'model_digest': state_digest(model.state_dict())}
    return models, lineage


# ------------------------------------------------------------------------------------------------------ inference

def _autocast(cuda):
    import contextlib
    import torch
    return torch.autocast('cuda', dtype=torch.bfloat16) if cuda else contextlib.nullcontext()


def _version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def infer(models, raw, cfg, device):
    """Boxes [K, 4] float32, detector scores [K] float32 and refiner probabilities [K, 512, 512] float32."""
    import torch
    cuda = device == 'cuda'
    dp, rf = cfg['proposal'], cfg['refiner']
    size = dp['tile_size']
    with torch.inference_mode():
        boxes = []; scores = []
        for top in dp['tile_starts']:
            for left in dp['tile_starts']:
                tile = np.repeat((raw[top:top + size, left:left + size].astype(np.float32) / 255)[None], 3, axis=0)
                with _autocast(cuda):
                    r = models['detector']([torch.from_numpy(tile).to(device)])[0]
                b = r['boxes'].float().cpu().numpy(); s = r['scores'].float().cpu().numpy()
                b += np.array([left, top, left, top], np.float32); boxes.extend(b.tolist()); scores.extend(s.tolist())
        boxes = np.asarray(boxes, np.float32).reshape(-1, 4); scores = np.asarray(scores, np.float32)
        keep = nms(boxes, scores, dp['global_box_nms'], dp['max_candidates']); boxes, scores = boxes[keep], scores[keep]
        probs = []
        step = rf['micro_batch']
        for start in range(0, len(boxes), step):
            batch = np.stack([roi_input(raw, b, rf['input'], rf['context_factor'], rf['minimum_context'])[0]
                              for b in boxes[start:start + step]])
            with _autocast(cuda):
                prob = models['refiner'](torch.from_numpy(batch).to(device)).float().sigmoid().cpu().numpy()[:, 0]
            probs.extend(prob)
    return boxes, scores, np.asarray(probs, np.float32).reshape(-1, rf['input'], rf['input'])


def record_of(name, source_sha256, raw, boxes, scores, prob, cfg):
    """The bank record of one image (the original's key set; ``predictions`` left empty)."""
    mask = cfg['mask_threshold']
    rf = cfg['refiner']
    digest = hashlib.sha256(boxes.tobytes() + scores.tobytes() + prob.tobytes()).hexdigest()
    views = {str(mask): candidates(raw, boxes, scores, prob, mask, rf['context_factor'], rf['minimum_context'])}
    assert all(np.isfinite(r['features']).all() and len(r['features']) == len(FEATURES) for r in views[str(mask)])
    return {'file_name': name, 'source_sha256': source_sha256, 'input_sha256': hashlib.sha256(raw.tobytes()).hexdigest(),
            'raw_predictions_sha256': digest, 'views': views, 'predictions': {}}


def run(tag, image_dir=None, out_dir=None, device='cuda', names=None, config=CONFIG, log=print):
    """Write the candidate bank of pair <tag> on the images in image_dir (default: the test photos) to out_dir (default
    ``<work>/banks/roi-test``): ``NNNN.json`` per image in sorted name order, then ``manifest.json``."""
    import torch
    t_process = time.monotonic()
    cfg = load_config(config)
    image_dir = Path(image_dir or paths.TEST_IMAGES)
    out_dir = Path(out_dir or paths.WORK / BANK_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    names = sorted(names if names is not None else (p.name for p in image_dir.glob('*.jpeg')))
    runtime = configure(device, cfg['seed'])
    tick = time.monotonic()
    models, lineage = load_models(cfg, device, tag)
    loads = time.monotonic() - tick
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    records = []; start = time.monotonic()
    for i, name in enumerate(names):
        tick = time.monotonic()
        path = image_dir / name
        raw = load_image(path)
        boxes, scores, prob = infer(models, raw, cfg, device)
        record = record_of(name, sha256(path), raw, boxes, scores, prob, cfg)
        file = out_dir / f'{i:04d}.json'
        atomic_json(file, record)
        records.append({'file_name': name, 'file': file.name, 'sha256': sha256(file), 'input_sha256': record['input_sha256'],
                        'raw_predictions_sha256': record['raw_predictions_sha256'],
                        'candidates': len(record['views'][str(cfg['mask_threshold'])]),
                        'base_seconds': time.monotonic() - tick})
        if i % 20 == 0:
            log(f'roi {i + 1}/{len(names)} {name} candidates {records[-1]["candidates"]} {records[-1]["base_seconds"]:.2f} s')
    result = {'passed': True, 'scope': 'test', 'names': names, 'records': records, 'lineage': lineage, 'runtime': runtime,
              'mask_threshold': cfg['mask_threshold'], 'model_load_seconds': loads, 'seconds': time.monotonic() - start,
              'process_seconds': time.monotonic() - t_process,
              'peak_mib': torch.cuda.max_memory_allocated() / 2 ** 20 if device == 'cuda' else None,
              'config_sha256': sha256(config), 'source_sha256': sha256(__file__),
              'versions': {k: _version(k) for k in ['numpy', 'opencv-python', 'opencv-python-headless', 'pillow',
                                                     'pycocotools', 'torch', 'torchvision']}}
    atomic_json(out_dir / 'manifest.json', result)
    return result
