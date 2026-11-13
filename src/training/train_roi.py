"""ROI detector (Faster R-CNN v2, COCO start) and refiner (ResNet18 U-Net, ImageNet start) training on the user's folds:
model f<k> trains on the photos of the other 4 folds (--fold -1: all 707 photos). Settings: configs/roi/training.json.
Resumable: a checkpoint every 1000 steps (the sampler is indexed by step, so a resume draws the same samples).

    python -m src.training.train_roi --prepare                      (once, before parallel trainings: the target cache)
    python -m src.training.train_roi --stage detector|refiner --fold K [--steps N] [--device cpu]

Weights: models/roi_<stage>-f<k>/weights.pt (+ run.json). The ROI code reads its configs, manifests, photos and target
cache through the workspace work/roi, prepared here.
"""
from pathlib import Path
import sys, os, argparse, time, math, json, shutil

from src import paths

ROOT = paths.WORK / 'roi'                                           # the ROI workspace (relative paths of the ROI code)
SHM = Path(os.environ.get('ROI_SCRATCH', str(paths.SCRATCH / 'roi_runs'))); DISK = Path(os.environ.get('ROI_MODELS', str(paths.MODELS)))
DET_URL = 'https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth'


def prepare_workspace():
    """work/roi: configs and manifests (links to configs/roi), the photos and annotations (links into the data folder)."""
    def link(target, name):
        name = ROOT / name; name.parent.mkdir(parents=True, exist_ok=True)
        if name.is_symlink() and Path(os.readlink(name)) == Path(target):
            return
        if name.is_symlink():
            name.unlink()
        name.symlink_to(target)
    link(paths.CONFIGS / 'roi', 'configs')
    link(paths.CONFIGS / 'roi' / 'manifests', 'manifests')
    link(paths.DATA, 'data/raw/MAGFiLO_1.0_Kaggle_2026')
    link(paths.ANNOTATIONS, 'data/annotations/MAGFiLO_1.0_Annotations_kaggle2026_train.json')
    (ROOT / 'checkpoints' / 'pretrained').mkdir(parents=True, exist_ok=True)
    os.chdir(ROOT)


def own_index(fold):
    from src.models.roi.dataset import ObservationIndex
    idx = ObservationIndex(validation_fold=0)
    uf = {n: k for k, names in enumerate(json.load(open(paths.FOLDS_JSON))['folds']) for n in names}
    assert set(uf) == set(idx.images), 'user folds must cover the 707 training photos'
    idx.development_names = sorted(idx.images)                    # every photo usable (the manifest's holdout is not used)
    idx.train_names = sorted(n for n in idx.images if uf[n] != fold)        # fold -1: all 707 photos (test model)
    idx.validation_names = sorted(n for n in idx.images if uf[n] == fold)
    idx.inner_names = []; idx.predict_names = []
    return idx


def prepare_targets():
    """Rasterise every annotator record into the target cache (work/roi/data/targets) in one process. Trainings run in
    parallel would otherwise build the same cache files at the same time."""
    prepare_workspace()
    from src.models.roi.dataset import AnnotationCache
    idx = own_index(0)
    cache = AnnotationCache(idx)
    n = 0
    for name in sorted(idx.images):
        for record_id in idx.images[name]['annotator_image_ids']:
            cache.load(record_id); n += 1
    print(json.dumps({'target_cache': str(cache.root), 'records': n}), flush=True)


def main():
    a = argparse.ArgumentParser(); a.add_argument('--stage', choices=['detector', 'refiner'])
    a.add_argument('--fold', type=int); a.add_argument('--steps', type=int, default=0)
    a.add_argument('--prepare', action='store_true', help='build the target cache, then exit')
    a.add_argument('--device', default='cuda'); a.add_argument('--seed', type=int, default=20261014); args = a.parse_args()
    if args.prepare:
        return prepare_targets()
    if args.stage is None or args.fold is None:
        a.error('--stage and --fold are required')
    prepare_workspace()
    from src.models.roi.common import read_json
    from src.models.roi.schedule import factor
    from src.models.roi.gpu.runtime import seed_everything, state_digest, rng_state
    from src.models.roi.gpu.model import ResNetUNet, prepare_weights, segmentation_loss
    from src.models.roi.g4_data import G4Samples
    import numpy as np
    import torch
    from torch import nn
    from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.anchor_utils import AnchorGenerator
    s = dict(read_json('configs/training.json')[args.stage]); steps = args.steps or s['steps']
    dev = torch.device(args.device)
    if dev.type == 'cuda':
        torch.cuda.set_device(0)
    seed_everything(args.seed); torch.use_deterministic_algorithms(False)
    idx = own_index(args.fold)
    tag = (f'roi_{args.stage}-f{args.fold}' if args.fold >= 0 else f'roi_{args.stage}-full') + (f'-smoke{steps}' if args.steps else '')
    d = SHM / tag; d.mkdir(parents=True, exist_ok=True)
    final = DISK / tag / 'weights.pt'
    if final.is_file():
        print(json.dumps({'tag': tag, 'status': 'already complete'})); return
    weights18, _ = prepare_weights()
    det_path = ROOT / 'checkpoints/pretrained' / Path(DET_URL).name
    if args.stage == 'detector' and not det_path.is_file():
        torch.hub.download_url_to_file(DET_URL, str(det_path), hash_prefix='dd69338a', progress=False)
    t0 = time.time(); data = G4Samples(idx, args.stage, seed=args.seed)
    print(json.dumps({'tag': tag, 'bank_photos': len(data.names), 'bank_seconds': round(time.time() - t0), 'held_out': len(idx.validation_names)}), flush=True)
    if args.stage == 'refiner':
        model = ResNetUNet(2, weights18)
    else:
        model = fasterrcnn_resnet50_fpn_v2(weights=None, weights_backbone=None, num_classes=91, min_size=1024, max_size=1024,
                                           box_score_thresh=.01, box_nms_thresh=.7, box_detections_per_img=128)
        model.load_state_dict(torch.load(det_path, map_location='cpu', weights_only=True))
        model.roi_heads.box_predictor = FastRCNNPredictor(model.roi_heads.box_predictor.cls_score.in_features, 2)
        model.rpn.anchor_generator = AnchorGenerator(tuple((v,) for v in s['anchors']), tuple(tuple(s['aspect_ratios']) for _ in range(5)))
    model = model.to(dev); model.train()
    if args.stage == 'detector':
        for module in model.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm): module.eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=s['learning_rate'], weight_decay=.0001)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda t: factor(t, s))
    first = 1
    if (d / 'checkpoint.pt').is_file():
        ck = torch.load(d / 'checkpoint.pt', map_location='cpu', weights_only=False)
        model.load_state_dict(ck['model']); optimizer.load_state_dict(ck['optimizer']); scheduler.load_state_dict(ck['scheduler'])
        first = ck['step'] + 1; print(json.dumps({'tag': tag, 'resumed_at': ck['step']}), flush=True)
    run = {'stage': args.stage, 'fold': args.fold, 'seed': args.seed, 'steps': steps, 'spec': s, 'fit_names': idx.train_names,
           'held_out_names': idx.validation_names, 'started': time.time()}
    (d / 'run.json').write_text(json.dumps(run))
    batch, micro = s['effective_batch'], s['micro_batch']; t_start = time.time(); amp = dev.type == 'cuda'
    for step in range(first, steps + 1):
        optimizer.zero_grad(set_to_none=True); value = 0.
        for offset in range(0, batch, micro):
            samples = [data.sample((step - 1) * batch + offset + j) for j in range(micro)]
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=amp):
                if args.stage == 'detector':
                    x = [torch.from_numpy(v[0]).to(dev) for v in samples]
                    targets = [{k: torch.from_numpy(vv).to(dev) for k, vv in v[1].items()} for v in samples]
                    loss = sum(model(x, targets).values())
                else:
                    x = torch.from_numpy(np.stack([v[0] for v in samples])).to(dev)
                    target = torch.from_numpy(np.stack([v[1] for v in samples])).to(dev)
                    loss = segmentation_loss(model(x), target)
            assert torch.isfinite(loss).item(); (loss * micro / batch).backward(); value += float(loss.detach()) * micro / batch
        norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)); assert math.isfinite(norm)
        optimizer.step(); scheduler.step()
        if step % 250 == 0 or step == steps or step == first:
            el = time.time() - t_start
            print(json.dumps({'tag': tag, 'step': step, 'loss': round(value, 4), 'sec_per_step': round(el / (step - first + 1), 3),
                              'eta_min': round(el / (step - first + 1) * (steps - step) / 60)}), flush=True)
        if step % 1000 == 0 and step < steps:
            tmp = d / 'checkpoint.partial'
            torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                        'rng': rng_state(), 'step': step}, tmp); tmp.replace(d / 'checkpoint.pt')
    final.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'model': model.state_dict(), 'step': steps, 'stage': args.stage, 'fold': args.fold,
                'digest': state_digest(model.state_dict())}, final.with_suffix('.partial'))
    final.with_suffix('.partial').replace(final)
    (DISK / tag / 'run.json').write_text(json.dumps(dict(run, done=time.time(), minutes=round((time.time() - run['started']) / 60, 1))))
    shutil.rmtree(d, ignore_errors=True)
    print(json.dumps({'tag': tag, 'status': 'complete', 'weights': str(final)}), flush=True)


if __name__ == '__main__':
    main()
