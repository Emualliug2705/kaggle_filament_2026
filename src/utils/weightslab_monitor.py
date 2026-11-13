"""WeightsLab supervision of one training run: live curves, per-photo signals and a web UI per (model, fold).

Used by the Mask2Former trainer (src/models/mask2former/train.py), the first-level models; the evidence models (U-Nets,
Mask R-CNN, ROI) train without it. A trainer calls `start(model, fold, out_dir, config, train_names, val_names)` once,
then reports what it computes:

    monitor = weightslab_monitor.start("unet_cnx", 0, out_dir, vars(args), train_names, val_names)
    with monitor.train_step():                       # one optimiser step (the UI's pause button waits here)
        ...
        monitor.log(step, **{"train/loss": loss, "train/lr": lr})
        monitor.samples(step, photo_names, {"train/photo_loss": per_crop_losses})
    with monitor.evaluation():                       # held-out pass
        monitor.samples(step, val_names, {"val/photo_loss": per_photo_losses})
    weightslab_monitor.finish(exit_code)             # flush, then exit

Training is not changed. WeightsLab never wraps the model, the optimiser, the losses or the data loaders: it is given
copies of the numbers the trainer already computes, and its own start-up runs with the random generators saved and
restored. So the weights do not depend on whether WeightsLab is on.

Names: curves are train/* and val/*; per-photo signals are train/photo_* and val/photo_* (each is also plotted as its
mean over the photos of the step).

Where things go (output_directory/models = FILAMENT_MODELS):
    models/{model}-f{fold}/                   the run's checkpoints, as before, and WeightsLab's experiment folder
        checkpoints/loggers/loggers.duckdb    curves and per-photo signal history
        checkpoints/data/                     per-photo table (origin fold{k}_train / fold{k}_val, tags, signals)
        weightslab.json                       the run's UI address and ports
    Experiment name: {model}-f{fold}.

The data WeightsLab shows are the 707 training photos: the original photo, its ground truth as a 256 px mask with two
classes (filament = drawn by at least one annotator, background), and, for held-out photos, the model's prediction in the
same two classes once an evaluation has run (see src/models/mask2former/train.py). Photos are registered as two splits,
fold{k}_train and fold{k}_val, and carry the categorical tag fold{k} = train | val.

Masks are handed to WeightsLab as uint16. WeightsLab 2.0.3 stores any other integer mask as uint16 marked "normalized"
without rescaling it, then rescales it on reading (value / 65535 x max), which turns a 0/1 mask into zeros.

Ports: run number r = position of (model, fold) in RUNS; UI http://localhost:{8100 + r}, gRPC 50100 + r. A busy port
falls back to a free one; the address used is printed and written to weightslab.json.

Environment:
    FILAMENT_WEIGHTSLAB=0          turn WeightsLab off (also off when the package is not installed)
    FILAMENT_WL_UI_PORT=8100       first UI port;  FILAMENT_WL_GRPC_PORT=50100  first gRPC port
    FILAMENT_WL_HOST=0.0.0.0       interface the UI binds
    FILAMENT_WL_LINGER=0           seconds the UI stays up after training (e.g. 3600 to inspect a finished run)
"""
from __future__ import annotations

import contextlib
import importlib.util
import json
import os
import random
import socket
import sys
import time
import traceback
from pathlib import Path

import numpy as np

from src import paths

MODEL_ORDER = ("mask2former", "mask2former_limb", "maskrcnn", "unet_cnx", "unet_cnxL", "unet_r34",
               "roi_detector", "roi_refiner")
SLOTS_PER_MODEL = 6                                   # folds 0-4 and the full-data run
MASK_SIZE = 256
CLASS_NAMES = ["background", "filament"]


def experiment_name(model: str, fold: int) -> str:
    return f"{model}-full" if fold < 0 else f"{model}-f{fold}"


def run_number(model: str, fold: int) -> int:
    base = MODEL_ORDER.index(model) if model in MODEL_ORDER else len(MODEL_ORDER)
    return base * SLOTS_PER_MODEL + (SLOTS_PER_MODEL - 1 if fold < 0 else fold)


def preferred_ports(model: str, fold: int) -> tuple[int, int]:
    r = run_number(model, fold)
    return (int(os.environ.get("FILAMENT_WL_UI_PORT", 8100)) + r, int(os.environ.get("FILAMENT_WL_GRPC_PORT", 50100)) + r)


def enabled() -> bool:
    if os.environ.get("FILAMENT_WEIGHTSLAB", "1").strip().lower() in ("0", "false", "no", "off"):
        return False
    return importlib.util.find_spec("weightslab") is not None


# ------------------------------------------------------------------------------------------------ random generators
def rng_state() -> dict:
    import torch
    state = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def set_rng_state(state: dict) -> None:
    import torch
    random.setstate(state["python"]); np.random.set_state(state["numpy"]); torch.set_rng_state(state["torch"])
    if "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


@contextlib.contextmanager
def preserve_rng():
    """Every random generator the trainers use is left exactly as it was found."""
    state = rng_state()
    try:
        yield
    finally:
        set_rng_state(state)


@contextlib.contextmanager
def preserve_modes(*modules):
    """train()/eval() of every submodule is restored afterwards (some trainers keep BatchNorm frozen in train mode)."""
    saved = [(m, m.training) for module in modules for m in module.modules()]
    try:
        yield
    finally:
        for m, training in saved:
            m.training = training


# ------------------------------------------------------------------------------------------------ the 707 photos
def _decode_record(annotations: list, height: int, width: int) -> np.ndarray:
    """Union of one annotator record's filaments at full resolution."""
    from pycocotools import mask as mu
    out = np.zeros((height, width), np.uint8)
    for annotation in annotations:
        seg = annotation["segmentation"]
        if isinstance(seg, str):
            seg = json.loads(seg)
        if isinstance(seg, list):
            polygons = [p for p in seg if len(p) >= 6]
            if not polygons:
                continue
            rle = mu.merge(mu.frPyObjects(polygons, height, width))
        elif isinstance(seg.get("counts"), list):
            rle = mu.frPyObjects(seg, height, width)
        else:
            rle = seg
        out |= mu.decode(rle).astype(np.uint8)
    return out


def _build_photo_bank(target: Path) -> None:
    import cv2
    from collections import defaultdict
    from concurrent.futures import ThreadPoolExecutor
    coco = json.loads(paths.ANNOTATIONS.read_text(encoding="utf-8"))
    records = defaultdict(list)                              # photo -> [record]
    for record in coco["images"]:
        records[record["file_name"]].append(record)
    by_record = defaultdict(list)
    for annotation in coco["annotations"]:
        by_record[annotation["image_id"]].append(annotation)
    user_fold = {n: k for k, names in enumerate(json.loads(paths.FOLDS_JSON.read_text())["folds"]) for n in names}
    names = sorted(p.name for p in paths.TRAIN_IMAGES.glob("*.jpeg"))

    def one(name):
        gray = cv2.imread(str(paths.TRAIN_IMAGES / name), cv2.IMREAD_GRAYSCALE)
        h, w = gray.shape
        count = np.zeros((h, w), np.uint8)
        filaments = []
        for record in records[name]:
            count += _decode_record(by_record[record["id"]], h, w)
            filaments.append(len(by_record[record["id"]]))
        thumb = cv2.resize(gray, (MASK_SIZE, MASK_SIZE), interpolation=cv2.INTER_AREA)
        # Block maximum keeps one-pixel-wide filaments visible after an 8x reduction.
        f = h // MASK_SIZE
        agreement = count[:f * MASK_SIZE, :f * MASK_SIZE].reshape(MASK_SIZE, f, MASK_SIZE, f).max((1, 3))
        meta = {"file_name": name, "date": f"{name[0:4]}-{name[4:6]}-{name[6:8]}", "site": Path(name).stem[-2],
                "annotators": len(records[name]), "filaments_per_annotator": round(float(np.mean(filaments)), 2) if filaments else 0.0,
                "user_fold": int(user_fold.get(name, -1))}
        return thumb, np.minimum(agreement, 3).astype(np.uint8), meta

    with ThreadPoolExecutor(16) as pool:
        rows = list(pool.map(one, names))
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.stem + f".{os.getpid()}.partial.npz")
    np.savez_compressed(partial, names=np.array(names), thumbs=np.stack([r[0] for r in rows]),
                        masks=np.stack([r[1] for r in rows]), meta=np.array(json.dumps([r[2] for r in rows])))
    partial.replace(target)


class PhotoBank:
    """Thumbnails, annotator-agreement masks and metadata of the training photos, cached in work/weightslab/."""

    def __init__(self):
        target = paths.WORK / "weightslab" / f"photos_{MASK_SIZE}.npz"
        if not target.is_file():
            lock = target.with_suffix(".lock")
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.close(fd)
                try:
                    _build_photo_bank(target)
                finally:
                    lock.unlink(missing_ok=True)
            except FileExistsError:                       # another trainer is building it
                deadline = time.time() + 900
                while not target.is_file() and lock.exists() and time.time() < deadline:
                    time.sleep(2)
                if not target.is_file():
                    _build_photo_bank(target)
        z = np.load(target)
        self.names = [str(n) for n in z["names"]]
        self.thumbs, self.masks = z["thumbs"], z["masks"]
        self.meta = json.loads(str(z["meta"]))
        self.row = {n: i for i, n in enumerate(self.names)}


def photo_view(bank: PhotoBank, names: list[str]):
    import torch

    class PhotoView(torch.utils.data.Dataset):
        """What WeightsLab shows: the original photo (`images`), the ground-truth mask, the photo's metadata."""
        task_type = "segmentation"
        num_classes = len(CLASS_NAMES)
        class_names = CLASS_NAMES

        def __init__(self):
            self.names = list(names)
            self.rows = [bank.row[n] for n in self.names]
            self.images = [str(paths.TRAIN_IMAGES / n) for n in self.names]

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, i):
            r = self.rows[i]
            # The thumbnail is also what WeightsLab hashes into the sample id, so a photo has the same id in every run.
            # Ground truth: filament where any annotator drew one; uint16 (see the module docstring).
            return bank.thumbs[r], self.names[i], (bank.masks[r] > 0).astype(np.uint16), dict(bank.meta[r])

    return PhotoView()


# ------------------------------------------------------------------------------------------------ sessions
def quiet_environment() -> None:
    """Settings applied before WeightsLab is imported (an explicit environment variable wins).

    Errors only on the terminal (the full log is in the experiment's weightslab_logs/); no 3-D projection hook; the
    UI's hyperparameter inputs read-only (the trainers do not read them back, so an edit would change nothing); no
    embedded Jupyter kernel unless FILAMENT_WL_KERNEL=1 (it takes over Ctrl+C and swaps sys.stdout; the UI notebook
    then uses WeightsLab's built-in executor)."""
    os.environ.setdefault("WEIGHTSLAB_LOG_LEVEL", "ERROR")
    os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
    os.environ.setdefault("WEIGHTSLAB_PROJECTION", "0")
    os.environ.setdefault("ENABLE_HYPERPARAMETERS_OPTIMIZATION", "0")
    if os.environ.get("FILAMENT_WL_KERNEL", "0") != "1":
        os.environ.setdefault("WEIGHTSLAB_DISABLE_EMBEDDED_KERNEL", "1")


def say(line: str) -> None:
    """A line on the process's own stdout, even if a library has replaced sys.stdout."""
    stream = sys.__stdout__ or sys.stdout
    stream.write(line + "\n"); stream.flush()


def _free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("0.0.0.0", port))
            return True
        except OSError:
            return False


def _pick(port: int) -> int:
    for candidate in (port, *range(port + 1000, port + 1100)):
        if _free(candidate):
            return candidate
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("0.0.0.0", 0))
        return s.getsockname()[1]


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


class _Off:
    """WeightsLab disabled: every call is a no-op."""
    active = False
    name = ""

    @contextlib.contextmanager
    def train_step(self):
        yield

    @contextlib.contextmanager
    def evaluation(self):
        yield

    def log(self, step, **scalars):
        pass

    def samples(self, step, names, signals, preds=None):
        pass

    def close(self):
        pass


class Session:
    active = True

    def __init__(self, model: str, fold: int, out_dir: Path, config: dict, train_names, val_names):
        self.model, self.fold, self.out_dir = model, fold, Path(out_dir)
        self.name = experiment_name(model, fold)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        quiet_environment()
        import weightslab as wl
        from weightslab.backend import ledgers
        from weightslab.ui import server as ui_server
        self.wl, self.ledgers = wl, ledgers
        split = "full" if fold < 0 else f"fold{fold}"
        hp = {"experiment_name": self.name, "root_log_dir": str(self.out_dir), "is_training": True,
              "skip_checkpoint_load": True, "checkpoint_manager": {"load_config": False, "load_weights": False},
              "model": model, "fold": fold, "training": _jsonable(config),
              "train_photos": len(train_names), "val_photos": len(val_names)}
        wl.watch_or_edit(dict(hp), flag="hyperparameters", name=self.name, defaults=dict(hp), poll_interval=5.0)

        bank = PhotoBank()
        self.uid = {}
        for names, origin, training in ((train_names, f"{split}_train", True), (val_names, f"{split}_val", False)):
            if not names:
                continue
            view = photo_view(bank, sorted(names))
            loader = wl.watch_or_edit(view, flag="data", loader_name=origin, batch_size=1, shuffle=False,
                                      is_training=training, compute_hash=True, preload_labels=True,
                                      preload_metadata=True, num_workers=0)
            ids = [str(u) for u in loader.tracked_dataset.unique_ids]
            if len(ids) != len(view.names) or len(set(ids)) != len(ids):
                raise RuntimeError(f"WeightsLab gave {len(set(ids))} distinct ids to {len(view.names)} photos")
            self.uid.update(zip(view.names, ids))
            if fold >= 0:
                wl.set_categorical_tag(ids, f"fold{fold}", "train" if training else "val")
        if fold >= 0:
            wl.register_categorical_tag(f"fold{fold}", ["train", "val"])

        ui_pref, grpc_pref = preferred_ports(model, fold)
        self.grpc_port = _pick(grpc_pref)
        wl.serve(serving_grpc=True, serving_cli=False, grpc_port=self.grpc_port)
        self.ui_port = _pick(ui_pref if ui_pref != self.grpc_port else ui_pref + 1)
        ui_server.serve_ui(ui_host=os.environ.get("FILAMENT_WL_HOST", "0.0.0.0"), ui_port=self.ui_port,
                           backend_host="localhost", backend_port=self.grpc_port, open_browser=False, block=False,
                           experiment_dir=str(self.out_dir))
        wl.start_training(timeout=0)
        self.url = f"http://localhost:{self.ui_port}"
        self._write_card(state="training")
        say(f"WEIGHTSLAB {self.name}: UI {self.url} (gRPC {self.grpc_port}), experiment folder {self.out_dir}")

    def _write_card(self, **extra):
        card = {"experiment": self.name, "model": self.model, "fold": self.fold, "ui": self.url, "ui_port": self.ui_port,
                "grpc_port": self.grpc_port, "pid": os.getpid(), "updated": time.strftime("%Y-%m-%dT%H:%M:%S"), **extra}
        (self.out_dir / "weightslab.json").write_text(json.dumps(card, indent=1))

    def _logger(self):
        return self.ledgers.get_logger()

    @contextlib.contextmanager
    def _guard(self, guard):
        # WeightsLab's guard swallows RuntimeError (CUDA out-of-memory included). The trainer must still see it.
        caught = []
        with guard:
            try:
                yield
            except BaseException as error:              # noqa: BLE001 - re-raised below, outside the guard
                caught.append(error)
        if caught:
            raise caught[0]

    def train_step(self):
        return self._guard(self.wl.guard_training_context)

    @contextlib.contextmanager
    def evaluation(self):
        with self._guard(self.wl.guard_testing_context):
            yield
        self._close_step()

    def _close_step(self) -> None:
        """Emit the mean of the per-photo signals of the current step now; WeightsLab otherwise waits for the next step."""
        try:
            lg = self._logger()
            with lg._lock:
                lg._flush_current_step_buffer(add_to_queue=True)
        except Exception:
            traceback.print_exc()

    def log(self, step: int, **scalars) -> None:
        """Scalar curves; names may hold '/' (train/loss)."""
        try:
            lg = self._logger()
            for name, value in scalars.items():
                if value is None:
                    continue
                lg.add_scalars(name, {name: float(value)}, global_step=int(step), signal_per_sample=None,
                               aggregate_by_step=False)
        except Exception:                                # monitoring never stops a training run
            traceback.print_exc()

    def samples(self, step: int, names, signals: dict, preds=None) -> None:
        """Per-photo signals: names[i] is the photo the i-th value of every signal belongs to. Each signal is also
        plotted as its mean over these photos. Call it inside train_step() for training values and inside
        evaluation() for held-out ones. preds: optional (N, H, W) masks (0 background, 1 filament), shown over the
        photos in the UI."""
        try:
            keep = [i for i, n in enumerate(names) if n in self.uid]
            if not keep:
                return
            ids = [self.uid[names[i]] for i in keep]
            values = {k: np.asarray(v, dtype=np.float32).reshape(len(names))[keep] for k, v in signals.items()}
            masks = None if preds is None else np.asarray(preds, dtype=np.uint16)[keep]
            self.wl.save_signals(signals=values, batch_ids=ids, step=int(step), log=True, preds=masks)
        except Exception:
            traceback.print_exc()

    def flush(self) -> None:
        try:
            self.wl.drain_signals()
            self._close_step()
            self._logger().flush_to_disk()
            df = self.ledgers.get_dataframe()
            if df is not None:
                df.flush()
            self.wl.write_dataframe()
        except Exception:
            traceback.print_exc()

    def close(self) -> None:
        self.flush()
        linger = int(os.environ.get("FILAMENT_WL_LINGER", "0") or 0)
        self._write_card(state="finished", ui_open_until=time.strftime("%Y-%m-%dT%H:%M:%S",
                                                                         time.localtime(time.time() + linger)))
        if linger > 0:
            say(f"WEIGHTSLAB {self.name}: training finished; UI stays at {self.url} for {linger} s")
            time.sleep(linger)


_SESSION = _Off()


def start(model: str, fold: int, out_dir, config: dict, train_names, val_names):
    """The run's WeightsLab session, or a no-op stand-in when WeightsLab is off or fails to start."""
    global _SESSION
    if not enabled():
        return _SESSION
    with preserve_rng():
        try:
            _SESSION = Session(model, fold, Path(out_dir), config, list(train_names), list(val_names))
        except Exception:
            traceback.print_exc()
            print("WEIGHTSLAB: could not start; training continues without it", flush=True)
            _SESSION = _Off()
    return _SESSION


def current():
    return _SESSION


def finish(code: int = 0):
    """Flush WeightsLab and end the process with `code`.

    WeightsLab's server threads make a normal interpreter exit abort ('terminate called without an active exception',
    status 134), which a launcher would read as a failed training. Once everything is on disk the process ends here.
    """
    code = int(code or 0)
    if not _SESSION.active:
        sys.exit(code)
    try:
        _SESSION.close()
    except Exception:
        traceback.print_exc()
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(code)


def run_main(main) -> None:
    """`main()` returns an exit code (or None); exceptions are printed and give exit code 1."""
    try:
        code = main()
    except SystemExit as stop:
        code = stop.code if isinstance(stop.code, int) else (0 if stop.code is None else 1)
        if stop.code not in (None, 0) and not isinstance(stop.code, int):
            print(stop.code, file=sys.stderr)
    except BaseException:                                  # noqa: BLE001
        traceback.print_exc()
        code = 1
    finish(code if isinstance(code, int) else 0)
