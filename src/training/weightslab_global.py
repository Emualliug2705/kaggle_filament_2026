"""One WeightsLab UI over every training run in output_directory/models: all curves, every run's last state, all folds.

    python -m src.training.weightslab_global              serve http://localhost:8099 (gRPC 50099) until Ctrl+C
    python -m src.training.weightslab_global --list       the runs found, their state and their own UI address

What it shows
    Current experiment: the most recently updated run found under models/ (no new experiment is created). Its last
        state is loaded: its per-photo signals (train/photo_*, val/photo_* last values), its tags, its seen counts,
        and its prediction masks. Pick any other run in the UI (plots board, runs popup -> restore) to load its own
        last state the same way.
    Curves: every run's curves (train/*, val/*), one curve per run hash, named {model}-f{fold}.
    Data: the 707 training photos once, each with its ground truth (filament / background), the categorical tags
        fold0 ... fold4 = train | val (the fold in which the photo was held out is `val`), and its metadata (site,
        date, annotators, user_fold).
    Masks: ground truth, and the loaded run's predictions: the emitted masks of its latest evaluation (its held-out
        photos), from its e###_pred2048.npz (or from its WeightsLab arrays.h5 when it was copied along).
    Per-photo columns pooled over the folds of each model, shown whatever run is loaded:
        {model}/oof_{signal}     the held-out value, from the one run in which the photo was not trained on
                                 (val/photo_{signal} at the run's last evaluation)
        {model}/train_{signal}   the training value, averaged over the four runs that trained on the photo
                                 (train/photo_{signal} averaged over each run's last 10 % of steps)

The runs are only read: WeightsLab opens models/ read-only (checkpoint_manager.read_only), so no manifest, database or
data file of a run is written to; a run that is still training is read from a copy of its database. The viewer's own
files (its data view, logs, notebook) go to models/_weightslab_global/, rebuilt at each start.
A run's last per-photo state is its WeightsLab data store (checkpoints/data/data.h5) or, when only that was copied
back from the training machine, its dataframe export ({hash}_dataframe.parquet, written when training ends).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from src import paths
from src.utils import weightslab_monitor as wm

GLOBAL = "_weightslab_global"


def find_runs(models: Path) -> list[Path]:
    return sorted(d for d in models.iterdir()
                  if d.is_dir() and d.name != GLOBAL and (d / "checkpoints" / "loggers" / "loggers.duckdb").is_file())


def alive(pid) -> bool:
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, TypeError, ValueError):
        return False


def list_runs(models: Path) -> None:
    rows = []
    for d in sorted(p for p in models.iterdir() if (p / "weightslab.json").is_file()):
        card = json.loads((d / "weightslab.json").read_text())
        running = alive(card.get("pid"))
        state = card.get("state", "?")
        rows.append((card.get("experiment", d.name), "training" if running and state == "training" else
                     ("UI open" if running else state), card.get("ui", "") if running else "-", card.get("updated", "")))
    if not rows:
        print(f"no WeightsLab runs under {models}")
        return
    width = max(len(r[0]) for r in rows)
    for name, state, ui, updated in rows:
        print(f"{name:<{width}}  {state:<9}  {ui:<24}  {updated}")


def readable_copy(db: Path, scratch: Path) -> Path:
    """The run's database, or a copy of it when a training process still holds it open."""
    import duckdb
    try:
        duckdb.connect(str(db), read_only=True).close()
        return db
    except Exception:
        target = scratch / f"{db.parent.parent.parent.name}.duckdb"
        shutil.copy2(db, target)
        wal = db.with_name(db.name + ".wal")
        if wal.is_file():
            shutil.copy2(wal, target.with_name(target.name + ".wal"))
        return target


def run_identity(run: Path) -> tuple[str, int]:
    card = run / "weightslab.json"
    if card.is_file():
        c = json.loads(card.read_text())
        return c["model"], int(c["fold"])
    model, _, fold = run.name.rpartition("-f")
    return model, int(fold) if fold.isdigit() else -1


def per_photo_columns(sources: dict, to_global: dict) -> dict:
    """{column: {sample id: value}} pooled per model over its folds (see the module docstring).

    to_global[(model, fold)] maps the run's sample ids to the global view's (through the photo names)."""
    import duckdb
    train_parts = defaultdict(lambda: defaultdict(list))         # column -> sample -> [value per run]
    oof = defaultdict(dict)
    for (model, fold), db in sources.items():
        remap = to_global.get((model, fold), {})
        con = duckdb.connect(str(db), read_only=True)
        for (metric,) in con.execute("SELECT DISTINCT metric_name FROM per_sample").fetchall():
            if metric.startswith("val/"):
                rows = con.execute("""
                    SELECT sample_id, arg_max(value, step) FROM per_sample
                    WHERE metric_name = ? AND step = (SELECT max(step) FROM per_sample WHERE metric_name = ?)
                    GROUP BY sample_id""", [metric, metric]).fetchall()
                column = f"{model}/oof_{metric[4:].removeprefix('photo_')}"
                for sid, value in rows:
                    if str(sid) in remap:
                        oof[column][remap[str(sid)]] = float(value)
            elif metric.startswith("train/"):
                last = con.execute("SELECT max(step) FROM per_sample WHERE metric_name = ?", [metric]).fetchone()[0]
                rows = con.execute("""
                    SELECT sample_id, avg(value) FROM per_sample
                    WHERE metric_name = ? AND step >= ? GROUP BY sample_id""", [metric, int(last * 0.9)]).fetchall()
                column = f"{model}/train_{metric[6:].removeprefix('photo_')}"
                for sid, value in rows:
                    if str(sid) in remap:
                        train_parts[column][remap[str(sid)]].append(float(value))
        con.close()
    out = dict(oof)
    for column, per_sample in train_parts.items():
        out[column] = {sid: float(np.mean(v)) for sid, v in per_sample.items()}
    return out


def latest_predictions(run: Path):
    """(photo names, (N, 2048, 2048) masks 0/1) of the run's latest e###_pred2048.npz (its held-out photos), or None."""
    files = sorted(run.glob("e[0-9][0-9][0-9]_pred2048.npz"))
    if not files:
        return None
    z = np.load(files[-1])
    return [str(n) for n in z["names"]], z["masks"]


def serve(models: Path, ui_port: int, grpc_port: int) -> None:
    root = models / GLOBAL                                     # the viewer's own files, never a run's
    shutil.rmtree(root, ignore_errors=True)                   # rebuilt from the runs at every start
    # A file another process still holds (e.g. the agent server of a viewer that was just stopped) survives the
    # removal: start anyway; whatever the viewer needs is written again.
    root.mkdir(parents=True, exist_ok=True)
    runs = find_runs(models)
    if not runs:
        raise SystemExit(f"no WeightsLab run under {models} (expected {{model}}-f{{fold}}/checkpoints/loggers)")
    wm.quiet_environment()
    import weightslab as wl
    from weightslab.backend import ledgers
    from weightslab.ui import server as ui_server

    # root_log_dir = models/: WeightsLab finds every run below it, takes the most recently updated one as the current
    # experiment and merges every run's curves. read_only: no new experiment, nothing written into the runs; the
    # viewer's own files (data view, logs, audit log, notebook) go to its workspace, models/_weightslab_global/.
    hp = {"experiment_name": "all-runs", "root_log_dir": str(models), "is_training": False,
          "checkpoint_manager": {"read_only": True, "workspace": str(root), "load_config": False,
                                 "load_weights": False},
          "runs": [r.name for r in runs]}
    wl.watch_or_edit(dict(hp), flag="hyperparameters", name="all-runs", defaults=dict(hp), poll_interval=60.0)
    cm = ledgers.get_checkpoint_manager()
    if not getattr(cm, "read_only", False):
        raise SystemExit("this WeightsLab has no read-only viewer mode (checkpoint_manager.read_only); "
                         "install the weightslab branch feat/multi-experiment-root")
    print(f"  {len(cm.discovered_root_dirs)} runs found; curves merged; current experiment: "
          f"{cm.root_log_dir.name} ({cm.current_exp_hash})", flush=True)

    bank = wm.PhotoBank()
    view = wm.photo_view(bank, bank.names)
    loader = wl.watch_or_edit(view, flag="data", loader_name="training_photos", batch_size=1, shuffle=False,
                              is_training=False, compute_hash=True, preload_labels=True, preload_metadata=True,
                              num_workers=0, root_log_dir=str(root))
    ids = [str(u) for u in loader.tracked_dataset.unique_ids]
    uid = dict(zip(view.names, ids))
    folds = json.loads(paths.FOLDS_JSON.read_text())["folds"]
    for k, held_out in enumerate(folds):
        held_out = set(held_out)
        wl.register_categorical_tag(f"fold{k}", ["train", "val"])
        wl.set_categorical_tag([uid[n] for n in view.names if n not in held_out], f"fold{k}", "train")
        wl.set_categorical_tag([uid[n] for n in view.names if n in held_out], f"fold{k}", "val")

    # Per-photo columns pooled over the folds of each model: the viewer's own, kept whatever run is loaded.
    scratch = Path(tempfile.mkdtemp(prefix="wl_global_"))
    sources = {run_identity(run): readable_copy(run / "checkpoints" / "loggers" / "loggers.duckdb", scratch)
               for run in runs}
    to_global = {}
    for run in runs:
        ids_file = run / "weightslab_ids.json"
        if ids_file.is_file():
            to_global[run_identity(run)] = {str(u): uid[n] for n, u in json.loads(ids_file.read_text()).items() if n in uid}
    columns = per_photo_columns(sources, to_global)
    known = set(ids)
    for column, values in sorted(columns.items()):
        sids = [s for s in values if s in known]
        if sids:
            wl.save_signals(signals={column: np.array([values[s] for s in sids], np.float32)}, batch_ids=sids, step=0,
                            log=False)
    wl.drain_signals()
    print(f"  per-photo columns: {len(columns)} ({', '.join(sorted(columns)[:6])}{' ...' if len(columns) > 6 else ''})",
          flush=True)

    @wl.on_experiment_load
    def show_run_predictions(exp_hash, run_dir):
        """After WeightsLab loaded a run's per-photo state: its prediction masks, which the trainer keeps in its own
        e###_pred2048.npz (WeightsLab's copy, arrays.h5, usually stays on the training machine)."""
        found = latest_predictions(Path(run_dir))
        if found is None:
            wm.say(f"  {Path(run_dir).name}: per-photo state loaded (no prediction file)")
            return
        names, masks = found
        keep = [i for i, n in enumerate(names) if n in uid]
        if keep:
            masks = masks[keep]
            wl.save_signals(signals={"val/pred_pixels": masks.reshape(len(keep), -1).sum(1).astype(np.float32)},
                            batch_ids=[uid[names[i]] for i in keep], preds=masks.astype(np.uint16), step=0, log=False)
            wl.drain_signals()
        wm.say(f"  {Path(run_dir).name}: per-photo state and {len(keep)} prediction masks loaded ({exp_hash})")

    if not wl.load_experiment():                              # the current (most recent) run's last state
        print("  WARNING: the current run's per-photo state could not be loaded", flush=True)

    grpc_port = wm._pick(grpc_port)
    wl.serve(serving_grpc=True, serving_cli=False, grpc_port=grpc_port)
    ui_port = wm._pick(ui_port)
    ui_server.serve_ui(ui_host=os.environ.get("FILAMENT_WL_HOST", "0.0.0.0"), ui_port=ui_port, backend_host="localhost",
                       backend_port=grpc_port, open_browser=False, block=False, experiment_dir=str(root))
    (root / "weightslab.json").write_text(json.dumps({"experiment": "all-runs", "ui": f"http://localhost:{ui_port}",
                                                      "ui_port": ui_port, "grpc_port": grpc_port, "pid": os.getpid(),
                                                      "runs": [r.name for r in runs]}, indent=1))
    wm.say(f"WEIGHTSLAB all runs ({len(runs)}): UI http://localhost:{ui_port} (gRPC {grpc_port}); Ctrl+C to stop")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    shutil.rmtree(scratch, ignore_errors=True)
    sys.stdout.flush()
    os._exit(0)                       # WeightsLab's server threads would make a normal exit abort


def main() -> None:
    a = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    a.add_argument("--models", type=Path, default=paths.MODELS, help="output_directory/models (default FILAMENT_MODELS)")
    a.add_argument("--list", action="store_true", help="list the runs and their UI addresses, then exit")
    a.add_argument("--port", type=int, default=8099)
    a.add_argument("--grpc-port", type=int, default=50099)
    args = a.parse_args()
    if args.list:
        return list_runs(args.models)
    serve(args.models.resolve(), args.port, args.grpc_port)


if __name__ == "__main__":
    main()
