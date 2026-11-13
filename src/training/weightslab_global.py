"""One WeightsLab UI over every training run in output_directory/models: all curves, all per-photo signals, all folds.

    python -m src.training.weightslab_global              serve http://localhost:8099 (gRPC 50099) until Ctrl+C
    python -m src.training.weightslab_global --list       the runs found, their state and their own UI address

What it shows
    Curves: every run's curves (train/*, val/*), one curve per run, named {model}-f{fold}.
    Data: the 707 training photos once. Each carries the categorical tags fold0 ... fold4 = train | val (the fold in
        which the photo was held out is `val`), and its metadata (site, date, annotators, user_fold).
    Per-photo columns, per model (folds pooled):
        {model}/oof_{signal}     the held-out value, from the one run in which the photo was not trained on
                                 (val/photo_{signal} at the run's last evaluation)
        {model}/train_{signal}   the training value, averaged over the four runs that trained on the photo
                                 (train/photo_{signal} averaged over each run's last 10 % of steps)
    Per-photo trajectories of every run stay available, under the run's experiment.

The runs are only read. The global view is rebuilt from them at each start in models/_weightslab_global/ (a run that is
still training is read from a copy of its database).
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


def per_photo_columns(sources: dict) -> dict:
    """{column: {sample id: value}} pooled per model over its folds (see the module docstring)."""
    import duckdb
    train_parts = defaultdict(lambda: defaultdict(list))         # column -> sample -> [value per run]
    oof = defaultdict(dict)
    for (model, fold), db in sources.items():
        con = duckdb.connect(str(db), read_only=True)
        for (metric,) in con.execute("SELECT DISTINCT metric_name FROM per_sample").fetchall():
            if metric.startswith("val/"):
                rows = con.execute("""
                    SELECT sample_id, arg_max(value, step) FROM per_sample
                    WHERE metric_name = ? AND step = (SELECT max(step) FROM per_sample WHERE metric_name = ?)
                    GROUP BY sample_id""", [metric, metric]).fetchall()
                column = f"{model}/oof_{metric[4:].removeprefix('photo_')}"
                for sid, value in rows:
                    oof[column][str(sid)] = float(value)
            elif metric.startswith("train/"):
                last = con.execute("SELECT max(step) FROM per_sample WHERE metric_name = ?", [metric]).fetchone()[0]
                rows = con.execute("""
                    SELECT sample_id, avg(value) FROM per_sample
                    WHERE metric_name = ? AND step >= ? GROUP BY sample_id""", [metric, int(last * 0.9)]).fetchall()
                column = f"{model}/train_{metric[6:].removeprefix('photo_')}"
                for sid, value in rows:
                    train_parts[column][str(sid)].append(float(value))
        con.close()
    out = dict(oof)
    for column, per_sample in train_parts.items():
        out[column] = {sid: float(np.mean(v)) for sid, v in per_sample.items()}
    return out


def serve(models: Path, ui_port: int, grpc_port: int) -> None:
    root = models / GLOBAL
    shutil.rmtree(root, ignore_errors=True)                   # rebuilt from the runs at every start
    root.mkdir(parents=True)
    runs = find_runs(models)
    if not runs:
        raise SystemExit(f"no WeightsLab run under {models} (expected {{model}}-f{{fold}}/checkpoints/loggers)")
    wm.quiet_environment()
    import yaml
    import weightslab as wl
    from weightslab.backend import ledgers
    from weightslab.ui import server as ui_server

    hp = {"experiment_name": "all-runs", "root_log_dir": str(root), "is_training": False, "skip_checkpoint_load": True,
          "checkpoint_manager": {"load_config": False, "load_weights": False},
          "runs": [r.name for r in runs]}
    wl.watch_or_edit(dict(hp), flag="hyperparameters", name="all-runs", defaults=dict(hp), poll_interval=60.0)

    scratch = Path(tempfile.mkdtemp(prefix="wl_global_"))
    logger = ledgers.get_logger()
    cm = ledgers.get_checkpoint_manager()
    manifest = cm._load_manifest()
    sources = {}
    for run in runs:
        db = readable_copy(run / "checkpoints" / "loggers" / "loggers.duckdb", scratch)
        logger.merge_from_disk(str(db))
        sources[run_identity(run)] = db
        run_manifest = yaml.safe_load((run / "checkpoints" / "manifest.yaml").read_text()) or {}
        for h, info in (run_manifest.get("experiments") or {}).items():
            manifest.setdefault("experiments", {})[h] = dict(info, experiment_name=info.get("experiment_name") or run.name,
                                                             notes=f"{run}")
        print(f"  {run.name}: curves merged", flush=True)
    cm._write_manifest(manifest)

    bank = wm.PhotoBank()
    view = wm.photo_view(bank, bank.names)
    loader = wl.watch_or_edit(view, flag="data", loader_name="training_photos", batch_size=1, shuffle=False,
                              is_training=False, compute_hash=True, preload_labels=True, preload_metadata=True,
                              num_workers=0)
    ids = [str(u) for u in loader.tracked_dataset.unique_ids]
    uid = dict(zip(view.names, ids))
    folds = json.loads(paths.FOLDS_JSON.read_text())["folds"]
    for k, held_out in enumerate(folds):
        held_out = set(held_out)
        wl.register_categorical_tag(f"fold{k}", ["train", "val"])
        wl.set_categorical_tag([uid[n] for n in view.names if n not in held_out], f"fold{k}", "train")
        wl.set_categorical_tag([uid[n] for n in view.names if n in held_out], f"fold{k}", "val")

    columns = per_photo_columns(sources)
    known = set(ids)
    for column, values in sorted(columns.items()):
        sids = [s for s in values if s in known]
        if sids:
            wl.save_signals(signals={column: np.array([values[s] for s in sids], np.float32)}, batch_ids=sids, step=0,
                            log=False)
    wl.drain_signals()
    print(f"  per-photo columns: {len(columns)} ({', '.join(sorted(columns)[:6])}{' ...' if len(columns) > 6 else ''})",
          flush=True)

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
