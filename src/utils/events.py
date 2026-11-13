"""JSON event lines for training logs."""
from __future__ import annotations

import json
from pathlib import Path


def emit(tag: str, payload: dict, log_path: Path | None = None) -> str:
    line = f"{tag} {json.dumps(payload, sort_keys=True, default=str)}"
    print(line, flush=True)
    if log_path is not None:
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
    return line
