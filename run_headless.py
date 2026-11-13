"""Runs PIPELINE.ipynb without Jupyter: every code cell in order, in one namespace, from the bundle folder.
A cell's last expression is printed (tables). Writes outputs/logs/RUN_DONE or RUN_FAILED.
    cd <repository> && mkdir -p outputs/logs && setsid nohup python run_headless.py > outputs/logs/headless.log 2>&1 &
The environment variables of the configuration cell (FILAMENT_*, GPUS, CPUS, RETRAIN, DATA_OK, FORCE) apply."""
import ast
import json
import os
import sys
import time
import traceback
from pathlib import Path


def main():
    here = Path(__file__).resolve().parent
    os.chdir(here); os.environ.setdefault('MPLBACKEND', 'Agg'); os.environ.setdefault('FILAMENT_REPO', str(here))
    logs = Path(os.environ.get('FILAMENT_OUTPUTS', here / 'outputs')) / 'logs'; logs.mkdir(parents=True, exist_ok=True)
    for f in ('RUN_DONE', 'RUN_FAILED'):
        (logs / f).unlink(missing_ok=True)
    nb = json.loads((here / 'PIPELINE.ipynb').read_text())
    ns = {'__name__': '__main__'}
    t_all = time.time()
    for i, cell in enumerate(c for c in nb['cells'] if c['cell_type'] == 'code'):
        src = ''.join(cell['source'])
        print(f"\n===== cell {i} ({time.strftime('%H:%M:%S')}) =====", flush=True)
        t0 = time.time()
        try:
            tree = ast.parse(src)
            last = tree.body.pop() if tree.body and isinstance(tree.body[-1], ast.Expr) else None
            exec(compile(tree, f'cell{i}', 'exec'), ns)
            if last is not None:
                v = eval(compile(ast.Expression(last.value), f'cell{i}', 'eval'), ns)
                if v is not None:
                    print(v.to_string() if hasattr(v, 'to_string') else v)
        except BaseException:
            traceback.print_exc()
            (logs / 'RUN_FAILED').write_text(f'cell {i}\n{traceback.format_exc()}')
            sys.exit(1)
        print(f'----- cell {i} done in {time.time() - t0:.0f}s', flush=True)
    (logs / 'RUN_DONE').write_text(f'{(time.time() - t_all) / 60:.1f} min')


if __name__ == '__main__':
    main()
