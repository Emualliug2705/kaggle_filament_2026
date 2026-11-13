"""Two-family candidate pools: the 16 raw views of mask2former and mask2former_limb (same fold) fused with the view
fusion rule (link IoU 0.5, pixel vote 0.5; score = mean member score x (0.5 + 0.5 members / 16))
-> work/pools/fold{k}_{val,test}.json, the input of the scorer.

    python -m src.inference.pools [link vote]
"""
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

from src import paths
from src.inference.m2f_caches import fuse_n

V = paths.WORK / "views"; OUT = paths.WORK / "pools"
LINK, VOTE = (float(sys.argv[1]), float(sys.argv[2])) if len(sys.argv) > 2 else (0.5, 0.5)


def photo(job):
    name, views = job
    entries = [(s, dict(size=[2048, 2048], counts=c.encode())) for vv in views for _, s, c in vv]
    fused = fuse_n(entries, LINK, VOTE, 16)
    return name, [dict(score=e["score"], rle=dict(size=e["rle"]["size"], counts=e["rle"]["counts"] if isinstance(e["rle"]["counts"], str) else e["rle"]["counts"].decode())) for e in fused]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for k in range(5):
        for split in ("val", "test"):
            a = json.loads((V / "mask2former" / f"fold{k}_{split}.json").read_text()); b = json.loads((V / "mask2former_limb" / f"fold{k}_{split}.json").read_text())
            assert set(a) == set(b)
            with ProcessPoolExecutor(int(os.environ.get("FILAMENT_CPUS", "20"))) as ex:
                res = dict(ex.map(photo, [(n, (a[n], b[n])) for n in sorted(a)], chunksize=2))
            (OUT / f"fold{k}_{split}.json").write_text(json.dumps(res))
            print(f"fold {k} {split}: {len(res)} photos, {sum(map(len, res.values()))} candidates "
                  f"(mask2former alone, 8 views: {sum(len(v) for v in json.loads((paths.WORK / 'mask2former' / f'fold{k}_{split}_d4.json').read_text()).values())})", flush=True)


if __name__ == "__main__":
    main()
