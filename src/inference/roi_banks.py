"""ROI candidate banks (src.models.roi.inference) with the trained detector + refiner pairs:
- training photos of user fold k: pair f<k>, which did not train on them -> work/banks/roi-f<k>/train
- test photos: each fold pair -> work/banks/roi-f<k>/test (the scorer averages the 5)

    python -m src.inference.roi_banks K        (K = 0..4; "full" runs the pair trained on all photos on the test photos)
"""
import json
import sys

from src import paths
from src.models.roi import inference as I

OUT = paths.WORK / 'banks'


def main():
    which = sys.argv[1]
    test_names = sorted(p.name for p in paths.TEST_IMAGES.glob('*.jpeg'))
    if which == 'full':
        jobs = [('full', 'test', paths.TEST_IMAGES, test_names)]
    else:
        k = int(which)
        held = sorted(json.load(open(paths.FOLDS_JSON))['folds'][k])
        jobs = [(f'f{k}', 'train', paths.TRAIN_IMAGES, held), (f'f{k}', 'test', paths.TEST_IMAGES, test_names)]
    for tag, split, image_dir, names in jobs:
        out = OUT / f'roi-{tag}' / split
        if (out / 'manifest.json').is_file():
            print('done already', out); continue
        r = I.run(tag, image_dir=image_dir, out_dir=out, device='cuda', names=names)
        print(json.dumps({'bank': str(out), 'photos': len(r['records']),
                          'candidates': sum(x['candidates'] for x in r['records']), 'seconds': round(r['seconds'])}), flush=True)


if __name__ == '__main__':
    main()
