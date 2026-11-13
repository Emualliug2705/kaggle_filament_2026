"""Fusion of the candidates of several views or models: candidates linked by mask IoU, a pixel vote for each group's
mask, and a group score from its members."""
import numpy as np
from pycocotools import mask as mu


def vote_rle(masks, needed):
    """Exact k-of-n voting using sparse RLE intersections and unions."""
    levels = [None] * needed
    for i, mask in enumerate(masks):
        for k in range(min(i + 1, needed), 1, -1):
            both = mu.merge([levels[k - 2], mask], intersect=True)
            levels[k - 1] = both if levels[k - 1] is None else mu.merge([levels[k - 1], both])
        levels[0] = mask if levels[0] is None else mu.merge([levels[0], mask])
    result = levels[-1]
    counts = result['counts']
    return dict(size=result['size'], counts=counts.decode('ascii') if isinstance(counts, bytes) else counts)


def fuse(entries, link, vote):
    order=sorted(range(len(entries)),key=lambda i:-entries[i][0])
    if not order: return []
    ious=mu.iou([e[1] for e in entries],[e[1] for e in entries],[0]*len(entries))
    used=set(); output=[]
    for seed in order:
        if seed in used: continue
        members=[seed];used.add(seed)
        for i in order:
            if i not in used and ious[seed,i]>=link:
                members.append(i);used.add(i)
        binary=vote_rle([entries[i][1] for i in members], max(1,round(vote*len(members))))
        if mu.area(binary)<40: continue
        score=float(np.mean([entries[i][0] for i in members]))
        output.append(dict(score=score*(.5+.5*len(members)/8),rle=binary))
    return output
