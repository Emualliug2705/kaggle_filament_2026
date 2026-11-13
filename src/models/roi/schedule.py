"""Learning-rate schedule of the ROI detector and refiner: linear warm-up, then cosine to 1 % of the peak."""
import math


def factor(step,s):
    if s['warmup'] and step<s['warmup']:return (step+1)/s['warmup']
    f=max(0,step-s['warmup'])/max(1,s['steps']-s['warmup'])
    return .01+.99*.5*(1+math.cos(math.pi*min(f,1)))
