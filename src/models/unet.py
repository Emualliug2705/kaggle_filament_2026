"""U-Net (segmentation_models_pytorch) with 1 input channel and 2 outputs (filament probability, spine), the input
normalisation and the 8-view (D4) full-resolution prediction."""
from __future__ import annotations

import contextlib

import numpy as np
import segmentation_models_pytorch as smp
import torch

def build_model(arch, encoder, weights='imagenet'):
    """The smp network of one dense model: 1 input channel, 2 output channels (mask, spine).

    ``weights='imagenet'`` loads the pretrained encoder (training); inference passes ``weights=None`` (no download)
    and then loads the EMA state dict.
    """
    cls = {'unet': smp.Unet, 'unetpp': smp.UnetPlusPlus, 'fpn': smp.FPN}[arch]
    kw = dict(encoder_name=encoder, encoder_weights=weights, in_channels=1, classes=2)
    if arch == 'unet' and encoder.startswith('tu-convnext'):
        kw['decoder_channels'] = (256, 128, 64, 32, 16)
    return cls(**kw)


def normalize(img_u8, mean, std):
    """float32 ``clip((img - mean) / std, -6, 4)``."""
    x = (img_u8.astype(np.float32) - mean) / std
    return np.clip(x, -6, 4)


def _amp(device):
    """bf16 autocast on CUDA; nothing (float32) on the CPU."""
    if torch.device(device).type == 'cuda':
        return torch.autocast('cuda', dtype=torch.bfloat16)
    return contextlib.nullcontext()


@torch.no_grad()
def predict_full(model, x):
    """x: (1, 1, H, W) normalised, on the model's device. Mean sigmoid over the 8 D4 views; returns (2, H, W) float32."""
    acc = 0
    for k in range(4):
        for fl in (False, True):
            v = torch.rot90(x, k, (2, 3))
            if fl:
                v = v.flip(3)
            with _amp(x.device):
                o = torch.sigmoid(model(v.contiguous(memory_format=torch.channels_last)).float())
            if fl:
                o = o.flip(3)
            acc = acc + torch.rot90(o, -k, (2, 3))
    return (acc / 8)[0]
