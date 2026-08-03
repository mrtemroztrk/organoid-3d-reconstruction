"""Extended depth of field: one image where every organoid is in focus.

Segmenting slices one at a time has a structural recall problem. An organoid is
crisply outlined only within ~3 slices of its focal plane; on all the others it
is a diffuse disc. So the segmenter gets exactly one good look at each object,
and if it misses that look -- because the rim is faint, or the organoid overlaps
a neighbour's haze on that particular slice -- the object is lost entirely.

The focus stack already contains the answer. For every pixel there is a slice
where it is sharpest; taking the intensity from that slice yields an image in
which *every* organoid shows its crisp rim simultaneously, and the slice index
itself is a depth map. Segment that once and the detector gets its best look at
all objects at the same time.

This is the standard focus-stacking / shape-from-focus construction, and it is
the closest honest analogue to a CT reconstruction that a brightfield stack
supports.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .focus import tenengrad


@dataclass
class FocusStack:
    edf: np.ndarray        # (Y, X) uint8 -- all-in-focus image
    best_z: np.ndarray     # (Y, X) int16 -- slice index of peak sharpness
    peak: np.ndarray       # (Y, X) float32 -- sharpness at that slice
    z_range: range

    @property
    def depth_map_um(self):
        raise NotImplementedError  # scale at call site; acq lives elsewhere


def build(volume: np.ndarray, z_range: range | None = None,
          smooth_sigma: float = 6.0, progress=None) -> FocusStack:
    """All-in-focus image + per-pixel depth of best focus.

    `smooth_sigma` blurs the sharpness map before the argmax. Raw per-pixel
    sharpness is far too noisy to compare across slices -- an organoid rim is
    sharp over a neighbourhood, not at an isolated pixel -- so the measure is
    pooled over roughly a rim's width first.

    Computed as a running maximum so the full sharpness volume never has to be
    held in memory.
    """
    zs = list(z_range if z_range is not None else range(volume.shape[0]))
    if not zs:
        raise ValueError("empty z range")

    h, w = volume.shape[1:]
    best_val = np.full((h, w), -np.inf, dtype=np.float32)
    best_z = np.zeros((h, w), dtype=np.int16)
    edf = np.zeros((h, w), dtype=np.uint8)

    for n, z in enumerate(zs):
        sl = volume[z]
        s = cv2.GaussianBlur(tenengrad(sl), (0, 0), smooth_sigma)
        better = s > best_val
        best_val[better] = s[better]
        best_z[better] = z
        edf[better] = sl[better]
        if progress:
            progress(n + 1, len(zs))

    return FocusStack(edf=edf, best_z=best_z, peak=best_val,
                      z_range=range(zs[0], zs[-1] + 1))
