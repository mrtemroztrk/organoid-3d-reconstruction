"""Re-measuring an organoid's outline on the plane where it is in focus.

An object found on the all-in-focus projection arrives with a contour drawn on
an image in which every depth is superimposed. Sweeping that contour through Z
finds the depth where its rim is sharpest, which is the right depth -- but the
outline itself is never revisited, so the shape reported at that plane was
measured somewhere else. Where two organoids overlap laterally at different
depths, the projection merges them and the outline belongs to neither.

That is the wrong edge in the most literal sense: it is drawn at the focal plane
and measured on a different image.

The fix is to keep the projection's contour as a *seed* and then move each
radius onto the rim as it appears on that organoid's own focal slice. Working in
r(theta) rather than re-segmenting is deliberate. The package already represents
every outline that way, the search stays inside a band around a shape that is
known to be roughly right, and a ray leaving the centre of a near-convex object
crosses its boundary exactly once -- so there is no ambiguity about which edge
is meant. A fresh segmentation of a crop has no such guarantee and would happily
return the neighbour.

What the rim looks like matters. In brightfield an organoid is a dark ring
around a lighter interior, sitting on a lighter background, so travelling
outwards along a ray the intensity climbs steeply as the ray leaves the rim.
That rising edge is what is tracked; the darkest point is not, because the rim
has width and its darkest part lies inside the boundary.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Refinement:
    """The outcome of moving one outline onto its focal plane."""

    r_theta: np.ndarray
    moved_px: float
    """Median distance each radius travelled."""
    response_before: float
    response_after: float
    """Mean edge strength under the outline, before and after. If refinement did
    anything worth doing, the second is larger; the test asserts it."""
    accepted: bool
    reason: str = ""

    @property
    def gain(self) -> float:
        return self.response_after / max(self.response_before, 1e-9)


def _smooth_periodic(values: np.ndarray, keep: int) -> np.ndarray:
    """Low-pass r(theta) by keeping only its lowest angular harmonics.

    A per-ray maximum is chosen independently on every ray, so a neighbouring
    organoid clipping one ray can pull a single radius far out and leave a spike
    that no cell membrane could make. Organoid outlines are smooth at this
    scale, so discarding the high harmonics removes the spike without imposing a
    circle -- the low harmonics are exactly the elongation and lobing that the
    shape features are meant to measure.
    """
    spectrum = np.fft.rfft(values)
    spectrum[keep + 1:] = 0.0
    return np.fft.irfft(spectrum, values.size)


def refine(image: np.ndarray, cx: float, cy: float, r_theta,
           band: float = 0.35, n_radius: int = 41, keep_harmonics: int = 6,
           min_radius_px: float = 3.0) -> Refinement:
    """Move a seed outline onto the rim as it appears in `image`.

    `band` is how far each radius may travel, as a fraction of itself. It is
    deliberately not generous: the seed came from a projection in which this
    organoid really is present, so the boundary is nearby, and a wide search
    only offers more chances to lock onto a neighbour.
    """
    seed = np.asarray(r_theta, dtype=np.float64)
    n_theta = seed.size
    if n_theta < 8 or not np.all(np.isfinite(seed)) or seed.mean() < min_radius_px:
        return Refinement(seed, 0.0, 0.0, 0.0, False, "seed outline unusable")

    angles = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    scale = np.linspace(1.0 - band, 1.0 + band, n_radius)
    radii = np.outer(seed, scale)                       # (n_theta, n_radius)

    xs = (cx + radii * np.cos(angles)[:, None]).astype(np.float32)
    ys = (cy + radii * np.sin(angles)[:, None]).astype(np.float32)
    h, w = image.shape[:2]
    if (xs.min() < 1 or ys.min() < 1 or xs.max() > w - 2 or ys.max() > h - 2):
        return Refinement(seed, 0.0, 0.0, 0.0, False, "search band leaves the frame")

    import cv2

    profile = cv2.remap(image.astype(np.float32), xs, ys, cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_REPLICATE)
    # Outward intensity rise: the boundary is where the ray leaves the dark rim
    # for the lighter surroundings. A central difference along the ray, smoothed
    # a little, because a single-pixel derivative on an 8-bit image is mostly
    # noise.
    gradient = np.gradient(profile, axis=1)
    gradient = cv2.GaussianBlur(gradient, (5, 1), 0)

    refined = seed * scale[np.argmax(gradient, axis=1)]
    refined = _smooth_periodic(refined, keep_harmonics)
    refined = np.clip(refined, seed * (1.0 - band), seed * (1.0 + band))

    # Both outlines are scored the same way, on the isotropic edge magnitude of
    # this slice, because the point of the comparison is "which of these two
    # curves lies on a boundary" and that has to be one question. The smoothed
    # curve is what gets scored, not the per-ray maxima it came from, since the
    # smoothed curve is what is returned.
    mag = gradient_image(image)
    before = _edge_strength(mag, cx, cy, seed, angles)
    after = _edge_strength(mag, cx, cy, refined, angles)
    moved = float(np.median(np.abs(refined - seed)))

    if after <= before:
        return Refinement(seed, 0.0, before, after, False,
                          "the seed already lay on a stronger edge")
    return Refinement(refined, moved, before, after, True)


def gradient_image(image: np.ndarray) -> np.ndarray:
    """Radial-agnostic edge magnitude, used only for scoring."""
    import cv2

    f = image.astype(np.float32)
    gx = cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=3)
    return np.hypot(gx, gy)


def _edge_strength(mag: np.ndarray, cx: float, cy: float, r_theta: np.ndarray,
                   angles: np.ndarray) -> float:
    """Mean edge magnitude along an outline -- how well it sits on a boundary."""
    xs = (cx + r_theta * np.cos(angles)).astype(np.float32)[:, None]
    ys = (cy + r_theta * np.sin(angles)).astype(np.float32)[:, None]
    import cv2

    return float(cv2.remap(mag, xs, ys, cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REPLICATE).mean())
