"""Focus measurement -- the backbone of the whole reconstruction.

A 4x / NA 0.20 brightfield stack has a ~50 um depth of field, so a slice is not
an optical section: it is a shadowgram of the entire dome. What *does* vary with
Z is sharpness. An organoid sitting at depth z0 shows a crisp dark rim on the
slice nearest z0 and a diffuse grey disc everywhere else.

So instead of asking "is there signal at (x, y, z)?" (which smears every object
into a Z column) we ask "at which z is the rim at (x, y) sharpest?". That is
shape-from-focus, and it is the only depth cue this modality actually provides.
"""
from __future__ import annotations

import cv2
import numpy as np


def tenengrad(img: np.ndarray) -> np.ndarray:
    """Per-pixel squared gradient magnitude (Sobel energy)."""
    f = img.astype(np.float32)
    gx = cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=3)
    return gx * gx + gy * gy


def abs_laplacian(img: np.ndarray) -> np.ndarray:
    """Per-pixel |Laplacian|. Sharper peak than Tenengrad, more noise-sensitive."""
    return np.abs(cv2.Laplacian(img.astype(np.float32), cv2.CV_32F, ksize=3))


def global_profile(stack: np.ndarray) -> np.ndarray:
    """Mean Tenengrad per slice -- one number per Z."""
    return np.array([float(tenengrad(stack[z]).mean()) for z in range(stack.shape[0])],
                    dtype=np.float64)


def find_substrate_plane(profile: np.ndarray, search_from: float = 0.5) -> int:
    """Index of the glass/plastic bottom of the well.

    The dish surface carries fine debris and Matrigel texture that is far
    sharper than anything biological, so it dominates the global focus profile.
    In a top-to-bottom stack it appears in the lower half; everything past it is
    outside the sample.

    Returns the 0-based slice index of the peak.
    """
    lo = int(len(profile) * search_from)
    tail = profile[lo:]
    if tail.size == 0:
        return len(profile) - 1
    return int(lo + np.argmax(tail))


def focus_curve(stack: np.ndarray, band: np.ndarray, z_range: range | None = None) -> np.ndarray:
    """Sharpness of one object's rim as a function of Z.

    `band` is a boolean mask of the ring straddling the object's outline; the
    rim is where brightfield contrast lives, and for cystic organoids the
    interior is featureless, so measuring the band beats measuring the disc.
    """
    zs = z_range if z_range is not None else range(stack.shape[0])
    ys, xs = np.nonzero(band)
    if ys.size == 0:
        return np.zeros(len(zs), dtype=np.float64)

    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    sub_band = band[y0:y1, x0:x1]

    out = np.empty(len(zs), dtype=np.float64)
    for i, z in enumerate(zs):
        patch = stack[z, y0:y1, x0:x1]
        out[i] = float(abs_laplacian(patch)[sub_band].mean())
    return out


def contour_band(mask: np.ndarray, half_width: int = 4) -> np.ndarray:
    """Boolean ring of width 2*half_width centred on the mask boundary."""
    m = mask.astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * half_width + 1,) * 2)
    return (cv2.dilate(m, k) > 0) & (cv2.erode(m, k) == 0)


def refine_at(curve: np.ndarray, i: int) -> float:
    """Sub-slice position of the peak at index `i`, by parabolic fit.

    The depth of field spans several slices, so the focus curve is smooth and
    interpolating it recovers depth finer than the raw Z step.
    """
    if i <= 0 or i >= curve.size - 1:
        return float(i)
    a, b, c = curve[i - 1], curve[i], curve[i + 1]
    denom = a - 2.0 * b + c
    if abs(denom) < 1e-12:
        return float(i)
    return float(i) + float(np.clip(0.5 * (a - c) / denom, -1.0, 1.0))


def refine_peak(curve: np.ndarray) -> tuple[float, float]:
    """(sub-slice argmax, peak value)."""
    if curve.size == 0:
        return 0.0, 0.0
    i = int(np.argmax(curve))
    return refine_at(curve, i), float(curve[i])


def find_focal_planes(curve: np.ndarray, min_separation: float,
                      min_prominence_frac: float = 0.15) -> list[tuple[float, float]]:
    """Every distinct focal plane along one lateral position.

    A single track can contain more than one organoid: two of them sitting at
    the same (x, y) but different depths are laterally indistinguishable, so the
    linker follows them as one object. They are separable in focus, though --
    each produces its own peak. Splitting on prominent, well-separated peaks
    recovers both instead of silently reporting one.

    `min_separation` is in slices and should be at least the depth of field,
    below which two peaks cannot be told apart anyway.

    Only interior maxima count. A curve that is still climbing when it hits the
    end of the search window has no measured focal plane -- the true one lies
    outside the analysed range, either above the top of the stack or down on the
    glass. Accepting the boundary value instead would pile those objects onto
    the last slice and report a depth the stack never recorded.

    Returns [(sub-slice position, prominence / max), ...], strongest first.
    Empty means "this position has no focal plane in range".
    """
    if curve.size < 3:
        return []

    from scipy.signal import find_peaks

    mx = float(curve.max())
    if mx <= 1e-12:
        return []

    span = mx - float(curve.min())
    idx, props = find_peaks(
        curve,
        prominence=max(1e-9, min_prominence_frac * span),
        distance=max(1, int(round(min_separation))),
    )
    if idx.size == 0:
        return []

    out = [(refine_at(curve, int(i)), float(p) / mx)
           for i, p in zip(idx, props["prominences"])]
    out.sort(key=lambda t: -t[1])
    return out


def sharpness_ratio(curve: np.ndarray) -> float:
    """How peaked the focus curve is: (max - median) / max, in [0, 1].

    A real object in focus produces a distinct peak. Flat background texture, or
    a blur artefact with no true focal plane, produces a flat curve. This is the
    single most useful score for throwing out false positives.
    """
    if curve.size == 0:
        return 0.0
    mx = float(curve.max())
    if mx <= 1e-12:
        return 0.0
    return float((mx - np.median(curve)) / mx)
