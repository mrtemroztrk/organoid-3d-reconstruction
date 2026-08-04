"""Per-slice 2D segmentation.

Deliberately 2D. Each slice is segmented on its own and the slices are stitched
afterwards in `link.py`, the same way a CT volume is built from independently
segmented sections. Running a 3D segmenter directly on this stack does not work:
with a ~50 um depth of field every organoid leaves a shadow on ~50 slices, so a
3D connected component is a column, not a sphere.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import cv2
import numpy as np
from skimage import measure, morphology, segmentation
from scipy import ndimage as ndi

from .config import Params


@dataclass
class Detection:
    """One 2D object on one slice."""

    z: int
    label: int
    cx: float
    cy: float
    area_px: float
    radius_px: float          # equivalent-circle radius
    circularity: float
    contour: np.ndarray       # (N, 2) float array of (x, y), closed polygon


def _detections_from_labels(labels: np.ndarray, z: int, p: Params,
                            min_r: float, max_r: float) -> list[Detection]:
    out: list[Detection] = []
    for reg in measure.regionprops(labels):
        area = float(reg.area)
        radius = float(np.sqrt(area / np.pi))
        if not (min_r <= radius <= max_r):
            continue

        perim = float(reg.perimeter) or 1.0
        circ = float(4.0 * np.pi * area / (perim * perim))
        if circ < p.min_circularity:
            continue

        mask = (labels == reg.label).astype(np.uint8)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        contour = max(cnts, key=cv2.contourArea).reshape(-1, 2).astype(np.float32)
        if contour.shape[0] < 8:
            continue

        cy, cx = reg.centroid
        out.append(Detection(
            z=z, label=int(reg.label), cx=float(cx), cy=float(cy),
            area_px=area, radius_px=radius, circularity=circ, contour=contour,
        ))
    return out


# --------------------------------------------------------------------------- #
# Cellpose-SAM
# --------------------------------------------------------------------------- #

class CellposeDetector:
    """Cellpose-SAM run slice by slice.

    `do_3D` and `stitch_threshold` are intentionally not used: they assume
    consecutive slices are genuinely different sections of the object. Here they
    are not -- they are the same object at different amounts of blur -- so
    Cellpose's own stitching would happily fuse an organoid with the out-of-focus
    haze of its neighbours. Linking is done in `link.py` with a focus criterion
    Cellpose has no notion of.
    """

    def __init__(self, acq, params: Params, gpu: bool = True):
        from cellpose import models

        self.p = params
        self.acq = acq
        self.diameter_px = params.expected_diameter_px
        self.cellprob_threshold = params.cellprob_threshold
        self.flow_threshold = params.flow_threshold
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model = models.CellposeModel(gpu=gpu)

    def segment(self, img: np.ndarray) -> np.ndarray:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            masks, _, _ = self.model.eval(
                img,
                diameter=self.diameter_px,
                batch_size=2,          # 4 GB GPU
                flow_threshold=self.flow_threshold,
                cellprob_threshold=self.cellprob_threshold,
                normalize=True,
            )
        return masks.astype(np.int32)


# --------------------------------------------------------------------------- #
# Classical fallback (no GPU / no cellpose)
# --------------------------------------------------------------------------- #

class ClassicalDetector:
    """Edge-energy + watershed segmentation.

    Keys off the fact that an in-focus organoid is a closed, high-contrast rim:
    flatten the illumination, measure local edge energy, keep the sharp rims,
    close them into discs, then split touching discs by distance-transform
    watershed.
    """

    def __init__(self, acq, params: Params):
        self.p = params
        self.acq = acq

    def segment(self, img: np.ndarray) -> np.ndarray:
        p, acq = self.p, self.acq
        f = img.astype(np.float32)

        # 1. flatten uneven illumination / the dome's own shading
        bg = cv2.GaussianBlur(f, (0, 0), sigmaX=40)
        flat = f - bg

        # 2. edge energy, smoothed to the scale of a rim
        gx = cv2.Sobel(flat, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(flat, cv2.CV_32F, 0, 1, ksize=3)
        edge = cv2.GaussianBlur(np.sqrt(gx * gx + gy * gy), (0, 0), sigmaX=2.0)

        # 3. keep only the sharpest rims on this slice
        thr = float(np.percentile(edge, 96.0))
        rim = edge > thr

        # 4. close rims into filled discs
        min_r = 0.5 * p.min_diameter_px
        k = max(3, int(round(min_r)))
        rim = morphology.binary_closing(rim, morphology.disk(k))
        filled = ndi.binary_fill_holes(rim)
        filled = morphology.remove_small_objects(filled, int(np.pi * min_r ** 2))
        filled = morphology.binary_opening(filled, morphology.disk(max(2, k // 2)))

        # 5. split touching organoids
        dist = ndi.distance_transform_edt(filled)
        peak_sep = max(3, int(round(0.5 * p.expected_diameter_px)))
        from skimage.feature import peak_local_max
        coords = peak_local_max(dist, min_distance=peak_sep, labels=filled,
                                exclude_border=False)
        markers = np.zeros(dist.shape, dtype=np.int32)
        for i, (y, x) in enumerate(coords, start=1):
            markers[y, x] = i
        if markers.max() == 0:
            return measure.label(filled).astype(np.int32)
        return segmentation.watershed(-dist, markers, mask=filled).astype(np.int32)


def build_detector(acq, params: Params, gpu: bool = True):
    if params.detector == "cellpose":
        try:
            return CellposeDetector(acq, params, gpu=gpu)
        except Exception as exc:  # noqa: BLE001 - fall back rather than abort
            print(f"  ! Cellpose unavailable ({exc}); falling back to classical detector")
            return ClassicalDetector(acq, params)
    return ClassicalDetector(acq, params)


def segment_stack(stack, params: Params, gpu: bool = True,
                  z_slice_range: range | None = None,
                  progress=None) -> tuple[np.ndarray, list[list[Detection]]]:
    """Segment every (z_step-th) slice.

    Returns (labels volume aligned to the stack, per-slice detections).
    Slices that are skipped stay empty; linking tolerates the gaps.
    """
    detector = build_detector(stack.acq, params, gpu=gpu)
    acq = stack.acq
    min_r = 0.5 * params.min_diameter_px
    max_r = 0.5 * params.max_diameter_px

    zs = list(z_slice_range if z_slice_range is not None else range(stack.depth))
    zs = zs[:: params.z_step]

    labels = np.zeros(stack.data.shape, dtype=np.int32)
    per_slice: list[list[Detection]] = [[] for _ in range(stack.depth)]

    for n, z in enumerate(zs):
        lab = detector.segment(stack.data[z])
        labels[z] = lab
        per_slice[z] = _detections_from_labels(lab, z, params, min_r, max_r)
        if progress:
            progress(n + 1, len(zs), z, len(per_slice[z]))

    return labels, per_slice
