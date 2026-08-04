"""Assembling the tiles into one picture, and flattening the illumination first.

Two jobs live here, and it is worth being clear that only one of them touches
the measurements.

**Flat-fielding does.** The illumination across a frame is not even. Averaging
all fifteen frames at one depth cancels the specimen and leaves the optical
pattern behind, and in this instrument that pattern is a smooth tilt spanning
about a hundred grey levels from one corner to the other. That matters more than
it sounds: an organoid in an overlap zone falls at a *different place in the
frame* in each of the two tiles that see it, so without correction the same
object has two different brightnesses and two different textures depending on
which tile is asked. Any classifier trained on those features would learn some
of the geometry of the scan. So intensity is corrected before anything is
measured from it, and the correction is estimated from the data rather than
modelled.

**Blending does not.** The feathered composite built here is for looking at --
QC overlays, the viewer, the figures. Detection and measurement always happen on
a single tile's own pixels, because a blended seam is a weighted average of two
slightly different views and its texture is an artefact of the blend rather than
a property of the specimen. Texture is precisely what the feature matrix is
trying to measure, so it is never measured on a blend.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .mosaic import Mosaic, Tile


# --------------------------------------------------------------------------- #
# flat field
# --------------------------------------------------------------------------- #

@dataclass
class FlatField:
    """A multiplicative illumination correction, estimated from the data.

    `gain` is normalised to a mean of one, so dividing by it flattens the field
    without changing the overall brightness scale -- grey levels stay
    comparable to the raw images, and to other runs.
    """

    gain: np.ndarray
    n_frames: int
    source: str
    span_before: float
    """Grey levels between the dimmest and brightest part of the raw pattern."""
    span_after: float
    """The same span once the correction is applied. Should be far smaller."""

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Correct one frame, returning float32 in the original grey range."""
        return image.astype(np.float32) / self.gain

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "n_frames": self.n_frames,
            "gain_min": round(float(self.gain.min()), 4),
            "gain_max": round(float(self.gain.max()), 4),
            "illumination_span_before": round(self.span_before, 1),
            "illumination_span_after": round(self.span_after, 1),
        }


def estimate_flat_field(mosaic: Mosaic, z_samples: int = 7,
                        smooth_sigma: float = 40.0,
                        progress=None) -> FlatField:
    """Recover the fixed illumination pattern from the frames themselves.

    The estimator is a per-pixel *median* across many frames, not a mean. The
    specimen is not uniformly distributed -- the droplet sits in the middle of
    the scan, so the centre tiles are full of organoids while the corner ones
    are nearly empty -- and a mean would fold that imbalance into the
    correction. A median across enough frames returns the background wherever
    the background is what most frames show.

    The result is then heavily smoothed. Illumination varies on the scale of the
    whole frame; anything sharper than that is specimen or sensor noise, and
    dividing by it would print the inverse of one organoid onto every other.
    """
    import cv2
    import tifffile as tiff

    frames: list[np.ndarray] = []
    for n, tile in enumerate(mosaic):
        files = sorted(tile.folder.glob("*.tif"))
        picks = np.linspace(0, len(files) - 1, z_samples).round().astype(int)
        for i in sorted(set(picks.tolist())):
            frames.append(tiff.imread(files[i]).astype(np.float32))
        if progress:
            progress(n + 1, len(mosaic))

    stack = np.stack(frames)
    pattern = np.median(stack, axis=0)
    smooth = cv2.GaussianBlur(pattern, (0, 0), smooth_sigma)

    gain = smooth / float(smooth.mean())
    # A frame that saw nothing at all would give a gain of zero somewhere and
    # turn the division into an infinity; clip well away from that.
    gain = np.clip(gain, 0.2, 5.0).astype(np.float32)

    before = float(smooth.max() - smooth.min())
    corrected = cv2.GaussianBlur(pattern / gain, (0, 0), smooth_sigma)
    after = float(corrected.max() - corrected.min())

    return FlatField(gain=gain, n_frames=len(frames),
                     source="per-pixel median across tiles and depths",
                     span_before=before, span_after=after)


# --------------------------------------------------------------------------- #
# feathered composition
# --------------------------------------------------------------------------- #

def feather_weights(height: int, width: int, overlap_x: float,
                    overlap_y: float) -> np.ndarray:
    """A weight map that fades a tile out over its overlapping margin.

    The ramp runs across the full width of the overlap, so where two tiles meet
    their weights sum to one everywhere and the composite has no step. Away from
    the margins the weight is flat, so a tile's interior is reproduced exactly
    rather than being scaled by however many neighbours it happens to have.
    """
    def ramp(n: int, margin: float) -> np.ndarray:
        w = np.ones(n, dtype=np.float32)
        m = int(round(margin))
        if m >= 1:
            edge = (np.arange(m, dtype=np.float32) + 0.5) / m
            w[:m] = edge
            w[-m:] = edge[::-1]
        return w

    wy = ramp(height, overlap_y)
    wx = ramp(width, overlap_x)
    # A floor keeps the far corners from reaching exactly zero, where a pixel
    # covered only by one tile's own margin would otherwise have no weight at
    # all and divide by zero.
    return np.maximum(np.outer(wy, wx), 1e-3)


class MosaicSlices:
    """Reads one depth of the assembled mosaic at a time.

    The whole mosaic is 750 million voxels, so it is never held. A single slice
    is six megapixels, which is nothing, and every consumer here -- the dome
    trace, the QC overlays, the viewer, the figures -- wants one depth at a
    time anyway.
    """

    def __init__(self, mosaic: Mosaic, flat: FlatField | None = None,
                 blend: bool = True):
        self.mosaic = mosaic
        self.flat = flat
        self.blend = blend
        self._files: dict[str, list[Path]] = {
            t.name: sorted(t.folder.glob("*.tif")) for t in mosaic}
        ox, oy = mosaic.overlap_px()
        self._weight = (feather_weights(mosaic.tile_height, mosaic.tile_width,
                                        ox, oy) if blend else None)

    @property
    def depth(self) -> int:
        return min(len(f) for f in self._files.values())

    def tile_slice(self, tile: Tile, z: int) -> np.ndarray:
        """One tile's own pixels at depth `z`, flat-fielded but not blended.

        This is what every measurement is made on.
        """
        import tifffile as tiff

        img = tiff.imread(self._files[tile.name][z])
        if img.ndim == 3:
            img = img.mean(axis=2)
        return self.flat.apply(img) if self.flat is not None else img.astype(np.float32)

    def slice(self, z: int) -> np.ndarray:
        """The assembled mosaic at depth `z`, as uint8, for display."""
        h, w = self.mosaic.height, self.mosaic.width
        acc = np.zeros((h, w), dtype=np.float32)
        wsum = np.zeros((h, w), dtype=np.float32)

        for tile in self.mosaic:
            img = self.tile_slice(tile, z)
            weight = (self._weight if self._weight is not None
                      else np.ones_like(img, dtype=np.float32))
            x0, y0 = int(round(tile.x0)), int(round(tile.y0))
            th, tw = img.shape
            acc[y0:y0 + th, x0:x0 + tw] += img * weight
            wsum[y0:y0 + th, x0:x0 + tw] += weight

        out = np.divide(acc, wsum, out=np.zeros_like(acc), where=wsum > 0)
        return np.clip(out, 0, 255).astype(np.uint8)

    def coverage(self) -> np.ndarray:
        """How many tiles see each pixel of the mosaic."""
        cov = np.zeros((self.mosaic.height, self.mosaic.width), dtype=np.uint8)
        for tile in self.mosaic:
            x0, y0 = int(round(tile.x0)), int(round(tile.y0))
            cov[y0:y0 + tile.height, x0:x0 + tile.width] += 1
        return cov
