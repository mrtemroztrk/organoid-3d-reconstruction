"""Refining the tile offsets against the pixels.

The stage log says where the microscope meant to be. It is very nearly right --
across this scan the steps are regular to the nanometre it reports -- but "very
nearly" is not the same as right, and correlating the overlapping strips finds
disagreements of several pixels. At 3.77 um per pixel that is over thirty
microns, which is a fair fraction of a small organoid, and it is the difference
between recognising the same object in two tiles and reporting it twice.

So the metadata is treated as a starting guess and the pixels get the last word.

Two properties of this particular data shape the method. First, the overlapping
strips are large -- 288 by 720 pixels, nearly a third of a frame -- so there is
plenty to match on wherever the specimen has structure. Second, there are places
with no structure at all: the bottom row of tiles lies almost entirely on empty
background beyond the droplet, and asking those strips where they line up
returns an answer with no evidence behind it. A method that believed every pair
equally would let those three pairs drag the whole mosaic. Every pair therefore
carries a confidence, and the global fit is weighted by it.

The fit is global rather than sequential for the same reason. Chaining offsets
tile by tile accumulates error along whatever path is chosen and gives a
different answer for a different path. Solving all fifteen positions at once
against all twenty-two measured pairs distributes the disagreement, and what is
left over is the residual -- the number that says whether the mosaic can be
trusted at all.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .blend import MosaicSlices
from .mosaic import REGISTERED, Mosaic


@dataclass
class PairShift:
    """How far one tile has to move to line up with its neighbour.

    `dx` and `dy` are corrections to be *added* to the second tile's offset.
    """

    a: str
    b: str
    axis: str

    dx: float
    dy: float
    confidence: float
    """0 to 1. Driven by how well the strips matched and how consistently they
    agreed from one depth to the next."""

    n_planes: int
    """Depths whose match was good enough to contribute."""
    scatter_px: float
    """Spread of the per-depth answers. Large means the strips did not really
    agree with each other, whatever the individual matches claimed."""

    residual_px: float = 0.0
    """Disagreement with the globally solved offsets, filled in after the fit."""

    def to_dict(self) -> dict:
        return {"a": self.a, "b": self.b, "axis": self.axis,
                "dx_px": round(self.dx, 3), "dy_px": round(self.dy, 3),
                "confidence": round(self.confidence, 4),
                "n_planes": self.n_planes,
                "scatter_px": round(self.scatter_px, 3),
                "residual_px": round(self.residual_px, 3)}


@dataclass
class Registration:
    """The outcome of refining every tile offset at once."""

    shifts: list[PairShift]
    solution: dict[str, tuple[float, float]] = field(default_factory=dict)
    """The solved offset of each tile, before it is rebased to the origin."""
    moved_px: dict[str, tuple[float, float]] = field(default_factory=dict)
    """How far each tile ended up from where the stage log put it."""

    residual_px: float = 0.0
    residual_p90_px: float = 0.0
    residual_max_px: float = 0.0
    n_used: int = 0
    n_rejected: int = 0
    tolerance_px: float = 2.0

    @property
    def reliable(self) -> bool:
        """Did the pairs agree well enough to believe the assembled frame?

        The residual is what one pair says about a tile minus what all the
        others say. Below a pixel or two it is measurement noise. Well above it,
        the pairs are describing incompatible geometries and no single set of
        offsets can satisfy them -- which usually means the specimen moved
        during the scan, and no amount of fitting will rescue it.
        """
        return self.n_used >= 2 and self.residual_p90_px <= self.tolerance_px

    def describe(self) -> str:
        moved = np.array([np.hypot(dx, dy) for dx, dy in self.moved_px.values()])
        lines = [
            f"{self.n_used} of {self.n_used + self.n_rejected} neighbour pairs used",
            f"  residual  median {self.residual_px:.2f} px, "
            f"p90 {self.residual_p90_px:.2f} px, worst {self.residual_max_px:.2f} px",
            f"  tiles moved {moved.mean():.1f} px on average from the stage log, "
            f"up to {moved.max():.1f} px",
        ]
        if not self.reliable:
            lines.append("  !! the pairs do not agree on one geometry; the mosaic "
                         "is not trustworthy at this residual")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "residual_px": round(self.residual_px, 4),
            "residual_p90_px": round(self.residual_p90_px, 4),
            "residual_max_px": round(self.residual_max_px, 4),
            "tolerance_px": self.tolerance_px,
            "n_used": self.n_used,
            "n_rejected": self.n_rejected,
            "reliable": self.reliable,
            "moved_px": {k: [round(v[0], 3), round(v[1], 3)]
                         for k, v in self.moved_px.items()},
            "pairs": [s.to_dict() for s in self.shifts],
        }


# --------------------------------------------------------------------------- #
# matching one pair of strips
# --------------------------------------------------------------------------- #

def _match(strip_a: np.ndarray, strip_b: np.ndarray,
           upsample: int = 20) -> tuple[float, float, float]:
    """Align two views of the same patch; returns (dy, dx, quality).

    Plain normalised cross-correlation, not the phase-normalised variant. Phase
    normalisation weights every spatial frequency equally, which is the right
    thing for a broadband image and the wrong thing here: a brightfield frame at
    4x is smooth, most of its high frequencies are noise, and lifting them to
    equal footing makes the correlation peak wander. On a synthetic pair with a
    known five-pixel offset the phase-normalised match returned zero shift while
    the plain one recovered the offset to a twentieth of a pixel.
    """
    from skimage.registration import phase_cross_correlation

    a = strip_a.astype(np.float32)
    b = strip_b.astype(np.float32)
    if a.std() < 1e-3 or b.std() < 1e-3:
        return 0.0, 0.0, 0.0

    shift, error, _ = phase_cross_correlation(a, b, upsample_factor=upsample,
                                              normalization=None)
    return float(shift[0]), float(shift[1]), float(np.clip(1.0 - error, 0.0, 1.0))


def _strips(image_a: np.ndarray, image_b: np.ndarray, tile_a, tile_b, axis: str):
    """The overlapping region of two neighbours, as seen by each of them."""
    if axis == "x":
        w = int(round(tile_a.width - (tile_b.x0 - tile_a.x0)))
        w = max(8, min(w, tile_a.width, tile_b.width))
        return image_a[:, -w:], image_b[:, :w]
    h = int(round(tile_a.height - (tile_b.y0 - tile_a.y0)))
    h = max(8, min(h, tile_a.height, tile_b.height))
    return image_a[-h:, :], image_b[:h, :]


def measure_pairs(mosaic: Mosaic, slices: MosaicSlices, z_samples: int = 9,
                  min_quality: float = 0.15, progress=None) -> list[PairShift]:
    """Correlate every pair of neighbours at several depths.

    Depth matters. Near the glass the whole field is sharp and matches easily;
    high above the droplet the strips are a smooth blur with nothing to lock
    onto. Rather than choose one plane and hope, each pair is measured at
    several and the answers are combined robustly, which also yields the
    scatter -- a pair whose planes disagree with each other has not really found
    anything, however confident any single plane looked.
    """
    depth = slices.depth
    planes = np.linspace(0, depth - 1, z_samples).round().astype(int)
    planes = sorted(set(int(p) for p in planes))

    pairs = mosaic.neighbour_pairs()
    votes: dict[tuple[str, str], list[tuple[float, float, float]]] = {
        (a.name, b.name): [] for a, b, _ in pairs}

    for n, z in enumerate(planes):
        frames = {t.name: slices.tile_slice(t, z) for t in mosaic}
        for a, b, axis in pairs:
            sa, sb = _strips(frames[a.name], frames[b.name], a, b, axis)
            dy, dx, quality = _match(sa, sb)
            if quality >= min_quality:
                votes[(a.name, b.name)].append((dy, dx, quality))
        if progress:
            progress(n + 1, len(planes))

    out: list[PairShift] = []
    for a, b, axis in pairs:
        v = votes[(a.name, b.name)]
        if not v:
            out.append(PairShift(a.name, b.name, axis, 0.0, 0.0, 0.0, 0, np.inf))
            continue
        arr = np.array(v)
        dy, dx = float(np.median(arr[:, 0])), float(np.median(arr[:, 1]))
        scatter = float(np.median(np.hypot(arr[:, 0] - dy, arr[:, 1] - dx)))
        # A pair is only as good as its agreement with itself. Quality says the
        # strips matched; scatter says they matched the *same way* at every
        # depth, which is the part that cannot be faked by a smooth gradient.
        agreement = 1.0 / (1.0 + scatter)
        confidence = float(np.median(arr[:, 2]) * agreement)
        out.append(PairShift(a.name, b.name, axis, dx, dy, confidence,
                             len(v), scatter))
    return out


# --------------------------------------------------------------------------- #
# solving all the offsets at once
# --------------------------------------------------------------------------- #

def solve(mosaic: Mosaic, shifts: list[PairShift], min_confidence: float = 0.05,
          tolerance_px: float = 2.0) -> Registration:
    """Least-squares offsets for every tile from the measured pairwise shifts.

    Each usable pair contributes one equation per axis: the difference between
    two tiles' offsets should equal the shift measured between them. The system
    is overdetermined -- twenty-two pairs constrain fifteen positions -- so it
    is solved rather than chained, and the leftover disagreement is reported
    instead of being hidden by a particular traversal order.

    The first tile is pinned. Only relative positions are observable from
    overlaps, so without pinning one tile the whole mosaic could slide and the
    system would be singular.
    """
    names = [t.name for t in mosaic]
    index = {n: i for i, n in enumerate(names)}
    usable = [s for s in shifts if s.confidence >= min_confidence and s.n_planes]

    reg = Registration(shifts=shifts, tolerance_px=tolerance_px,
                       n_used=len(usable), n_rejected=len(shifts) - len(usable))
    if len(usable) < 2:
        return reg

    rows = len(usable) + 1
    a = np.zeros((rows, len(names)), dtype=float)
    bx = np.zeros(rows)
    by = np.zeros(rows)
    w = np.zeros(rows)

    base = {t.name: (t.x0, t.y0) for t in mosaic}
    for k, s in enumerate(usable):
        i, j = index[s.a], index[s.b]
        a[k, i], a[k, j] = -1.0, 1.0
        # The measured shift corrects where tile b sits, so the offset
        # difference the pair asks for is its current difference plus that.
        bx[k] = base[s.b][0] - base[s.a][0] + s.dx
        by[k] = base[s.b][1] - base[s.a][1] + s.dy
        w[k] = s.confidence

    a[-1, 0] = 1.0                        # pin the first tile where it is
    bx[-1], by[-1] = base[names[0]][0], base[names[0]][1]
    w[-1] = max(1.0, w[:-1].max())

    aw = a * w[:, None]
    x, *_ = np.linalg.lstsq(aw, bx * w, rcond=None)
    y, *_ = np.linalg.lstsq(aw, by * w, rcond=None)

    residuals = []
    for s in usable:
        i, j = index[s.a], index[s.b]
        rx = (x[j] - x[i]) - (base[s.b][0] - base[s.a][0] + s.dx)
        ry = (y[j] - y[i]) - (base[s.b][1] - base[s.a][1] + s.dy)
        s.residual_px = float(np.hypot(rx, ry))
        residuals.append(s.residual_px)

    reg.solution = {n: (float(x[i]), float(y[i])) for n, i in index.items()}
    reg.moved_px = {n: (float(x[i]) - base[n][0], float(y[i]) - base[n][1])
                    for n, i in index.items()}
    reg.residual_px = float(np.median(residuals))
    reg.residual_p90_px = float(np.percentile(residuals, 90))
    reg.residual_max_px = float(np.max(residuals))
    return reg


def apply(mosaic: Mosaic, reg: Registration) -> Mosaic:
    """Move the tiles onto the solved offsets, and record that it happened.

    The mosaic is rebased so the top-left tile sits at the origin again; only
    relative positions were ever measured, and leaving a small global shift
    behind would make the canvas quietly larger than the specimen.
    """
    solution = getattr(reg, "solution", None)
    if not solution or not reg.reliable:
        return mosaic

    x0 = min(v[0] for v in solution.values())
    y0 = min(v[1] for v in solution.values())
    for tile in mosaic:
        if tile.name in solution:
            tile.x0 = solution[tile.name][0] - x0
            tile.y0 = solution[tile.name][1] - y0
            tile.offset_source = REGISTERED
    mosaic.notes.append(
        f"offsets refined against the overlapping pixels; "
        f"median pair residual {reg.residual_px:.2f} px")
    return mosaic


def register(mosaic: Mosaic, slices: MosaicSlices, z_samples: int = 9,
             tolerance_px: float = 2.0, progress=None
             ) -> tuple[Mosaic, Registration]:
    """Measure, solve, and apply -- the whole refinement in one call."""
    shifts = measure_pairs(mosaic, slices, z_samples=z_samples, progress=progress)
    reg = solve(mosaic, shifts, tolerance_px=tolerance_px)
    return apply(mosaic, reg), reg
