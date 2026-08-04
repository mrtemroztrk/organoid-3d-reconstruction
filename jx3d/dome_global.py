"""Fitting the Matrigel droplet across the whole mosaic.

`jx3d.dome` fits a spherical cap to one field of view. It has to, because one
field is all it can see, and it is careful to report how badly constrained that
is. It is worth being precise about the problem: a 960 px field spans about a
tenth of this droplet, so the fit is extrapolating a 4 mm radius from a short
arc. The three per-field fits already on disk return contact radii of 932, 982
and 1020 px while the whole mosaic gives 1061 px -- all three low, and each one
quoting a bootstrap spread of a quarter of a percent. The error was not being
underestimated by a little. It was being underestimated by a factor of twenty,
and it was a bias rather than noise, so averaging the fifteen fits would have
preserved it while making the error bar look better.

With all fifteen tiles the entire footprint is inside the frame. Every azimuth
returns a rim, so the contact circle stops being an extrapolation and becomes a
measurement. That is the whole reason the mosaic matters for this: not more
pixels, but a closed curve instead of an arc.

The trace itself follows from what the interface is. Where the curved gel/medium
boundary crosses the focal plane it leaves a broad band of high texture, and the
gel *ends* at the far side of that band, so the rim is the band's outermost
edge. Once a global axis exists, "outermost" finally has a meaning: sample each
slice along rays leaving the axis and take the last crossing on each ray. That
also disposes of the interior tiles, which contain no interface at all and whose
first threshold crossing is meaningless -- discarded by construction, because it
is never the outermost.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .blend import MosaicSlices
from .dome import Dome
from .focus import tenengrad
from .mosaic import Mosaic


@dataclass
class Substrate:
    """The glass plane, agreed on by the tiles that could actually see it."""

    slice_index: float
    votes: dict[str, float]
    rejected: dict[str, float]
    contrast: dict[str, float]

    @property
    def spread_slices(self) -> float:
        v = list(self.votes.values())
        return float(max(v) - min(v)) if v else float("nan")

    def to_dict(self) -> dict:
        return {
            "slice_index": round(self.slice_index, 3),
            "n_votes": len(self.votes),
            "n_rejected": len(self.rejected),
            "spread_slices": round(self.spread_slices, 3),
            "votes": {k: round(v, 3) for k, v in self.votes.items()},
            "rejected": {k: round(v, 3) for k, v in self.rejected.items()},
        }

    def describe(self) -> str:
        lines = [f"glass at slice {self.slice_index + 1:.1f} "
                 f"(median of {len(self.votes)} tiles, spread "
                 f"{self.spread_slices:.1f} slices)"]
        for name, value in sorted(self.rejected.items()):
            lines.append(f"  {name} did not see it (contrast "
                         f"{self.contrast[name]:.2f}) and was left out")
        return "\n".join(lines)


@dataclass
class Ring:
    """The droplet's cross-section on one slice, fitted independently."""

    z: int
    cx: float
    cy: float
    radius_px: float
    residual_px: float
    n_azimuths: int
    used: bool = True

    def to_dict(self) -> dict:
        return {"z": int(self.z), "cx_px": round(self.cx, 2),
                "cy_px": round(self.cy, 2), "radius_px": round(self.radius_px, 2),
                "residual_px": round(self.residual_px, 2),
                "n_azimuths": self.n_azimuths, "used": self.used}


@dataclass
class GlobalFit:
    """Everything the fit learned, kept so the number can be argued with."""

    dome: Dome | None
    substrate: Substrate
    rings: list[Ring] = field(default_factory=list)
    axis_scatter_px: tuple[float, float] = (0.0, 0.0)
    """Spread of the per-slice circle centres. Each slice measures the axis
    independently, so this is a direct estimate of how well it is pinned down --
    and unlike a bootstrap it cannot flatter itself, because the slices really
    are separate measurements."""
    cap_residual_px: float = 0.0

    def to_dict(self) -> dict:
        return {
            "dome": self.dome.to_dict() if self.dome else None,
            "substrate": self.substrate.to_dict(),
            "axis_scatter_px": [round(v, 3) for v in self.axis_scatter_px],
            "cap_residual_px": round(self.cap_residual_px, 3),
            "rings": [r.to_dict() for r in self.rings],
        }


# --------------------------------------------------------------------------- #
# the glass plane
# --------------------------------------------------------------------------- #

def find_substrate(mosaic: Mosaic, slices: MosaicSlices,
                   search_from: float = 0.6, contrast_min: float = 1.5,
                   progress=None) -> Substrate:
    """One glass plane for the whole mosaic, by consensus among the tiles.

    The tiles were all captured at the same stage Z, so the well bottom is
    physically one plane and there is exactly one right answer. Asking each tile
    separately does not give fifteen noisy estimates of it; it gives thirteen
    good ones and two that are wrong, because under the thickest part of the
    droplet the glass is buried under a millimetre of gel and organoids and its
    reflection never becomes the sharpest thing in the field. Those two tiles do
    not deserve a vote, and a contrast gate is what withholds it: a tile whose
    focus profile has no prominent peak has not seen the glass, whatever its
    argmax happens to be.
    """
    votes: dict[str, float] = {}
    rejected: dict[str, float] = {}
    contrast: dict[str, float] = {}

    depth = slices.depth
    lo = int(depth * search_from)

    for n, tile in enumerate(mosaic):
        profile = np.array([float(tenengrad(slices.tile_slice(tile, z)).mean())
                            for z in range(lo, depth)])
        peak = float(profile.max())
        ratio = peak / max(float(np.median(profile)), 1e-9)
        contrast[tile.name] = ratio

        i = int(np.argmax(profile))
        if 0 < i < profile.size - 1:
            a, b, c = profile[i - 1], profile[i], profile[i + 1]
            denom = a - 2.0 * b + c
            i_ref = i + (0.5 * (a - c) / denom if abs(denom) > 1e-12 else 0.0)
        else:
            i_ref = float(i)

        if ratio >= contrast_min:
            votes[tile.name] = lo + i_ref
        else:
            rejected[tile.name] = lo + i_ref
        if progress:
            progress(n + 1, len(mosaic))

    plane = float(np.median(list(votes.values()))) if votes else float(depth - 1)
    return Substrate(plane, votes, rejected, contrast)


# --------------------------------------------------------------------------- #
# tracing the rim
# --------------------------------------------------------------------------- #

def _seed_axis(mosaic: Mosaic, slices: MosaicSlices, sigma: float,
               threshold_k: float, z_planes: list[int]) -> tuple[float, float]:
    """A first guess at the droplet's axis, from where the texture sits.

    Only a starting point is needed: the trace refines it. The upper slices are
    used because there the medium above the droplet is smooth and the gel stands
    out from it clearly, whereas near the glass the whole field carries debris
    and the droplet stops being the only textured thing in the frame.
    """
    import cv2

    xs, ys, ws = [], [], []
    for z in z_planes:
        img = slices.slice(z).astype(np.float32)
        small = cv2.resize(img, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
        t = np.log1p(cv2.GaussianBlur(tenengrad(small), (0, 0), sigma * 0.25))
        hot = t > (float(np.median(t)) + threshold_k * float(t.std()))
        if hot.sum() < 100:
            continue
        yy, xx = np.nonzero(hot)
        xs.append(xx.mean() * 4.0)
        ys.append(yy.mean() * 4.0)
        ws.append(hot.sum())
    if not xs:
        return mosaic.width / 2.0, mosaic.height / 2.0
    return float(np.average(xs, weights=ws)), float(np.average(ys, weights=ws))


def _runs(flags: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous True spans of a boolean vector, as (start, stop) pairs."""
    if not flags.any():
        return []
    padded = np.r_[False, flags, False]
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return list(zip(edges[0::2].tolist(), edges[1::2].tolist()))


def trace_rim(mosaic: Mosaic, slices: MosaicSlices, centre: tuple[float, float],
              z_planes: list[int], r_max: float, sigma: float = 28.0,
              threshold_k: float = 0.8, n_azimuths: int = 360,
              dr: float = 3.0, min_run_px: float = 24.0,
              progress=None) -> tuple[np.ndarray, np.ndarray]:
    """Radius of the gel/medium interface, per azimuth, per slice.

    Sampling is done tile by tile rather than on a composited mosaic, so the
    texture measured is each camera's own pixels and never a blend of two
    slightly different views of a seam.

    Returns (rim, azimuths); rim is (len(z_planes), n_azimuths) with NaN where
    no interface was found along that ray.
    """
    import cv2

    cx, cy = centre
    az = (np.arange(n_azimuths) + 0.5) * 2.0 * np.pi / n_azimuths
    rr = np.arange(0.0, r_max, dr, dtype=np.float32)
    cos_a, sin_a = np.cos(az)[:, None], np.sin(az)[:, None]
    gx = (cx + rr[None, :] * cos_a).astype(np.float32)
    gy = (cy + rr[None, :] * sin_a).astype(np.float32)

    rim = np.full((len(z_planes), n_azimuths), np.nan, dtype=np.float64)

    for n, z in enumerate(z_planes):
        hot = np.zeros((n_azimuths, rr.size), dtype=bool)
        for tile in mosaic:
            img = slices.tile_slice(tile, z)
            t = np.log1p(cv2.GaussianBlur(tenengrad(img), (0, 0), sigma))
            thr = float(np.median(t)) + threshold_k * float(t.std())
            sx = gx - np.float32(tile.x0)
            sy = gy - np.float32(tile.y0)
            inside = ((sx >= 0) & (sx < tile.width - 1) &
                      (sy >= 0) & (sy < tile.height - 1))
            sampled = cv2.remap(t, sx, sy, cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
            hot |= inside & (sampled > thr)

        min_run = max(1, int(round(min_run_px / dr)))
        for a in range(n_azimuths):
            spans = [s for s in _runs(hot[a]) if s[1] - s[0] >= min_run]
            if spans:
                rim[n, a] = float(rr[spans[-1][1] - 1])
        if progress:
            progress(n + 1, len(z_planes))

    return rim, az


# --------------------------------------------------------------------------- #
# circles, then the cap
# --------------------------------------------------------------------------- #

def _huber_circle(points: np.ndarray, delta: float = 12.0, iters: int = 10):
    """Robust algebraic circle fit; returns (cx, cy, r, median |residual|)."""
    w = np.ones(len(points))
    cx = cy = r = 0.0
    resid = np.zeros(len(points))
    for _ in range(iters):
        a = np.c_[2.0 * points, np.ones(len(points))] * w[:, None]
        b = (points ** 2).sum(1) * w
        sol, *_ = np.linalg.lstsq(a, b, rcond=None)
        cx, cy = float(sol[0]), float(sol[1])
        r = float(np.sqrt(max(sol[2] + cx * cx + cy * cy, 1e-9)))
        resid = np.abs(np.hypot(points[:, 0] - cx, points[:, 1] - cy) - r)
        w = np.where(resid <= delta, 1.0, delta / np.maximum(resid, 1e-9))
    return cx, cy, r, float(np.median(resid))


def fit_rings(rim: np.ndarray, az: np.ndarray, centre: tuple[float, float],
              z_planes: list[int], max_residual_px: float = 30.0,
              min_azimuths: int = 200) -> list[Ring]:
    """One circle per slice, each an independent measurement of the droplet.

    A per-azimuth median-absolute-deviation filter runs first. The trace
    occasionally locks onto something outside the droplet -- the wall of the
    well shows in one corner of this mosaic -- and those rays return a radius
    far from every other ray on the same slice. Removing them costs a handful of
    azimuths out of three hundred and sixty and takes the circle residual from
    tens of pixels down to single figures on the worst slices.
    """
    cx0, cy0 = centre
    rings: list[Ring] = []
    for n, z in enumerate(z_planes):
        radii = rim[n]
        good = np.isfinite(radii)
        if good.sum() < min_azimuths:
            continue
        med = float(np.median(radii[good]))
        mad = 1.4826 * float(np.median(np.abs(radii[good] - med)))
        keep = good & (np.abs(radii - med) <= 4.0 * max(mad, 3.0))
        if keep.sum() < min_azimuths:
            continue
        pts = np.c_[cx0 + radii[keep] * np.cos(az[keep]),
                    cy0 + radii[keep] * np.sin(az[keep])]
        cx, cy, r, resid = _huber_circle(pts)
        rings.append(Ring(z=z, cx=cx, cy=cy, radius_px=r, residual_px=resid,
                          n_azimuths=int(keep.sum()),
                          used=resid <= max_residual_px))
    return rings


def fit_cap(rings: list[Ring], anisotropy: float, substrate: float,
            field_shape: tuple[int, int], n_bootstrap: int = 40,
            seed: int = 0) -> tuple[Dome | None, float]:
    """A spherical cap from the per-slice radii; returns (dome, rms residual).

    For a sphere, r(z)^2 + (z_iso - cz_iso)^2 = R^2, so r^2 + z_iso^2 is *linear*
    in z_iso. That turns the cap into a straight-line fit whose slope is twice
    the centre depth and whose intercept gives the radius -- no iteration, no
    initial guess, and a residual that means something because each point going
    in is one slice's independently fitted circle rather than a raw pixel.
    """
    used = [r for r in rings if r.used]
    if len(used) < 4:
        return None, float("nan")

    z_iso = np.array([r.z for r in used], dtype=float) * anisotropy
    radii = np.array([r.radius_px for r in used], dtype=float)

    def solve(idx):
        a = np.c_[z_iso[idx], np.ones(idx.size)]
        y = radii[idx] ** 2 + z_iso[idx] ** 2
        sol, *_ = np.linalg.lstsq(a, y, rcond=None)
        cz = float(sol[0]) / 2.0
        rad2 = float(sol[1]) + cz * cz
        return cz, (np.sqrt(rad2) if rad2 > 0 else float("nan"))

    all_idx = np.arange(len(used))
    cz_iso, radius = solve(all_idx)
    if not np.isfinite(radius) or radius <= 0:
        return None, float("nan")

    predicted = np.sqrt(np.clip(radius ** 2 - (z_iso - cz_iso) ** 2, 0.0, None))
    rms = float(np.sqrt(np.mean((predicted - radii) ** 2)))

    zg = float(substrate) * anisotropy
    contact = float(np.sqrt(max(radius ** 2 - (zg - cz_iso) ** 2, 0.0)))

    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_bootstrap):
        idx = rng.choice(len(used), len(used), replace=True)
        cz_b, r_b = solve(idx)
        if np.isfinite(r_b) and r_b > 0:
            boots.append(np.sqrt(max(r_b ** 2 - (zg - cz_b) ** 2, 0.0)))
    spread = (100.0 * float(np.std(boots)) / max(float(np.median(boots)), 1e-9)
              if boots else float("nan"))

    cx = float(np.median([r.cx for r in used]))
    cy = float(np.median([r.cy for r in used]))
    apex = (cz_iso - radius) / anisotropy
    h, w = field_shape

    dome = Dome(
        cx_px=cx, cy_px=cy, cz_slice=cz_iso / anisotropy, radius_px=radius,
        anisotropy=anisotropy,
        n_points=sum(r.n_azimuths for r in used),
        residual_px=float(np.median([r.residual_px for r in used])),
        residual_p90_px=float(np.percentile([r.residual_px for r in used], 90)),
        spread_pct=spread,
        substrate_slice=int(round(substrate)),
        contact_radius_px=contact,
        height_slices=float(substrate - apex),
        apex_slice=float(apex),
        covers_field=contact > 0.5 * float(np.hypot(w, h)),
        inlier_frac=len(used) / max(1, len(rings)),
    )
    return dome, rms


# --------------------------------------------------------------------------- #
# the whole thing
# --------------------------------------------------------------------------- #

def fit(mosaic: Mosaic, slices: MosaicSlices, substrate: Substrate,
        anisotropy: float, sigma: float = 28.0, threshold_k: float = 0.8,
        z_step: int = 4,
        substrate_margin_slices: int = 8, n_azimuths: int = 360,
        iterations: int = 3, progress=None) -> GlobalFit:
    """Trace the rim about a provisional axis, refit the axis, repeat.

    The trace has to stop well above the glass. From a few slices up the whole
    field is sharp -- debris on the dish, the gel, everything -- and the
    outermost threshold crossing stops being the droplet and runs away to
    whatever the search limit is. Where that happens is visible as an abrupt
    jump in the median rim radius, and the margin here is set to keep clear of
    it rather than to trust the fit to reject it afterwards.
    """
    z_stop = int(max(4, round(substrate.slice_index) - substrate_margin_slices))
    z_planes = list(range(0, min(z_stop, slices.depth), z_step))
    if len(z_planes) < 4:
        return GlobalFit(None, substrate)

    centre = _seed_axis(mosaic, slices, sigma, threshold_k, z_planes[:6])
    r_max = 0.62 * float(np.hypot(mosaic.width, mosaic.height))

    rings: list[Ring] = []
    rim = np.empty((0, n_azimuths))
    az = np.zeros(n_azimuths)
    for it in range(iterations):
        rim, az = trace_rim(mosaic, slices, centre, z_planes, r_max,
                            sigma=sigma, threshold_k=threshold_k,
                            n_azimuths=n_azimuths, progress=progress)
        rings = fit_rings(rim, az, centre, z_planes)
        used = [r for r in rings if r.used]
        if not used:
            break
        moved = np.hypot(np.median([r.cx for r in used]) - centre[0],
                         np.median([r.cy for r in used]) - centre[1])
        centre = (float(np.median([r.cx for r in used])),
                  float(np.median([r.cy for r in used])))
        if moved < 1.0:
            break

    used = [r for r in rings if r.used]
    scatter = ((float(np.std([r.cx for r in used])),
                float(np.std([r.cy for r in used]))) if used else (0.0, 0.0))

    dome, rms = fit_cap(rings, anisotropy, substrate.slice_index,
                        (mosaic.height, mosaic.width))
    return GlobalFit(dome=dome, substrate=substrate, rings=rings,
                     axis_scatter_px=scatter, cap_residual_px=rms)
