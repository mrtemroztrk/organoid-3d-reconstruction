"""The Matrigel dome surface, measured rather than assumed.

Organoids are grown inside a droplet of Matrigel sitting on the well bottom.
That droplet has an outer surface, and how far an organoid sits from it is a
real biological variable -- nutrient access, oxygen, and mechanical confinement
all depend on it.

The surface is directly visible in the data. Where the curved gel/medium
interface crosses the focal plane it refracts light strongly and leaves a bright,
high-texture ridge. Because the surface is curved, that ridge sits at a
different place on every slice: it moves outward as the focus descends towards
the glass, tracing the widening cross-section of the droplet.

So the ridge, collected over all slices, is a point cloud lying *on* the dome
surface. Fitting a sphere to it recovers the droplet geometry. Nothing here is
assumed except that a sessile droplet is a spherical cap, which is what surface
tension makes it.

Units: everything here works in pixels and slice indices. The caller converts to
micrometres only if the stack is actually calibrated.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import cv2
import numpy as np

from .focus import tenengrad


@dataclass
class Dome:
    """A fitted spherical cap, in pixel/slice coordinates.

    `cz_slice` and `radius_*` share a common isotropic frame: lateral distances
    are in pixels and axial distances have been divided by `anisotropy`
    (slice spacing / pixel spacing) so that a sphere really is a sphere.
    """

    cx_px: float
    cy_px: float
    cz_slice: float
    radius_px: float          # sphere radius, in lateral pixel units
    anisotropy: float         # slice spacing / pixel spacing used for the fit

    n_points: int
    residual_px: float        # median |distance to surface|
    residual_p90_px: float
    spread_pct: float         # bootstrap variability of the contact radius

    substrate_slice: int
    contact_radius_px: float  # where the surface meets the glass
    height_slices: float      # apex to glass
    apex_slice: float
    covers_field: bool        # is the contact circle wider than the frame?
    inlier_frac: float = 1.0
    encloses_frac: float | None = None
    """Fraction of detected organoids that fall inside the fitted droplet.
    Organoids grow in the gel, so a good fit encloses essentially all of them;
    a low value means the surface is wrong, not that the biology is strange."""
    reliable: bool = True

    def distance_px(self, x_px, y_px, z_slice) -> np.ndarray:
        """Signed distance from a point to the surface, in lateral pixels.

        Positive = inside the droplet, negative = outside it.
        """
        dx = np.asarray(x_px, dtype=float) - self.cx_px
        dy = np.asarray(y_px, dtype=float) - self.cy_px
        dz = (np.asarray(z_slice, dtype=float) - self.cz_slice) * self.anisotropy
        return self.radius_px - np.sqrt(dx * dx + dy * dy + dz * dz)

    def surface_z_slice(self, x_px, y_px):
        """Slice index of the dome surface above (x, y); NaN outside the cap."""
        dx = np.asarray(x_px, dtype=float) - self.cx_px
        dy = np.asarray(y_px, dtype=float) - self.cy_px
        rad2 = self.radius_px ** 2 - (dx * dx + dy * dy)
        out = np.full(np.shape(dx), np.nan, dtype=float)
        ok = rad2 > 0
        out[ok] = self.cz_slice - np.sqrt(rad2[ok]) / self.anisotropy
        return out

    def to_dict(self) -> dict:
        d = asdict(self)
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items()}


# --------------------------------------------------------------------------- #
# ridge detection
# --------------------------------------------------------------------------- #

def ridge_points(volume: np.ndarray, z_range: range, sigma: float = 28.0,
                 x_min_frac: float = 0.0, threshold_k: float = 0.8,
                 row_step: int = 6, edge: str = "outer",
                 progress=None) -> np.ndarray:
    """Points on the gel/medium interface, as (x_px, y_px, z_slice).

    Where the curved interface crosses the focal plane it leaves a broad band of
    high texture, not a thin line: near the edge you look through a long slanted
    path of gel. The gel *ends* at the far side of that band, so the interface is
    the band's **outer edge**, not its brightest point. Tracking the bright point
    instead puts the surface inside the droplet and leaves about a quarter of the
    organoids apparently outside the gel they grew in.

    "Outer" has no fixed direction, though. The droplet edge can enter the field
    from any side, and scanning every row left-to-right -- as this first did --
    silently assumes it lies to the right. On a field where it does not, the scan
    returns debris instead and the fit is rejected downstream. So each line is
    scanned from both ends, in rows and in columns, and every first-crossing is
    offered to the consensus fit, which keeps whichever set actually lies on a
    sphere.

    `sigma` is deliberately large. At organoid scale the texture map is full of
    organoids; only at a much coarser scale does the interface band dominate.
    """
    h, w = volume.shape[1:]
    x0 = int(w * x_min_frac)
    pts: list[tuple[float, float, float]] = []

    for n, z in enumerate(z_range):
        t = np.log1p(cv2.GaussianBlur(tenengrad(volume[z].astype(np.float32)),
                                      (0, 0), sigma))
        thr = float(np.median(t)) + threshold_k * float(t.std())
        hot = t > thr

        if edge == "peak":
            for y in range(row_step, h - row_step, row_step):
                x = int(np.argmax(t[y, x0:])) + x0
                if x0 + 4 < x < w - 6:
                    pts.append((float(x), float(y), float(z)))
        else:
            # rows, scanned inward from each side
            for y in range(row_step, h - row_step, row_step):
                idx = np.flatnonzero(hot[y, x0:])
                if idx.size == 0:
                    continue
                for x in (int(idx[-1]) + x0, int(idx[0]) + x0):
                    if x0 + 4 < x < w - 6:
                        pts.append((float(x), float(y), float(z)))
            # columns, likewise -- catches an edge running across the frame
            for x in range(row_step, w - row_step, row_step):
                idx = np.flatnonzero(hot[:, x])
                if idx.size == 0:
                    continue
                for y in (int(idx[-1]), int(idx[0])):
                    if 4 < y < h - 6:
                        pts.append((float(x), float(y), float(z)))

        if progress:
            progress(n + 1, len(z_range))

    return np.asarray(pts, dtype=float).reshape(-1, 3)


# --------------------------------------------------------------------------- #
# sphere fitting
# --------------------------------------------------------------------------- #

def _fit_sphere(q: np.ndarray, weights: np.ndarray | None = None):
    """Algebraic least-squares sphere fit; returns (centre, radius)."""
    a = np.c_[2 * q, np.ones(len(q))]
    b = (q ** 2).sum(1)
    if weights is not None:
        a = a * weights[:, None]
        b = b * weights
    c, *_ = np.linalg.lstsq(a, b, rcond=None)
    centre = c[:3]
    radius = np.sqrt(max(c[3] + (centre ** 2).sum(), 1e-9))
    return centre, radius


def _irls(q: np.ndarray, iters: int = 12, delta: float = 12.0):
    """Huber-weighted refit.

    Hard trimming was tried first and drifts: each round throws away the points
    that disagree, which lets the radius wander off to whatever the surviving
    subset likes. Down-weighting instead keeps every point in the problem.
    `delta` is in lateral pixels.
    """
    w = np.ones(len(q))
    centre, radius, resid = np.zeros(3), 1.0, np.array([0.0])
    for _ in range(iters):
        centre, radius = _fit_sphere(q, w)
        resid = np.abs(np.linalg.norm(q - centre, axis=1) - radius)
        w = np.where(resid <= delta, 1.0, delta / np.maximum(resid, 1e-9))
    return centre, radius, resid


def _ransac(q: np.ndarray, tol: float, iters: int, rng) -> np.ndarray | None:
    """Inlier mask for the dominant spherical surface in the point cloud.

    The ridge trace is not pure: inside the droplet it occasionally locks onto a
    cluster of organoids instead of the interface. Those points are a coherent
    minority, and plain reweighting can be dragged by them, so the surface is
    first found by consensus and only then refined.
    """
    best = None
    best_n = 0
    n = len(q)
    if n < 8:
        return None
    for _ in range(iters):
        idx = rng.choice(n, 4, replace=False)
        try:
            c, r = _fit_sphere(q[idx])
        except np.linalg.LinAlgError:
            continue
        if not np.isfinite(r) or r <= 0:
            continue
        inl = np.abs(np.linalg.norm(q - c, axis=1) - r) < tol
        k = int(inl.sum())
        if k > best_n:
            best_n, best = k, inl
    return best


def fit(points: np.ndarray, anisotropy: float, substrate_slice: int,
        field_shape: tuple[int, int], n_bootstrap: int = 20,
        seed: int = 0, ransac_iters: int = 400,
        inlier_tol_px: float = 25.0) -> Dome | None:
    """Fit the dome to a ridge point cloud.

    `anisotropy` (slice spacing / pixel spacing) makes the frame isotropic so a
    sphere fit is meaningful. The bootstrap spread is reported because the
    droplet is usually wider than the field of view, so only an arc of it is
    ever seen and the radius is the least well constrained parameter.
    """
    if len(points) < 100:
        return None

    q = points.copy()
    q[:, 2] *= anisotropy                       # into isotropic pixel units

    rng0 = np.random.default_rng(seed)
    inliers = _ransac(q, inlier_tol_px, ransac_iters, rng0)
    if inliers is not None and inliers.sum() >= max(100, 0.25 * len(q)):
        q = q[inliers]

    centre, radius, resid = _irls(q)
    if not np.isfinite(radius) or radius <= 0:
        return None

    h, w = field_shape
    zg = float(substrate_slice)

    def contact(c, r):
        dz = (zg - c[2] / anisotropy) * anisotropy
        return float(np.sqrt(max(r * r - dz * dz, 0.0)))

    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_bootstrap):
        idx = rng.choice(len(q), max(60, int(len(q) * 0.6)), replace=False)
        cb, rb, _ = _irls(q[idx])
        if np.isfinite(rb) and rb > 0:
            boots.append(contact(cb, rb))
    spread = (100.0 * float(np.std(boots)) / max(float(np.median(boots)), 1e-9)
              if boots else float("nan"))

    cz_slice = float(centre[2]) / anisotropy
    contact_r = contact(centre, radius)
    apex = cz_slice - radius / anisotropy

    return Dome(
        cx_px=float(centre[0]),
        cy_px=float(centre[1]),
        cz_slice=cz_slice,
        radius_px=float(radius),
        anisotropy=float(anisotropy),
        n_points=len(q),
        residual_px=float(np.median(resid)),
        residual_p90_px=float(np.percentile(resid, 90)),
        spread_pct=spread,
        substrate_slice=int(substrate_slice),
        contact_radius_px=contact_r,
        height_slices=float(zg - apex),
        apex_slice=float(apex),
        covers_field=contact_r > 0.5 * float(np.hypot(w, h)),
        inlier_frac=float(len(q)) / float(len(points)),
    )


def detect(volume: np.ndarray, z_range: range, anisotropy: float,
           substrate_slice: int, sigma: float = 28.0,
           progress=None) -> tuple[Dome | None, np.ndarray]:
    """Convenience: find the ridge and fit the dome. Returns (dome, points)."""
    pts = ridge_points(volume, z_range, sigma=sigma, progress=progress)
    d = fit(pts, anisotropy, substrate_slice, volume.shape[1:])
    return d, pts


# --------------------------------------------------------------------------- #
# meshing
# --------------------------------------------------------------------------- #

def mesh(d: Dome, field_shape: tuple[int, int], n_u: int = 96, n_v: int = 48,
         margin_px: float = 0.0):
    """Triangulated cap surface, clipped to the imaged field and the glass.

    The droplet is normally wider than the field of view, so most of the cap is
    never imaged. Vertices outside the frame are pulled to the frame edge rather
    than drawn floating in space -- what is shown is the part the microscope
    actually saw.

    Returns (vertices in (x_px, y_px, z_slice), faces).
    """
    h, w = field_shape
    lo, hi = -margin_px, None

    # parameterise the cap by polar angle from the apex
    max_theta = np.arccos(np.clip(
        ((d.cz_slice - d.substrate_slice) * d.anisotropy) / d.radius_px, -1.0, 1.0))
    theta = np.linspace(0.0, max_theta, n_v)
    phi = np.linspace(0.0, 2.0 * np.pi, n_u, endpoint=False)

    st, ct = np.sin(theta)[:, None], np.cos(theta)[:, None]
    x = d.cx_px + d.radius_px * st * np.cos(phi)[None, :]
    y = d.cy_px + d.radius_px * st * np.sin(phi)[None, :]
    z = d.cz_slice - (d.radius_px * ct * np.ones((1, n_u))) / d.anisotropy

    verts = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)

    faces = []
    for i in range(n_v - 1):
        for j in range(n_u):
            j2 = (j + 1) % n_u
            a = i * n_u + j
            b = i * n_u + j2
            c = (i + 1) * n_u + j
            e = (i + 1) * n_u + j2
            faces.append([a, c, e])
            faces.append([a, e, b])
    faces = np.asarray(faces, dtype=np.int64)

    # keep only faces with at least one vertex inside the imaged field
    inside = ((verts[:, 0] >= lo) & (verts[:, 0] <= w + margin_px) &
              (verts[:, 1] >= lo) & (verts[:, 1] <= h + margin_px))
    keep = inside[faces].any(axis=1)
    faces = faces[keep]

    used, faces = np.unique(faces, return_inverse=True)
    return verts[used], faces.reshape(-1, 3)


def validate(d: Dome | None, organoids, min_enclosed: float = 0.90) -> Dome | None:
    """Check the fit against a physical constraint: organoids live in the gel.

    If the fitted surface leaves a meaningful fraction of them outside, the fit
    is wrong. Returning it flagged rather than silently is the point -- a
    plausible-looking dome that excludes a third of the sample would quietly
    corrupt every clearance measurement downstream.
    """
    if d is None or not organoids:
        return d
    gaps = d.distance_px([o.x_px for o in organoids],
                         [o.y_px for o in organoids],
                         [o.z_slice for o in organoids])
    radii = np.array([o.radius_px for o in organoids])
    d.encloses_frac = float(np.mean(gaps - radii > -0.25 * radii))
    d.reliable = d.encloses_frac >= min_enclosed
    return d
