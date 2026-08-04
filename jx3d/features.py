"""The appearance of one organoid, reduced to numbers a classifier can learn from.

The point of this module is not description. It is to produce, for each
organoid, the vector a model will use to decide whether it is alive or dead from
brightfield appearance alone -- no viability dye, no fluorescence. That goal
determines what belongs here and, more importantly, what must be kept out.

**What the biology looks like.** A healthy cystic organoid is a fluid-filled
sphere: a thin refractile epithelial shell around an optically empty lumen, so
in transmitted light its centre is bright and smooth and its rim is a sharp dark
ring. As it dies the lumen fills with sloughed cells and debris, the centre goes
dark and granular, and the shell thickens and loses its crisp edge. The
discriminative quantity is therefore not the organoid's overall brightness or
some global texture number -- it is the *contrast between its core and its rim*.
Every region-based feature here exists to measure that.

**What would ruin it.** Brightfield intensity is illumination times
transmittance. The illumination is not constant: it tilts about sixty grey
levels across a frame, the fifteen tiles differ by ninety in mean level, and the
same organoid seen in two overlapping tiles falls at two different places in the
field. A model fed raw grey values would learn where an organoid was imaged
rather than what it is, and it would do so invisibly. So no raw intensity is
reported. Every intensity feature is an optical density measured against a
background estimated from a ring around that individual organoid, which cancels
the illumination at that position and depth and leaves a quantity that is
additive in absorbers and comparable across the mosaic.

The background is fitted as a plane rather than taken as a mean, because across
the patch of a large organoid the illumination tilts by several percent, and a
scalar background would push exactly that tilt into the texture statistics --
again as a function of position in the tile.

**What is measured and what is modelled.** The outline, the depth of best focus
and everything derived from the equatorial slice are measurements. The axial
extent is not: a 4x / NA 0.20 stack carries no axial shape information, so the
organoid's extent in Z comes from assuming it is a spheroid. Column names that
depend on that assumption are prefixed `model_`, so nobody has to read the
documentation to find out which numbers the microscope did not record.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

FEATURE_VERSION = "2.0.0"

MIN_RADIUS_PX = 6.0
"""Below this the region decomposition has no room: the erosion that separates
core from rim would consume the object. Texture is reported as missing rather
than as a number computed from four pixels."""

OD_RANGE = (-0.30, 2.00)
"""Fixed quantisation window for every texture operator, so a grey-level
co-occurrence matrix means the same thing for a bright organoid and a dark one.
Rescaling each object to its own range would destroy the contrast information
the whole block exists to capture."""

GLCM_LEVELS = 16
GLCM_DISTANCES = (1, 2, 4)
GLCM_ANGLES = (0.0, math.pi / 4, math.pi / 2, 3 * math.pi / 4)
LBP_POINTS = 8
LBP_RADIUS = 2


# --------------------------------------------------------------------------- #
# regions
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class RegionWidths:
    """Widths of the core/annulus/halo decomposition, in pixels.

    Each is a fraction of the organoid's own radius, so the decomposition means
    the same thing for a small organoid and a large one. The absolute floors are
    there because the optics and the biology impose a length scale that does not
    shrink with the object: the epithelial shell is ten to twenty micrometres,
    which is three to five pixels at this calibration, and the segmented
    boundary is not trustworthy to better than about two.
    """

    rim_half: float
    core_guard: float
    halo_inner: float
    halo_outer: float

    @classmethod
    def for_radius(cls, radius_px: float) -> "RegionWidths":
        return cls(rim_half=max(3.0, 0.20 * radius_px),
                   core_guard=max(2.0, 0.05 * radius_px),
                   halo_inner=0.50 * radius_px,
                   halo_outer=1.20 * radius_px)


@dataclass
class Regions:
    """Core, rim annulus and background halo, in patch-local coordinates.

    Patch-local because a twenty-pixel organoid does not need a full frame, and
    the origin is kept so every mask can be drawn back onto the raw slice. That
    is what makes these numbers checkable against the photograph rather than
    merely reproducible.
    """

    body: np.ndarray
    core: np.ndarray
    annulus: np.ndarray
    halo: np.ndarray
    patch: np.ndarray
    origin: tuple[int, int]
    centre: tuple[float, float]
    radius_px: float
    widths: RegionWidths
    halo_lost_to_neighbours: int = 0

    def counts(self) -> dict[str, int]:
        return {"core": int(self.core.sum()), "annulus": int(self.annulus.sum()),
                "halo": int(self.halo.sum()), "body": int(self.body.sum())}


def _disc(shape: tuple[int, int], cx: float, cy: float, r: float) -> np.ndarray:
    yy, xx = np.ogrid[:shape[0], :shape[1]]
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r


def build_regions(image: np.ndarray, x_px: float, y_px: float, radius_px: float,
                  radial_profile_px, neighbours=(), pad_frac: float = 1.4
                  ) -> Regions | None:
    """Cut one organoid out of a slice and derive its three measurement regions.

    Returns None when the organoid sits too close to the edge of the frame for a
    complete background ring. A partial ring is a biased ring -- it samples the
    illumination on one side of the object only -- and that bias would propagate
    into every intensity feature without ever announcing itself.

    Neighbours are removed from the halo for the same reason. A background ring
    containing a second organoid reports a background that is too dark, which
    makes the object under test look more transmissive, which is to say
    healthier, than it is.
    """
    import cv2

    if radius_px < MIN_RADIUS_PX:
        return None

    pad = int(math.ceil((1.0 + pad_frac) * radius_px)) + 4
    y0, x0 = int(round(y_px)) - pad, int(round(x_px)) - pad
    y1, x1 = int(round(y_px)) + pad, int(round(x_px)) + pad
    if y0 < 0 or x0 < 0 or y1 > image.shape[0] or x1 > image.shape[1]:
        return None

    patch = image[y0:y1, x0:x1].astype(np.float32)
    cy, cx = y_px - y0, x_px - x0

    r = np.asarray(radial_profile_px, dtype=np.float64)
    theta = np.linspace(0.0, 2.0 * np.pi, r.size, endpoint=False)
    outline = np.stack([cx + r * np.cos(theta), cy + r * np.sin(theta)],
                       axis=1).astype(np.int32)
    body = np.zeros(patch.shape, dtype=np.uint8)
    cv2.fillPoly(body, [outline], 1)
    if body.sum() < 9:
        return None

    w = RegionWidths.for_radius(radius_px)

    def kernel(radius: float):
        k = max(1, int(round(radius)))
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1,) * 2)

    # The shell lies *inside* the outline, not straddling it. A band centred on
    # the boundary is half background, and since background is the brightest
    # thing in a transmitted-light image that band's optical density is dragged
    # towards zero -- which would make the rim look less dense than the core for
    # every object darker than its surroundings, i.e. for all of them. The
    # epithelium is inside the organoid, so the measurement should be too.
    core = cv2.erode(body, kernel(w.rim_half + w.core_guard)).astype(bool)
    annulus = (body.astype(bool) &
               ~cv2.erode(body, kernel(w.rim_half)).astype(bool))
    halo = (cv2.dilate(body, kernel(w.halo_outer)).astype(bool) &
            ~cv2.dilate(body, kernel(w.halo_inner)).astype(bool))

    before = int(halo.sum())
    for nb_x, nb_y, nb_r in neighbours:
        halo &= ~_disc(halo.shape, nb_x - x0, nb_y - y0, 1.3 * nb_r)

    if core.sum() < 9 or annulus.sum() < 9 or halo.sum() < 60:
        return None

    return Regions(body=body.astype(bool), core=core, annulus=annulus, halo=halo,
                   patch=patch, origin=(y0, x0), centre=(cy, cx),
                   radius_px=radius_px, widths=w,
                   halo_lost_to_neighbours=before - int(halo.sum()))


# --------------------------------------------------------------------------- #
# background and optical density
# --------------------------------------------------------------------------- #

@dataclass
class Background:
    """The local illumination estimate, and enough to audit it."""

    plane: np.ndarray
    level: float
    tilt_frac: float
    """How much the fitted illumination varies across the object, as a fraction
    of its level. A large value on a small organoid means the halo was
    contaminated rather than that the lamp is steeply tilted."""
    clipped_frac: float
    """Fraction of the halo sitting at the top of the 8-bit range. The
    transmitted-light background saturates over much of this dataset, and where
    it does the background level is a lower bound rather than a measurement --
    so every optical density derived from it must be read with this beside it."""
    source: str = "local-halo-plane"


def estimate_background(regions: Regions, dark_dn: float = 0.0,
                        saturation_dn: float = 255.0) -> Background:
    """Fit the illumination over the organoid from the ring around it.

    A plane, fitted robustly. Robustly because a halo will occasionally still
    contain debris or the edge of a neighbour that was not in the catalogue, and
    a couple of dark outliers would drag a plain least-squares background down
    and make the organoid look brighter than it is.
    """
    ys, xs = np.nonzero(regions.halo)
    values = regions.patch[regions.halo].astype(np.float64)
    cy, cx = regions.centre

    design = np.stack([np.ones_like(values), xs - cx, ys - cy], axis=1)
    weights = np.ones_like(values)
    coeffs = np.array([float(np.median(values)), 0.0, 0.0])
    for _ in range(3):
        sol, *_ = np.linalg.lstsq(design * weights[:, None], values * weights,
                                  rcond=None)
        coeffs = sol
        residual = values - design @ coeffs
        scale = 1.4826 * float(np.median(np.abs(residual - np.median(residual)))) + 1e-6
        weights = (np.abs(residual) < 2.5 * scale).astype(np.float64)

    h, w = regions.patch.shape
    yy, xx = np.mgrid[:h, :w]
    plane = coeffs[0] + coeffs[1] * (xx - cx) + coeffs[2] * (yy - cy)

    over_body = plane[regions.body]
    level = float(coeffs[0])
    tilt = float((over_body.max() - over_body.min()) / max(abs(level), 1e-6))
    clipped = float((regions.patch[regions.halo] >= saturation_dn - 0.5).mean())

    return Background(plane=np.maximum(plane, dark_dn + 1.0), level=level,
                      tilt_frac=tilt, clipped_frac=clipped)


def optical_density(regions: Regions, background: Background,
                    dark_dn: float = 0.0) -> np.ndarray:
    """Convert a patch to optical density against its own local background.

    Brightfield forms I = I0 * T + d. Dividing by an estimate of I0 at this
    organoid's position and taking the negative logarithm gives a quantity that
    is additive in absorbing material and free of the illumination field. It is
    the only form in which intensity from two different tiles can honestly be
    compared.
    """
    transmittance = ((regions.patch - dark_dn) /
                     np.maximum(background.plane - dark_dn, 1e-6))
    return -np.log(np.clip(transmittance, 1.0 / 255.0, 4.0))


# --------------------------------------------------------------------------- #
# feature blocks
# --------------------------------------------------------------------------- #

def _stats(values: np.ndarray, prefix: str) -> dict[str, float]:
    """Distribution summary of one region's optical density."""
    if values.size < 4:
        return {}
    from scipy import stats as st

    hist, _ = np.histogram(values, bins=32, range=OD_RANGE)
    p = hist / max(hist.sum(), 1)
    nz = p[p > 0]
    return {
        f"{prefix}_od_mean": float(values.mean()),
        f"{prefix}_od_std": float(values.std()),
        f"{prefix}_od_p10": float(np.percentile(values, 10)),
        f"{prefix}_od_median": float(np.median(values)),
        f"{prefix}_od_p90": float(np.percentile(values, 90)),
        f"{prefix}_od_iqr": float(np.subtract(*np.percentile(values, [75, 25]))),
        f"{prefix}_od_skew": float(st.skew(values)),
        f"{prefix}_od_kurtosis": float(st.kurtosis(values)),
        f"{prefix}_od_entropy": float(-(nz * np.log2(nz)).sum()),
    }


def intensity_features(regions: Regions, od: np.ndarray) -> dict[str, float]:
    """Optical density in the core, the rim, and the contrast between them.

    The last few entries are the point of the block. A live cyst is empty in the
    middle and dark at the edge, so its rim-minus-core contrast is large and
    positive; a necrotic one has filled in, and the contrast collapses towards
    zero. That single difference is the most direct expression of viability this
    modality offers.
    """
    out: dict[str, float] = {}
    out.update(_stats(od[regions.core], "core"))
    out.update(_stats(od[regions.annulus], "rim"))
    out.update(_stats(od[regions.body], "body"))

    core = od[regions.core]
    rim = od[regions.annulus]
    if core.size >= 4 and rim.size >= 4:
        core_mean, rim_mean = float(core.mean()), float(rim.mean())
        out["rim_minus_core_od"] = rim_mean - core_mean
        out["core_over_rim_od"] = core_mean / rim_mean if abs(rim_mean) > 1e-6 else np.nan
        pooled = math.sqrt(0.5 * (core.var() + rim.var())) + 1e-9
        # Separation in units of the within-region scatter: a large value means
        # the core and the rim are genuinely different populations of pixels,
        # not merely different in their averages.
        out["core_rim_separation"] = (rim_mean - core_mean) / pooled
        out["core_fill_fraction"] = float((core > 0.5 * rim_mean).mean())
    return out


def texture_features(regions: Regions, od: np.ndarray) -> dict[str, float]:
    """Granularity of the core, which is what filling with debris produces.

    Co-occurrence descriptors are averaged over the four angles, because an
    organoid has no preferred orientation and an unaveraged descriptor would
    mostly encode how the object happened to be lying. They are quantised over a
    fixed optical-density window rather than each object's own range, so that
    "high contrast" means the same thing from row to row.
    """
    from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

    out: dict[str, float] = {}
    ys, xs = np.nonzero(regions.core)
    if ys.size < 25:
        return out
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    sub = od[y0:y1, x0:x1]
    mask = regions.core[y0:y1, x0:x1]
    if min(sub.shape) < 2 * max(GLCM_DISTANCES) + 1:
        return out

    lo, hi = OD_RANGE
    q = np.clip((sub - lo) / (hi - lo), 0, 1)
    q = (q * (GLCM_LEVELS - 1)).astype(np.uint8)
    # Pixels outside the core are pushed to a level that is then dropped from
    # the matrix, so the descriptor describes the core and not its boundary.
    q[~mask] = 0
    glcm = graycomatrix(q, distances=list(GLCM_DISTANCES),
                        angles=list(GLCM_ANGLES), levels=GLCM_LEVELS,
                        symmetric=True, normed=False)
    glcm[0, :, :, :] = 0
    glcm[:, 0, :, :] = 0
    totals = glcm.sum(axis=(0, 1), keepdims=True)
    if not np.all(totals > 0):
        return out
    glcm = glcm / totals

    for prop in ("contrast", "dissimilarity", "homogeneity", "energy", "correlation"):
        values = graycoprops(glcm, prop)
        for k, d in enumerate(GLCM_DISTANCES):
            out[f"core_glcm_{prop}_d{d}"] = float(np.mean(values[k]))

    lbp = local_binary_pattern(np.clip((sub - lo) / (hi - lo), 0, 1) * 255.0,
                               LBP_POINTS, LBP_RADIUS, method="uniform")
    counts = np.bincount(lbp[mask].astype(int), minlength=LBP_POINTS + 2)
    counts = counts / max(counts.sum(), 1)
    for i, v in enumerate(counts[:LBP_POINTS + 2]):
        out[f"core_lbp_{i:02d}"] = float(v)

    grad_y, grad_x = np.gradient(od)
    magnitude = np.hypot(grad_x, grad_y)
    out["core_grad_mean"] = float(magnitude[regions.core].mean())
    out["core_grad_p90"] = float(np.percentile(magnitude[regions.core], 90))
    out["rim_grad_mean"] = float(magnitude[regions.annulus].mean())
    out["rim_over_core_grad"] = (out["rim_grad_mean"] /
                                 max(out["core_grad_mean"], 1e-9))
    return out


def morphology_features(radial_profile_px, area_px2: float,
                        radius_px: float) -> dict[str, float]:
    """Shape of the measured outline, in scale-free terms wherever possible.

    The radial harmonics are the useful part. An outline resampled as r(theta)
    has a Fourier series whose coefficients are rotation-invariant in magnitude,
    so they describe the shape without describing how it was lying: the second
    harmonic is elongation, the third a trefoil lobing, and the higher ones the
    boundary roughness that distinguishes a smooth cyst from a collapsing one.
    """
    r = np.asarray(radial_profile_px, dtype=np.float64)
    if r.size < 8 or not np.all(np.isfinite(r)) or r.mean() <= 0:
        return {}

    theta = np.linspace(0.0, 2.0 * np.pi, r.size, endpoint=False)
    dtheta = 2.0 * np.pi / r.size
    # Polygon area and perimeter from the sampled outline, which is the shape
    # actually measured; the mask's own pixel count is a rasterisation of it.
    area = float(0.5 * np.sum(r * np.roll(r, -1) * np.sin(dtheta)))
    dx = np.diff(np.r_[r * np.cos(theta), r[0] * np.cos(theta[0])])
    dy = np.diff(np.r_[r * np.sin(theta), r[0] * np.sin(theta[0])])
    perimeter = float(np.sum(np.hypot(dx, dy)))

    out = {
        "outline_area_px2": area,
        "outline_perimeter_px": perimeter,
        "circularity": float(4.0 * np.pi * area / max(perimeter ** 2, 1e-9)),
        "radius_mean_px": float(r.mean()),
        "radius_std_px": float(r.std()),
        "radius_cv": float(r.std() / r.mean()),
        "radius_min_over_max": float(r.min() / max(r.max(), 1e-9)),
    }

    spectrum = np.abs(np.fft.rfft(r)) / r.size
    dc = max(spectrum[0], 1e-9)
    for k in range(1, 7):
        if k < spectrum.size:
            out[f"harmonic_{k}"] = float(spectrum[k] / dc)
    if spectrum.size > 7:
        out["harmonic_high"] = float(spectrum[7:].sum() / dc)

    from scipy.spatial import ConvexHull

    points = np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1)
    try:
        hull = ConvexHull(points)
        out["solidity"] = float(area / max(hull.volume, 1e-9))
        out["convexity"] = float(hull.area / max(perimeter, 1e-9))
        # An organoid is convex at this scale, so a deep notch in its outline is
        # a segmentation failure and not a shape. The commonest one here is a
        # wedge bitten out of an otherwise round object, which leaves solidity
        # well below one while circularity still looks respectable. It is
        # flagged rather than repaired: filling the notch with the convex hull
        # would draw a boundary the microscope never showed, and every feature
        # measured inside it would then be measured partly on invented pixels.
        out["shape_suspect"] = float(out["solidity"] < 0.93 or
                                     out["radius_cv"] > 0.22)
    except Exception:
        # A degenerate outline has no hull. Reporting nothing is right; a
        # solidity of 1.0 would read as a perfectly convex object.
        pass
    return out


def spatial_features(x_px: float, y_px: float, z_slice: float, radius_px: float,
                     dome, substrate_slice: float, anisotropy: float
                     ) -> dict[str, float]:
    """Where the organoid sits in the droplet, and how close it is to leaving it.

    The clearance is measured from the organoid's *surface*, not from its
    centre, so zero means touching the gel/medium boundary. It is the quantity
    the nearest-border line in the viewer draws, and the closest point is
    returned with it so that line has somewhere to end.
    """
    out = {
        "x_mosaic_px": float(x_px),
        "y_mosaic_px": float(y_px),
        "height_above_glass_slices": float(substrate_slice - z_slice),
    }
    if dome is None:
        return out

    dx, dy = x_px - dome.cx_px, y_px - dome.cy_px
    radial = float(np.hypot(dx, dy))
    out["dome_radial_px"] = radial
    out["dome_azimuth_deg"] = float(np.degrees(np.arctan2(dy, dx)) % 360.0)
    out["dome_radial_frac"] = radial / max(dome.contact_radius_px, 1e-9)

    centre_gap = float(dome.distance_px(x_px, y_px, z_slice))
    out["dome_distance_px"] = centre_gap - radius_px

    # The nearest boundary is not always the curved cap: for an organoid sitting
    # low and near the rim, the circle where the gel meets the glass is closer.
    # Reporting only the cap would overstate how sheltered it is.
    lateral_to_contact = dome.contact_radius_px - radial
    depth_to_glass = (substrate_slice - z_slice) * anisotropy
    out["contact_edge_distance_px"] = float(
        np.hypot(max(lateral_to_contact, 0.0), depth_to_glass) - radius_px
        if lateral_to_contact < 0 else lateral_to_contact - radius_px)
    out["nearest_border_is_cap"] = float(
        out["dome_distance_px"] <= out["contact_edge_distance_px"])
    out["nearest_border_px"] = float(min(out["dome_distance_px"],
                                         out["contact_edge_distance_px"]))
    return out


def neighbourhood_features(index: int, points: np.ndarray, radii: np.ndarray,
                           k: int = 5) -> dict[str, float]:
    """How crowded this organoid's surroundings are, in isotropic pixels.

    Crowding is a plausible covariate of viability -- organoids competing for
    nutrients in a dense patch do worse -- and it is also a confounder worth
    being able to control for, which requires measuring it.
    """
    if points.shape[0] < 2:
        return {}
    delta = points - points[index]
    distance = np.sqrt((delta ** 2).sum(axis=1))
    distance[index] = np.inf
    order = np.argsort(distance)
    nearest = distance[order[0]]
    out = {
        "nn_distance_px": float(nearest),
        "nn_gap_px": float(nearest - radii[index] - radii[order[0]]),
        "n_within_5r": float((distance < 5.0 * max(radii[index], 1e-6)).sum()),
    }
    take = order[:min(k, order.size)]
    finite = distance[take][np.isfinite(distance[take])]
    if finite.size:
        out[f"mean_distance_{finite.size}nn_px"] = float(finite.mean())
    return out


# --------------------------------------------------------------------------- #
# assembling one row
# --------------------------------------------------------------------------- #

@dataclass
class FeatureRow:
    values: dict[str, float] = field(default_factory=dict)
    missing: list[str] = field(default_factory=list)
    """Blocks that could not be computed, and are absent rather than guessed."""

    def update(self, block: dict[str, float], name: str) -> None:
        if block:
            self.values.update(block)
        else:
            self.missing.append(name)


def measure(image: np.ndarray, x_px: float, y_px: float, radius_px: float,
            radial_profile_px, area_px2: float, neighbours=(),
            dark_dn: float = 0.0) -> FeatureRow:
    """Every appearance feature for one organoid on its equatorial slice."""
    row = FeatureRow()
    row.update(morphology_features(radial_profile_px, area_px2, radius_px),
               "morphology")

    regions = build_regions(image, x_px, y_px, radius_px, radial_profile_px,
                            neighbours=neighbours)
    if regions is None:
        row.missing.extend(["intensity", "texture"])
        row.values["appearance_measurable"] = 0.0
        return row

    background = estimate_background(regions, dark_dn=dark_dn)
    od = optical_density(regions, background, dark_dn=dark_dn)

    row.update(intensity_features(regions, od), "intensity")
    row.update(texture_features(regions, od), "texture")
    row.values.update({
        "appearance_measurable": 1.0,
        "background_dn": background.level,
        "background_tilt_frac": background.tilt_frac,
        "background_clipped_frac": background.clipped_frac,
        "halo_px": float(regions.halo.sum()),
        "halo_lost_to_neighbours_px": float(regions.halo_lost_to_neighbours),
    })
    return row


# --------------------------------------------------------------------------- #
# what a person actually reads
# --------------------------------------------------------------------------- #

VIABILITY_COLUMNS: list[tuple[str, str]] = [
    # --- which organoid, and can this row be trusted at all ---
    ("uid", "identity in this dataset"),
    ("tile", "which field it was measured in"),
    ("x_mosaic_px", "position in the assembled dome"),
    ("y_mosaic_px", "position in the assembled dome"),
    ("z_slice", "depth of its focal plane, in slices"),
    ("appearance_measurable", "0 when the object was too small or too close to a "
                              "frame edge for the core/rim decomposition; every "
                              "appearance column below is missing on those rows"),
    ("background_clipped_frac", "fraction of the local background sitting at 255. "
                                "Where this is high the background is a lower "
                                "bound rather than a measurement, and every "
                                "optical density derived from it is soft"),
    ("focus_sharpness", "prominence of the rim's focus peak, 0-1. Low values are "
                        "objects that never came properly into focus"),
    ("clipped", "the outline touches its field's edge, so shape and texture "
                "describe the cut and not the organoid"),

    # --- the viability construct itself ---
    ("rim_minus_core_od", "optical density of the rim minus the core. A live "
                          "cystic organoid is a fluid-filled sphere with a thin "
                          "refractile shell, so its centre is empty and its rim "
                          "is dense and this is large and positive. As it dies "
                          "the lumen fills with debris, the centre darkens, and "
                          "this collapses towards zero. The single most direct "
                          "expression of viability this modality offers"),
    ("core_rim_separation", "the same difference divided by the scatter within "
                            "the two regions, so it says whether core and rim "
                            "are genuinely different populations of pixels or "
                            "merely different on average"),
    ("core_fill_fraction", "fraction of the core as dense as the rim: how full "
                           "the lumen has become"),
    ("core_od_mean", "how absorbing the interior is"),
    ("core_od_std", "how uneven it is. A smooth lumen is uniform; a necrotic one "
                    "is not"),
    ("core_od_entropy", "granularity of the interior, as information content"),
    ("core_glcm_contrast_d2", "local texture contrast inside the core. Debris "
                              "gives a grainy interior and raises this"),
    ("core_glcm_homogeneity_d2", "the counterpart: high for a smooth lumen"),
    ("rim_od_mean", "how dense the shell is"),
    ("rim_over_core_grad", "how much sharper the boundary is than the interior. "
                           "A live organoid has a crisp refractile edge around a "
                           "featureless middle; a dying one loses the contrast"),

    # --- shape, which changes as an organoid degenerates ---
    ("diameter_um", "equatorial diameter"),
    ("circularity", "4*pi*A/P^2. Healthy cysts are round; collapsing ones are not"),
    ("solidity", "outline area over its convex hull: how lobed or dented it is"),
    ("radius_cv", "variation of the radius around the outline, a scale-free "
                  "measure of how irregular the boundary is"),
    ("shape_suspect", "the outline has a notch or a raggedness an organoid does "
                      "not have, so the segmenter probably cut into it or merged "
                      "it with a neighbour. Affects a small minority -- two of a "
                      "hundred and twenty-six on the field this was measured on "
                      "-- and those rows should be dropped before any shape or "
                      "texture comparison"),

    # --- where it sits, which is a real biological variable ---
    ("nearest_border_px", "distance from the organoid's surface to the nearest "
                          "gel boundary. Nutrient access, oxygen and mechanical "
                          "confinement all depend on it"),
    ("dome_radial_frac", "how far out in the droplet it sits, 0 at the axis and "
                         "1 at the rim"),
    ("height_above_glass_slices", "how far above the well bottom"),
    ("nn_gap_px", "clear distance to the nearest neighbouring organoid"),
]
"""The columns a person is meant to read.

The full matrix runs to a hundred and thirty-nine columns and most of them are
there because they were cheap to compute, not because anyone would look at them:
five co-occurrence descriptors at three distances each, ten local-binary-pattern
bins, percentiles of three regions. That is a fine input to a model and a poor
thing to hand a biologist who wants to know which organoids are dying.

So this is the default, and nothing is thrown away -- the rest is still computed
and still written when it is asked for, because a classifier may well find
signal in a texture bin no human would have picked. What changes is which one
you get without asking.

Every entry carries the reason it earned its place. A column that cannot be
justified in a sentence does not belong in a matrix meant for reading.
"""

VIABILITY_COLUMN_NAMES = [name for name, _ in VIABILITY_COLUMNS]
