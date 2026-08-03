"""Turn linked tracks into measured 3D organoids.

The reconstruction is deliberately *model-based* rather than voxel-based, and it
is worth being explicit about why.

A 4x / NA 0.20 brightfield stack carries no true axial shape information: the
rim of an organoid looks the same whether you are 30 um above or below its
equator. What the stack does give, reliably, is

  * where the organoid is laterally,          (x, y)
  * at what depth its rim is sharpest,        (z, by shape-from-focus)
  * and its full in-plane outline at that depth.

The plane of sharpest rim contrast is the organoid's equator, so its outline
there is the object's true widest cross-section. From those measurements we
build a solid of revolution: the measured outline r(theta) swept over a
spheroidal profile. Non-circular organoids stay non-circular; nothing is
invented in Z beyond the near-spherical assumption, which is stated in
`Params.axial_ratio` and can be changed.

Anything more elaborate would be fabricating detail the microscope did not
record.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import Acquisition, Params
from .focus import contour_band, find_focal_planes, focus_curve
from .link import Track


@dataclass
class Organoid:
    """One measured organoid.

    Pixel/slice fields are the measurement. The micrometre fields are a
    convenience conversion and are ``None`` whenever the stack is uncalibrated
    -- they are never filled in from a guessed scale.
    """

    oid: int

    # --- position, image coordinates (the measurement) ---
    x_px: float
    y_px: float
    z_slice: float          # fractional, sub-slice interpolated
    best_slice: int         # integer slice the outline was measured on

    # --- size, lateral pixels and slices (the measurement) ---
    radius_px: float        # equivalent-circle radius of the equatorial outline
    diameter_px: float
    radius_z_slices: float  # semi-axis along Z, in slices
    area_px2: float         # equatorial cross-section
    volume_voxels: float    # ellipsoid volume, in px*px*slice units

    # --- shape / quality ---
    circularity: float
    focus_sharpness: float  # peak prominence of the focus curve, 0..1
    n_slices: int
    z_extent_slices: tuple[int, int]
    source: str             # 'edf' or 'slices'

    # --- geometry: equatorial outline sampled at n_theta angles, in pixels ---
    radial_profile_px: list[float]

    # --- distance to the Matrigel dome surface (None if no dome was fitted) ---
    dome_distance_px: float | None = None
    """Shortest distance from the organoid surface to the gel/medium interface,
    in lateral pixels. Positive = inside the droplet."""
    dome_surface_slice: float | None = None
    """Slice index where the dome surface sits directly above this organoid."""

    # --- calibrated conversions; None when the stack has no known scale ---
    x_um: float | None = None
    y_um: float | None = None
    z_um: float | None = None
    radius_um: float | None = None
    diameter_um: float | None = None
    radius_z_um: float | None = None
    area_um2: float | None = None
    volume_um3: float | None = None
    dome_distance_um: float | None = None
    radial_profile_um: list[float] | None = None

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        d["z_extent_slices"] = list(self.z_extent_slices)
        d["radial_profile_px"] = [round(float(r), 3) for r in self.radial_profile_px]
        if self.radial_profile_um is not None:
            d["radial_profile_um"] = [round(float(r), 3) for r in self.radial_profile_um]
        for k, v in d.items():
            if isinstance(v, float):
                d[k] = round(v, 4)
        return d


def _radial_profile(contour: np.ndarray, cx: float, cy: float,
                    n_theta: int) -> np.ndarray:
    """Resample a closed polygon as r(theta) on a uniform angular grid."""
    dx = contour[:, 0] - cx
    dy = contour[:, 1] - cy
    ang = np.mod(np.arctan2(dy, dx), 2 * np.pi)
    rad = np.hypot(dx, dy)

    order = np.argsort(ang)
    ang, rad = ang[order], rad[order]

    # wrap one sample on each side so interpolation is continuous across 0/2pi
    ang_w = np.concatenate([ang[-1:] - 2 * np.pi, ang, ang[:1] + 2 * np.pi])
    rad_w = np.concatenate([rad[-1:], rad, rad[:1]])

    grid = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    return np.interp(grid, ang_w, rad_w)


def _focus_window(z_center: int, depth: int, half: int) -> range:
    return range(max(0, z_center - half), min(depth, z_center + half + 1))


def measure_tracks(stack, tracks: list[Track], params: Params,
                   z_max: int | None = None, progress=None) -> list[Organoid]:
    """Locate each track's focal plane(s) and measure the organoid there.

    `z_max` is exclusive and must exclude the substrate: the dish surface is
    sharper than any organoid, so a focus sweep allowed to reach it has its peak
    dragged down onto the glass and the organoid is reported at the wrong depth.
    """
    acq: Acquisition = stack.acq
    vol = stack.data
    z_limit = stack.depth if z_max is None else max(3, min(stack.depth, z_max))
    out: list[Organoid] = []

    dof_slices = acq.depth_of_field_slices
    # Window wide enough to contain the focus peak, narrow enough to stay cheap.
    half_window = max(6, int(round(3.0 * dof_slices)))
    # Two focal planes closer than the depth of field are not separable.
    min_sep = max(2.0, dof_slices)

    for n, t in enumerate(tracks):
        # --- 1. coarse guess: which of the track's own slices is sharpest? ---
        best_det, best_val = None, -1.0
        for d in t.dets:
            if d.z >= z_limit:
                continue
            band = _band_from_contour(d, vol.shape[1:], params.focus_band_px)
            if band is None:
                continue
            val = focus_curve(vol, band, range(d.z, d.z + 1))[0]
            if val > best_val:
                best_val, best_det = val, d
        if best_det is None:
            continue

        # --- 2. sweep that outline through Z to find the focal plane(s) ---
        band = _band_from_contour(best_det, vol.shape[1:], params.focus_band_px)
        zwin = _focus_window(best_det.z, z_limit, half_window)
        curve = focus_curve(vol, band, zwin)
        planes = find_focal_planes(curve, min_sep)
        if not planes:
            continue

        for peak_idx, prominence in planes:
            z_focus = zwin.start + peak_idx
            z_int = int(round(z_focus))
            if z_int >= z_limit:
                continue

            # --- 3. measure the outline on the slice closest to that plane ---
            det = t.det_at(z_int) or _nearest_det(t, z_int, z_limit) or best_det
            out.append(_organoid_at(det, z_focus, prominence, acq, params,
                                    n_slices=t.n_slices,
                                    z_extent=(t.z_first, t.z_last),
                                    source="slices"))
        if progress:
            progress(n + 1, len(tracks))

    return _drop_duplicates(out, acq)


def measure_regions(stack, detections, params: Params, z_max: int | None = None,
                    z_hint: np.ndarray | None = None,
                    progress=None) -> list[Organoid]:
    """Measure objects segmented on the all-in-focus image.

    There is no track here -- one outline per organoid, taken from the EDF, where
    the rim is by construction as sharp as that organoid ever gets. Depth comes
    from sweeping that outline's rim band through the whole stack.

    `z_hint` is the EDF depth map; its median inside the rim band gives a
    starting depth, which is only used to report how far the refined focus peak
    moved (a large disagreement means the outline straddles two objects).
    """
    acq: Acquisition = stack.acq
    vol = stack.data
    z_limit = stack.depth if z_max is None else max(3, min(stack.depth, z_max))
    dof_slices = acq.depth_of_field_slices
    min_sep = max(2.0, dof_slices)

    out: list[Organoid] = []
    zwin = range(0, z_limit)

    for n, det in enumerate(detections):
        band = _band_from_contour(det, vol.shape[1:], params.focus_band_px)
        if band is None:
            continue
        curve = focus_curve(vol, band, zwin)
        planes = find_focal_planes(curve, min_sep)
        # slices over which the rim stays at least half as sharp as at its peak
        in_focus = int((curve >= 0.5 * curve.max()).sum()) if curve.size else 0
        for peak_idx, prominence in planes:
            out.append(_organoid_at(det, peak_idx, prominence, acq, params,
                                    n_slices=in_focus,
                                    z_extent=(zwin.start, zwin.stop - 1),
                                    source="edf"))
        if progress:
            progress(n + 1, len(detections))

    return out


def _organoid_at(det, z_focus: float, prominence: float, acq: Acquisition,
                 params: Params, n_slices: int, z_extent,
                 source: str = "edf") -> Organoid:
    """Assemble one measurement, in pixels and slices."""
    r_prof_px = _radial_profile(det.contour, det.cx, det.cy, params.n_theta)
    r_eq_px = float(np.sqrt(det.area_px / np.pi))

    # The axial semi-axis is a lateral length scaled by `axial_ratio`, expressed
    # back in slices via the anisotropy so that all outputs stay in image units.
    rz_iso = r_eq_px * params.axial_ratio
    rz_slices = rz_iso / acq.anisotropy

    o = Organoid(
        oid=0,
        x_px=float(det.cx),
        y_px=float(det.cy),
        z_slice=float(z_focus),
        best_slice=int(round(z_focus)),
        radius_px=r_eq_px,
        diameter_px=2.0 * r_eq_px,
        radius_z_slices=float(rz_slices),
        area_px2=float(det.area_px),
        volume_voxels=float((4.0 / 3.0) * np.pi * r_eq_px * r_eq_px * rz_slices),
        circularity=float(det.circularity),
        focus_sharpness=float(prominence),
        n_slices=int(n_slices),
        z_extent_slices=tuple(z_extent),
        source=source,
        radial_profile_px=[float(v) for v in r_prof_px],
    )
    add_calibrated(o, acq)
    return o


def add_calibrated(o: Organoid, acq: Acquisition) -> None:
    """Fill the micrometre fields -- only if the stack actually has a scale."""
    if not acq.calibrated:
        return
    p, zu = acq.px_um, acq.z_um
    o.x_um = o.x_px * p
    o.y_um = o.y_px * p
    o.z_um = o.z_slice * zu
    o.radius_um = o.radius_px * p
    o.diameter_um = o.diameter_px * p
    o.radius_z_um = o.radius_z_slices * zu
    o.area_um2 = o.area_px2 * p * p
    o.volume_um3 = (4.0 / 3.0) * np.pi * o.radius_um ** 2 * o.radius_z_um
    o.radial_profile_um = [r * p for r in o.radial_profile_px]
    if o.dome_distance_px is not None:
        o.dome_distance_um = o.dome_distance_px * p


def attach_dome(organoids: list[Organoid], dome, acq: Acquisition) -> None:
    """Record each organoid's clearance to the gel/medium interface.

    Measured from the organoid's *surface*, not its centre, so a value of zero
    means the organoid is touching the outside of the droplet. Positive is
    inside the gel.
    """
    if dome is None:
        return
    for o in organoids:
        centre_gap = float(dome.distance_px(o.x_px, o.y_px, o.z_slice))
        o.dome_distance_px = centre_gap - o.radius_px
        o.dome_surface_slice = float(dome.surface_z_slice(
            np.array([o.x_px]), np.array([o.y_px]))[0])
        if acq.calibrated:
            o.dome_distance_um = o.dome_distance_px * acq.px_um


def merge_sources(*groups: list[Organoid], acq: Acquisition) -> list[Organoid]:
    """Union of organoids found by different detectors, deduplicated."""
    allo: list[Organoid] = []
    for g in groups:
        allo.extend(g)
    return _drop_duplicates(allo, acq)


def _band_from_contour(det, shape, half_width: int):
    """Rasterise a detection's outline and return its boundary band.

    The band, not the disc: in brightfield the contrast lives on the rim, and a
    cystic organoid's interior is featureless.
    """
    import cv2

    mask = np.zeros(shape, dtype=np.uint8)
    cv2.fillPoly(mask, [det.contour.astype(np.int32)], 1)
    if mask.sum() == 0:
        return None
    return contour_band(mask, half_width)


def _nearest_det(t: Track, z: int, z_limit: int):
    cands = [d for d in t.dets if d.z < z_limit]
    return min(cands, key=lambda d: abs(d.z - z)) if cands else None


def _drop_duplicates(organoids: list[Organoid], acq: Acquisition) -> list[Organoid]:
    """Merge objects that split into two focal planes but are the same organoid.

    Peak splitting is deliberately generous; this is the counterweight. Two
    entries at nearly the same (x, y, z) with similar radii are one organoid
    whose focus curve happened to be double-humped -- keep the sharper one.

    Distances are compared in isotropic pixel units so the axial and lateral
    tolerances mean the same thing.
    """
    aniso = acq.anisotropy
    dof_iso = acq.depth_of_field_slices * aniso
    kept: list[Organoid] = []
    for o in sorted(organoids, key=lambda x: -x.focus_sharpness):
        dup = False
        for k in kept:
            lateral = float(np.hypot(o.x_px - k.x_px, o.y_px - k.y_px))
            axial = abs(o.z_slice - k.z_slice) * aniso
            ref = max(o.radius_px, k.radius_px)
            if lateral < 0.5 * ref and axial < max(dof_iso, 0.6 * ref):
                dup = True
                break
        if not dup:
            kept.append(o)
    kept.sort(key=lambda x: (x.z_slice, x.x_px))
    for i, o in enumerate(kept, start=1):
        o.oid = i
    return kept


def filter_organoids(organoids: list[Organoid], min_sharpness: float = 0.25
                     ) -> tuple[list[Organoid], int]:
    """Drop objects with no genuine focal plane.

    Out-of-focus haze and flat background texture produce a focus curve with no
    peak. Requiring real peak prominence is what separates an organoid from its
    own shadow.
    """
    keep = [o for o in organoids if o.focus_sharpness >= min_sharpness]
    return keep, len(organoids) - len(keep)


# --------------------------------------------------------------------------- #
# Meshing
# --------------------------------------------------------------------------- #

def spheroid_mesh(o: Organoid, n_phi: int, anisotropy: float
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Vertices/faces of one organoid, in isotropic pixel units.

    The equatorial ring is the measured outline r(theta); it is scaled by
    sin(phi) towards the poles, with the poles at +/- the axial semi-axis.
    Working in isotropic pixels keeps the mesh correct whether or not the stack
    has a micrometre calibration.
    """
    r_theta = np.asarray(o.radial_profile_px, dtype=np.float64)
    n_theta = r_theta.size
    theta = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    phi = np.linspace(0, np.pi, n_phi)

    sin_p = np.sin(phi)[:, None]
    cos_p = np.cos(phi)[:, None]
    rz_iso = o.radius_z_slices * anisotropy

    x = o.x_px + r_theta[None, :] * sin_p * np.cos(theta)[None, :]
    y = o.y_px + r_theta[None, :] * sin_p * np.sin(theta)[None, :]
    z = o.z_slice * anisotropy + rz_iso * cos_p * np.ones((1, n_theta))

    verts = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)

    faces = []
    for i in range(n_phi - 1):
        for j in range(n_theta):
            j2 = (j + 1) % n_theta
            a = i * n_theta + j
            b = i * n_theta + j2
            c = (i + 1) * n_theta + j
            d = (i + 1) * n_theta + j2
            faces.append([a, c, d])
            faces.append([a, d, b])
    return verts, np.asarray(faces, dtype=np.int64)


def export_mesh(organoids: list[Organoid], params: Params, path: str,
                acq: Acquisition) -> tuple[str, str]:
    """Write every organoid into one colour-coded PLY.

    Returns (path, unit). The mesh is in micrometres when the stack is
    calibrated and in isotropic pixels otherwise -- never in a made-up scale.
    """
    import trimesh
    from matplotlib import colormaps

    cmap = colormaps.get_cmap("turbo")
    if not organoids:
        raise ValueError("no organoids to export")

    scale = acq.px_um if acq.calibrated else 1.0
    unit = "um" if acq.calibrated else "isotropic_px"

    zs = np.array([o.z_slice for o in organoids])
    lo, hi = float(zs.min()), float(zs.max())
    span = max(hi - lo, 1e-6)

    all_v, all_f, all_c = [], [], []
    offset = 0
    for o in organoids:
        v, f = spheroid_mesh(o, params.n_phi, acq.anisotropy)
        rgba = cmap((o.z_slice - lo) / span)
        colour = (np.array(rgba[:3]) * 255).astype(np.uint8)
        all_v.append(v * scale)
        all_f.append(f + offset)
        all_c.append(np.tile(colour, (v.shape[0], 1)))
        offset += v.shape[0]

    mesh = trimesh.Trimesh(
        vertices=np.vstack(all_v),
        faces=np.vstack(all_f),
        vertex_colors=np.vstack(all_c),
        process=False,
    )
    mesh.export(path)
    return path, unit
