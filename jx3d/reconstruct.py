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
    oid: int

    # position, in micrometres from the stack origin (x, y, depth)
    x_um: float
    y_um: float
    z_um: float

    # same, in native image coordinates
    cx_px: float
    cy_px: float
    z_slice: float          # fractional, sub-slice interpolated

    # size
    radius_um: float        # equivalent-circle radius of the equatorial outline
    diameter_um: float
    radius_z_um: float      # semi-axis along Z
    volume_um3: float
    area_um2: float         # equatorial cross-section

    # shape / quality
    circularity: float
    focus_sharpness: float  # peak prominence of the focus curve, 0..1
    n_slices: int
    z_extent_slices: tuple[int, int]

    # geometry: equatorial outline sampled at n_theta angles, in micrometres
    radial_profile_um: list[float]

    best_slice: int         # integer slice the outline was measured on
    source: str = "edf"     # which detector found it: 'edf' or 'slices'

    def to_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items()}
        d["z_extent_slices"] = list(self.z_extent_slices)
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

    dof_slices = acq.depth_of_field_um / acq.z_um
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
    dof_slices = acq.depth_of_field_um / acq.z_um
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
    r_prof_px = _radial_profile(det.contour, det.cx, det.cy, params.n_theta)
    r_eq_px = float(np.sqrt(det.area_px / np.pi))
    r_eq_um = r_eq_px * acq.px_um
    rz_um = r_eq_um * params.axial_ratio
    return Organoid(
        oid=0,
        x_um=det.cx * acq.px_um,
        y_um=det.cy * acq.px_um,
        z_um=z_focus * acq.z_um,
        cx_px=det.cx,
        cy_px=det.cy,
        z_slice=float(z_focus),
        radius_um=r_eq_um,
        diameter_um=2.0 * r_eq_um,
        radius_z_um=rz_um,
        volume_um3=(4.0 / 3.0) * np.pi * r_eq_um * r_eq_um * rz_um,
        area_um2=det.area_px * acq.px_um ** 2,
        circularity=det.circularity,
        focus_sharpness=float(prominence),
        n_slices=n_slices,
        z_extent_slices=tuple(z_extent),
        radial_profile_um=list(r_prof_px * acq.px_um),
        best_slice=int(round(z_focus)),
        source=source,
    )


def merge_sources(*groups: list[Organoid], acq: Acquisition) -> list[Organoid]:
    """Union of organoids found by different detectors, deduplicated."""
    allo: list[Organoid] = []
    for g in groups:
        allo.extend(g)
    return _drop_duplicates(allo, acq)


def _nearest_det(t: Track, z: int, z_limit: int):
    cands = [d for d in t.dets if d.z < z_limit]
    return min(cands, key=lambda d: abs(d.z - z)) if cands else None


def _drop_duplicates(organoids: list[Organoid], acq: Acquisition) -> list[Organoid]:
    """Merge objects that split into two focal planes but are the same organoid.

    Peak splitting is deliberately generous; this is the counterweight. Two
    entries at nearly the same (x, y, z) with similar radii are one organoid
    whose focus curve happened to be double-humped -- keep the sharper one.
    """
    kept: list[Organoid] = []
    for o in sorted(organoids, key=lambda x: -x.focus_sharpness):
        dup = False
        for k in kept:
            lateral = np.hypot(o.x_um - k.x_um, o.y_um - k.y_um)
            axial = abs(o.z_um - k.z_um)
            ref = max(o.radius_um, k.radius_um)
            if lateral < 0.5 * ref and axial < max(acq.depth_of_field_um, 0.6 * ref):
                dup = True
                break
        if not dup:
            kept.append(o)
    kept.sort(key=lambda x: (x.z_um, x.x_um))
    for i, o in enumerate(kept, start=1):
        o.oid = i
    return kept


def _band_from_contour(det, shape, half_width: int):
    """Rasterise a detection's outline and return its boundary band."""
    import cv2

    mask = np.zeros(shape, dtype=np.uint8)
    cv2.fillPoly(mask, [det.contour.astype(np.int32)], 1)
    if mask.sum() == 0:
        return None
    return contour_band(mask, half_width)


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

def spheroid_mesh(o: Organoid, n_phi: int) -> tuple[np.ndarray, np.ndarray]:
    """Vertices/faces (in um) of one organoid.

    The equatorial ring is the measured outline r(theta); it is scaled by
    sin(phi) going towards the poles and the poles sit at +/- radius_z.
    """
    r_theta = np.asarray(o.radial_profile_um, dtype=np.float64)
    n_theta = r_theta.size
    theta = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    phi = np.linspace(0, np.pi, n_phi)

    sin_p = np.sin(phi)[:, None]
    cos_p = np.cos(phi)[:, None]

    x = o.x_um + r_theta[None, :] * sin_p * np.cos(theta)[None, :]
    y = o.y_um + r_theta[None, :] * sin_p * np.sin(theta)[None, :]
    z = o.z_um + o.radius_z_um * cos_p * np.ones((1, n_theta))

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


def export_mesh(organoids: list[Organoid], params: Params, path: str) -> str:
    """Write every organoid into one coloured PLY, in micrometres."""
    import trimesh
    from matplotlib import colormaps

    cmap = colormaps.get_cmap("turbo")
    if not organoids:
        raise ValueError("no organoids to export")

    zs = np.array([o.z_um for o in organoids])
    lo, hi = float(zs.min()), float(zs.max())
    span = max(hi - lo, 1e-6)

    all_v, all_f, all_c = [], [], []
    offset = 0
    for o in organoids:
        v, f = spheroid_mesh(o, params.n_phi)
        rgba = cmap((o.z_um - lo) / span)
        colour = (np.array(rgba[:3]) * 255).astype(np.uint8)
        all_v.append(v)
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
    return path
