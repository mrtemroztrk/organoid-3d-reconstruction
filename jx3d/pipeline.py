"""End-to-end pipeline: Z-stack folder in, measured 3D organoids out."""
from __future__ import annotations

import csv
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from . import dome as domemod
from . import edf as edfmod
from . import focus as focusmod
from . import qc, viewer
from .config import Params
from .detect import Detection, _detections_from_labels, build_detector, segment_stack
from .link import drop_substrate_tracks, link_tracks
from .reconstruct import (Organoid, attach_dome, export_mesh, filter_organoids,
                          measure_regions, measure_tracks, merge_sources)
from .stack import ZStack, load_stack


def _viewer_version() -> str:
    from .viewer import template_version
    return template_version()


def _log(msg: str = "") -> None:
    print(msg, flush=True)


def _bar(done: int, total: int, width: int = 28) -> str:
    filled = int(width * done / max(1, total))
    return "[" + "#" * filled + "." * (width - filled) + f"] {done}/{total}"


def _inline(msg: str) -> None:
    sys.stdout.write("\r  " + msg + "   ")
    sys.stdout.flush()


@dataclass
class Result:
    stack: ZStack
    organoids: list[Organoid]
    focus_profile: np.ndarray
    substrate_slice: int
    outdir: Path
    focus_stack: edfmod.FocusStack | None = None
    dome: domemod.Dome | None = None


# --------------------------------------------------------------------------- #
# caching
# --------------------------------------------------------------------------- #

def _cache_key(stack: ZStack, params: Params, z_limit: int) -> dict:
    """What a cached result depends on.

    `z_limit` belongs here even though it is not a segmentation parameter. The
    projection and the labels are built over a range of slices, so a run that
    analyses a different range must not be served a cached result computed for
    the old one -- and that is not hypothetical: supplying the substrate plane
    from outside changes the range on exactly the tiles where the per-tile
    search got it wrong, which are the tiles whose cached results would be most
    misleading to reuse.
    """
    return {
        "detector": params.detector,
        "z_step": params.z_step,
        "shape": list(stack.data.shape),
        "expected_diameter_px": params.expected_diameter_px,
        "files": len(stack.files),
        "z_limit": int(z_limit),
    }


def _cache_load(outdir: Path, name: str, key: dict) -> np.ndarray | None:
    arr_path, meta_path = outdir / f"cache_{name}.npz", outdir / f"cache_{name}.json"
    if not (arr_path.exists() and meta_path.exists()):
        return None
    if json.loads(meta_path.read_text()) != key:
        return None
    return np.load(arr_path)["a"]


def _cache_save(outdir: Path, name: str, key: dict, arr: np.ndarray) -> None:
    np.savez_compressed(outdir / f"cache_{name}.npz", a=arr)
    (outdir / f"cache_{name}.json").write_text(json.dumps(key, indent=2))


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def run(folder: str | Path, outdir: str | Path, params: Params | None = None,
        gpu: bool = True, use_cache: bool = True, jpeg_quality: int = 72,
        min_sharpness: float = 0.25, build_html: bool = True,
        calibration: dict | None = None,
        substrate_override: float | None = None) -> Result:
    params = params or Params()
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    # ---------------------------------------------------------------- 1. load
    _log("=" * 74)
    _log("[1/7] Loading Z-stack")
    stack = load_stack(folder)
    for field, value in (calibration or {}).items():
        setattr(stack.acq, field, value)
        if field in ("px_um", "z_um"):
            setattr(stack.acq, f"{field}_source", "user")
    _log("  " + stack.describe().replace("\n", "\n  "))
    if stack.meta.get("source"):
        _log(f"  metadata: {Path(stack.meta['source']).name}")

    # ------------------------------------------------------- 2. focus profile
    _log("\n[2/7] Focus profile and glass surface")
    profile = focusmod.global_profile(stack.data)
    if substrate_override is not None:
        # In a tiled scan the glass is one plane shared by every tile, and some
        # tiles cannot see it: under the thickest gel its reflection never
        # becomes the sharpest thing in the field, and the per-field search then
        # settles thirty slices high, truncating the analysed range through the
        # densest part of the droplet. When the caller knows the plane from all
        # the tiles at once, that is better evidence than this field alone.
        substrate = int(round(substrate_override))
        own = focusmod.find_substrate_plane(profile)
        _log(f"  glass plane      : Z{substrate + 1:03d}  (given by the mosaic; "
             f"this field alone would have said Z{own + 1:03d})")
    else:
        substrate = focusmod.find_substrate_plane(profile)
        _log(f"  sharpest plane   : Z{substrate + 1:03d}  (well bottom / glass surface)")
    z_limit = max(3, min(stack.depth, substrate - params.substrate_margin_slices + 1))
    key = _cache_key(stack, params, z_limit)
    _log(f"  analysed range   : Z001 - Z{z_limit:03d}")
    _log(f"  depth of field   : ~{stack.acq.depth_of_field_um:.0f} µm "
         f"(~{stack.acq.depth_of_field_um / stack.acq.z_um:.1f} slices)"
         f"  ->  NOT optical sectioning; depth comes from focus")
    np.save(outdir / "focus_profile.npy", profile)
    seg_range = range(0, z_limit)

    # ----------------------------------------------- 2b. Matrigel dome surface
    dome = None
    if params.fit_dome:
        _log("\n[2b]  Matrigel dome surface")
        pts = domemod.ridge_points(stack.data, seg_range, sigma=params.dome_sigma,
                                   x_min_frac=params.dome_x_min_frac,
                                   threshold_k=params.dome_threshold_k,
                                   progress=lambda n, t: _inline(_bar(n, t)))
        sys.stdout.write("\n")
        dome = domemod.fit(pts, stack.acq.anisotropy, substrate,
                           stack.data.shape[1:])
        if dome is None:
            _log(f"  only {len(pts)} interface points found - no dome fitted")
        else:
            _log(f"  {dome.n_points} interface points, residual "
                 f"{dome.residual_px:.1f} px (p90 {dome.residual_p90_px:.1f})")
            _log(f"  contact radius {dome.contact_radius_px:.0f} px "
                 f"(+/-{dome.spread_pct:.0f}% bootstrap), "
                 f"height {dome.height_slices:.0f} slices, "
                 f"apex at slice {dome.apex_slice:.0f}")
            if dome.covers_field:
                _log("  droplet is wider than the field of view - the surface is "
                     "clipped to the imaged region")
            if stack.acq.calibrated:
                a = dome.contact_radius_px * stack.acq.px_um
                h = dome.height_slices * stack.acq.z_um
                # Spherical-cap volume. Printed every run as a standing check on
                # the axial scale: the Z pitch is stored as a bare integer with
                # no unit, and only the right reading of it yields a droplet
                # that could have been pipetted into a well.
                vol_ul = np.pi * h / 6.0 * (3.0 * a * a + h * h) / 1e9
                _log(f"  = {2 * a / 1000:.2f} mm across, {h / 1000:.2f} mm tall")
                _log(f"  droplet volume {vol_ul:.1f} µl"
                     + ("  (consistent with a pipetted Matrigel dome)"
                        if 5 <= vol_ul <= 120 else
                        "  <- OUTSIDE the usual 20-50 µl; check the Z scale"))

    edf_orgs: list[Organoid] = []
    slice_orgs: list[Organoid] = []
    fstack: edfmod.FocusStack | None = None

    # ------------------------------------------------- 3. all-in-focus (EDF)
    if params.mode in ("edf", "both"):
        _log("\n[3/7] All-in-focus projection (EDF) + depth map")
        cached = _cache_load(outdir, "edf", key) if use_cache else None
        if cached is not None:
            fstack = edfmod.FocusStack(edf=cached[0].astype(np.uint8),
                                       best_z=cached[1].astype(np.int16),
                                       peak=cached[2].astype(np.float32),
                                       z_range=seg_range)
            _log("  loaded from cache")
        else:
            fstack = edfmod.build(stack.data, seg_range, params.edf_smooth_sigma,
                                  progress=lambda n, t: _inline(_bar(n, t)))
            sys.stdout.write("\n")
            if use_cache:
                _cache_save(outdir, "edf", key,
                            np.stack([fstack.edf.astype(np.float32),
                                      fstack.best_z.astype(np.float32),
                                      fstack.peak]))
        cv2.imwrite(str(outdir / "edf.png"), fstack.edf)
        depth_vis = cv2.applyColorMap(
            (fstack.best_z.astype(np.float32) / max(1, z_limit - 1) * 255)
            .clip(0, 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        cv2.imwrite(str(outdir / "edf_depth.png"), depth_vis)
        _log("  wrote edf.png / edf_depth.png")

        # ------------------------------- 4a. segment the projection once
        _log("\n[4/7] Segmenting the projection (single pass)")
        lab = _cache_load(outdir, "edflabels", key) if use_cache else None
        if lab is None:
            det = build_detector(stack.acq, params, gpu=gpu)
            lab = det.segment(fstack.edf).astype(np.int32)
            if use_cache:
                _cache_save(outdir, "edflabels", key, lab.astype(np.int16))
        lab = lab.astype(np.int32)
        min_r = 0.5 * params.min_diameter_px
        max_r = 0.5 * params.max_diameter_px
        edf_dets = _detections_from_labels(lab, 0, params, min_r, max_r)
        _log(f"  {lab.max()} objects found, {len(edf_dets)} passed the size/shape "
             f"filter")

        _log("  locating focal planes (sweeping each outline's rim through Z)")
        edf_orgs = measure_regions(stack, edf_dets, params, z_max=z_limit,
                                   progress=lambda n, t: _inline(_bar(n, t)))
        sys.stdout.write("\n")
        _log(f"  {len(edf_orgs)} focal planes measured")
    else:
        _log("\n[3/7] EDF skipped (--mode slices)")
        _log("[4/7] —")

    # --------------------------------------- 5. per-slice segmentation + link
    if params.mode in ("slices", "both"):
        _log(f"\n[5/7] Per-slice segmentation ({params.detector}, "
             f"every {params.z_step} slice) + Z linking")
        labels = _cache_load(outdir, "labels", key) if use_cache else None
        if labels is not None:
            labels = labels.astype(np.int32)
            _log("  loaded from cache")
            min_r = 0.5 * params.min_diameter_px
            max_r = 0.5 * params.max_diameter_px
            per_slice: list[list[Detection]] = [[] for _ in range(stack.depth)]
            for z in range(stack.depth):
                if labels[z].max() > 0:
                    per_slice[z] = _detections_from_labels(labels[z], z, params,
                                                           min_r, max_r)
        else:
            labels, per_slice = segment_stack(
                stack, params, gpu=gpu, z_slice_range=seg_range,
                progress=lambda n, t, z, k: _inline(f"{_bar(n, t)}  Z{z + 1:03d}: {k}"))
            sys.stdout.write("\n")
            if use_cache:
                _cache_save(outdir, "labels", key, labels.astype(np.int16))

        n_det = sum(len(d) for d in per_slice)
        n_z = sum(1 for d in per_slice if d)
        _log(f"  {n_det} 2D objects across {n_z} slices "
             f"({n_det / max(1, n_z):.1f} per slice)")

        tracks = link_tracks(per_slice, params)
        tracks, dropped = drop_substrate_tracks(tracks, substrate,
                                                params.substrate_margin_slices)
        _log(f"  {len(tracks)} tracks" + (f" ({dropped} dropped as debris on the glass)"
                                          if dropped else ""))
        slice_orgs = measure_tracks(stack, tracks, params, z_max=z_limit,
                                    progress=lambda n, t: _inline(_bar(n, t)))
        sys.stdout.write("\n")
        _log(f"  {len(slice_orgs)} focal planes measured")
    else:
        _log("\n[5/7] Per-slice segmentation skipped (--mode edf)")

    # ------------------------------------------------------ 6. merge + filter
    _log("\n[6/7] Merge and quality filter")
    organoids = merge_sources(edf_orgs, slice_orgs, acq=stack.acq)
    if params.mode == "both":
        n_shared = len(edf_orgs) + len(slice_orgs) - len(organoids)
        _log(f"  EDF {len(edf_orgs)} + slices {len(slice_orgs)} "
             f"-> {len(organoids)} unique ({n_shared} overlapping)")

    organoids, dropped_flat = filter_organoids(organoids, min_sharpness)
    if dropped_flat:
        _log(f"  {dropped_flat} objects dropped for having no prominent focus peak "
             f"(out-of-focus ghosts / debris)")
    for i, o in enumerate(organoids, start=1):
        o.oid = i
    dome = domemod.validate(dome, organoids)
    if dome is not None and not dome.reliable:
        _log(f"  !! dome fit rejected: it encloses only "
             f"{100 * dome.encloses_frac:.0f}% of the organoids, and organoids "
             f"grow inside the gel. Clearance values are not reported.")
        dome = None
    attach_dome(organoids, dome, stack.acq)
    _log(f"  → {len(organoids)} organoids")

    if not organoids:
        # An empty field is a real answer, not a failure to answer. In a tiled
        # scan the corner tiles fall outside the droplet entirely, and a caller
        # collecting fifteen results needs an empty table from those rather than
        # a missing file it has to guess the meaning of.
        _log("\n   No organoids in this field. If that is unexpected, lower "
             "--min-sharpness or widen --min-diameter / --max-diameter.")
        _write_tables(organoids, stack, params, substrate, outdir, dome)
        return Result(stack, organoids, profile, substrate, outdir, fstack, dome)

    d = np.array([o.diameter_px for o in organoids])
    z = np.array([o.z_slice for o in organoids])
    by_src = {}
    for o in organoids:
        by_src[o.source] = by_src.get(o.source, 0) + 1
    _log(f"  diameter : median {np.median(d):.1f} px "
         f"(quartiles {np.percentile(d, 25):.1f} / {np.percentile(d, 75):.1f}, "
         f"max {d.max():.1f})")
    _log(f"  depth    : slices {z.min():.1f} - {z.max():.1f} (median {np.median(z):.1f})")
    _log(f"  volume   : {sum(o.volume_voxels for o in organoids) / 1e6:.2f} Mvoxel total")
    if dome is not None:
        gap = np.array([o.dome_distance_px for o in organoids if o.dome_distance_px is not None])
        if gap.size:
            _log(f"  dome gap : median {np.median(gap):.0f} px "
                 f"(min {gap.min():.0f}, max {gap.max():.0f})"
                 + (f"  outside the droplet: {(gap < 0).sum()}" if (gap < 0).any() else ""))
    if stack.acq.calibrated:
        du = np.array([o.diameter_um for o in organoids])
        _log(f"  calibrated: diameter median {np.median(du):.0f} µm, "
             f"total volume {sum(o.volume_um3 for o in organoids) / 1e9:.3f} µl "
             f"[{stack.acq.px_um_source} / {stack.acq.z_um_source}]")
    else:
        _log("  NOT CALIBRATED - no micrometre values are reported")
    _log(f"  source   : " + ", ".join(f"{k}={v}" for k, v in sorted(by_src.items())))

    # -------------------------------------------------------------- 7. output
    _log("\n[7/7] Outputs")
    _write_tables(organoids, stack, params, substrate, outdir, dome)
    _, mesh_unit = export_mesh(organoids, params, str(outdir / "organoids.ply"),
                               stack.acq)
    _log(f"  organoids.ply    (coordinates in {mesh_unit})")
    qc.montage(stack, organoids, params, outdir / "qc_slices.png")
    _log("  qc_slices.png    <- raw slices + measured outlines")
    if fstack is not None:
        qc.edf_overlay(fstack.edf, organoids, stack.acq, params,
                       outdir / "qc_edf.png")
        _log("  qc_edf.png       <- all-in-focus image + outlines (recall check)")
    qc.focus_plot(profile, substrate, organoids, outdir / "qc_focus.png")
    _log("  qc_focus.png")

    if build_html:
        html = viewer.build_viewer(stack, organoids, params, profile, substrate,
                                   outdir / "viewer.html", jpeg_quality=jpeg_quality, dome=dome,
                                   progress=lambda n, t: _inline(f"viewer {_bar(n, t)}"))
        sys.stdout.write("\n")
        _log(f"  viewer.html      ({html.stat().st_size / 1e6:.1f} MB, single file)")

    _log(f"\nDone in {time.time() - t_start:.1f} s")
    _log("=" * 74)
    return Result(stack, organoids, profile, substrate, outdir, fstack, dome)


def _write_tables(organoids: list[Organoid], stack: ZStack, params: Params,
                  substrate: int, outdir: Path, dome=None) -> None:
    """Write the per-object feature matrix.

    Pixel and slice columns come first because they are the measurement.
    Micrometre columns are appended only when the stack carries a real
    calibration; on an uncalibrated stack they are absent entirely rather than
    present and wrong.
    """
    rows = [o.to_dict() for o in organoids]
    acq = stack.acq

    (outdir / "organoids.json").write_text(json.dumps({
        "dataset": stack.name,
        "units": {
            "primary": "pixels (lateral) and slice indices (axial)",
            "calibrated": acq.calibrated,
            "px_um": acq.px_um,
            "px_um_source": acq.px_um_source,
            "z_um": acq.z_um,
            "z_um_source": acq.z_um_source,
            "anisotropy": round(acq.anisotropy, 4),
            "anisotropy_source": acq.anisotropy_source,
            "volume_voxels_definition": "1 voxel = 1 px * 1 px * 1 slice",
        },
        "acquisition": acq.to_dict(),
        "params": params.to_dict(),
        "substrate_slice": int(substrate),
        "viewer_version": _viewer_version(),
        "dome": dome.to_dict() if dome is not None else None,
        "n_organoids": len(organoids),
        "organoids": rows,
    }, indent=2), encoding="utf-8")

    # --- feature matrix: one row per object ---
    px_cols = [
        "oid", "source",
        "x_px", "y_px", "z_slice", "best_slice",
        "diameter_px", "radius_px", "radius_z_slices",
        "area_px2", "volume_voxels",
        "circularity", "focus_sharpness", "n_slices",
        "dome_distance_px", "dome_surface_slice",
    ]
    um_cols = ["x_um", "y_um", "z_um", "diameter_um", "radius_um", "radius_z_um",
               "area_um2", "volume_um3", "dome_distance_um"]
    cols = px_cols + (um_cols if acq.calibrated else [])

    with (outdir / "organoids.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    # --- the r(theta) outlines, one row per object ---
    with (outdir / "outlines_px.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["oid"] + [f"r{i:02d}_px" for i in range(params.n_theta)])
        for o in organoids:
            w.writerow([o.oid] + [round(v, 3) for v in o.radial_profile_px])

    unit_note = "px + µm" if acq.calibrated else "px only (uncalibrated)"
    _log(f"  organoids.csv    ({len(rows)} rows x {len(cols)} features, {unit_note})")
    _log(f"  outlines_px.csv  (r(theta) contour, {params.n_theta} angles per object)")
    _log("  organoids.json   (features + calibration provenance + dome fit)")
