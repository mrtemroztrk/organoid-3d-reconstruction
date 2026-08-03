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

from . import edf as edfmod
from . import focus as focusmod
from . import qc, viewer
from .config import Params
from .detect import Detection, _detections_from_labels, build_detector, segment_stack
from .link import drop_substrate_tracks, link_tracks
from .reconstruct import (Organoid, export_mesh, filter_organoids, measure_regions,
                          measure_tracks, merge_sources)
from .stack import ZStack, load_stack


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


# --------------------------------------------------------------------------- #
# caching
# --------------------------------------------------------------------------- #

def _cache_key(stack: ZStack, params: Params) -> dict:
    return {
        "detector": params.detector,
        "z_step": params.z_step,
        "shape": list(stack.data.shape),
        "expected_diameter_um": params.expected_diameter_um,
        "files": len(stack.files),
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
        min_sharpness: float = 0.25, build_html: bool = True) -> Result:
    params = params or Params()
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    # ---------------------------------------------------------------- 1. load
    _log("=" * 74)
    _log("[1/7] Loading Z-stack")
    stack = load_stack(folder)
    key = _cache_key(stack, params)
    _log("  " + stack.describe().replace("\n", "\n  "))
    if stack.meta.get("source"):
        _log(f"  metadata: {Path(stack.meta['source']).name}")

    # ------------------------------------------------------- 2. focus profile
    _log("\n[2/7] Focus profile and glass surface")
    profile = focusmod.global_profile(stack.data)
    substrate = focusmod.find_substrate_plane(profile)
    z_limit = max(3, min(stack.depth, substrate - params.substrate_margin_slices + 1))
    _log(f"  sharpest plane   : Z{substrate + 1:03d}  (well bottom / glass surface)")
    _log(f"  analysed range   : Z001 - Z{z_limit:03d}")
    _log(f"  depth of field   : ~{stack.acq.depth_of_field_um:.0f} µm "
         f"(~{stack.acq.depth_of_field_um / stack.acq.z_um:.1f} slices)"
         f"  ->  NOT optical sectioning; depth comes from focus")
    np.save(outdir / "focus_profile.npy", profile)
    seg_range = range(0, z_limit)

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
        min_r = 0.5 * params.min_diameter_um / stack.acq.px_um
        max_r = 0.5 * params.max_diameter_um / stack.acq.px_um
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
            min_r = 0.5 * params.min_diameter_um / stack.acq.px_um
            max_r = 0.5 * params.max_diameter_um / stack.acq.px_um
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
    _log(f"  → {len(organoids)} organoids")

    if not organoids:
        _log("\n!! No organoids found. Lower --min-sharpness, or widen the size "
             "range with --min-diameter / --max-diameter.")
        return Result(stack, organoids, profile, substrate, outdir, fstack)

    d = np.array([o.diameter_um for o in organoids])
    z = np.array([o.z_um for o in organoids])
    by_src = {}
    for o in organoids:
        by_src[o.source] = by_src.get(o.source, 0) + 1
    _log(f"  diameter : median {np.median(d):.0f} µm "
         f"(quartiles {np.percentile(d, 25):.0f} / {np.percentile(d, 75):.0f}, "
         f"max {d.max():.0f})")
    _log(f"  depth    : {z.min():.0f} - {z.max():.0f} µm (median {np.median(z):.0f})")
    _log(f"  volume   : {sum(o.volume_um3 for o in organoids) / 1e9:.3f} µl total")
    _log(f"  source   : " + ", ".join(f"{k}={v}" for k, v in sorted(by_src.items())))

    # -------------------------------------------------------------- 7. output
    _log("\n[7/7] Outputs")
    _write_tables(organoids, stack, params, substrate, outdir)
    export_mesh(organoids, params, str(outdir / "organoids.ply"))
    _log("  organoids.ply")
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
                                   outdir / "viewer.html", jpeg_quality=jpeg_quality,
                                   progress=lambda n, t: _inline(f"viewer {_bar(n, t)}"))
        sys.stdout.write("\n")
        _log(f"  viewer.html      ({html.stat().st_size / 1e6:.1f} MB, single file)")

    _log(f"\nDone in {time.time() - t_start:.1f} s")
    _log("=" * 74)
    return Result(stack, organoids, profile, substrate, outdir, fstack)


def _write_tables(organoids: list[Organoid], stack: ZStack, params: Params,
                  substrate: int, outdir: Path) -> None:
    rows = [o.to_dict() for o in organoids]

    (outdir / "organoids.json").write_text(json.dumps({
        "dataset": stack.name,
        "acquisition": stack.acq.to_dict(),
        "params": params.to_dict(),
        "substrate_slice": int(substrate),
        "n_organoids": len(organoids),
        "organoids": rows,
    }, indent=2), encoding="utf-8")

    cols = ["oid", "x_um", "y_um", "z_um", "diameter_um", "radius_um", "radius_z_um",
            "volume_um3", "area_um2", "circularity", "focus_sharpness",
            "best_slice", "n_slices", "source", "cx_px", "cy_px", "z_slice"]
    with (outdir / "organoids.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    _log(f"  organoids.csv / organoids.json  ({len(rows)} rows)")
