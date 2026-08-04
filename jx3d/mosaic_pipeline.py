"""From fifteen tile folders to one feature matrix.

The order of the stages is not arbitrary; each one exists because the next
cannot be trusted without it.

The tiles have to be placed before anything can be said about the droplet, and
placed against the pixels rather than against the stage log, because the stage
is out by up to thirty pixels and that is enough to make one organoid look like
two. The illumination has to be flattened before any appearance is measured,
because otherwise the measurement partly encodes where in the frame the object
fell. The glass plane has to be settled across all fifteen tiles before any of
them is analysed, because two of them cannot see it and would otherwise cut
their own analysis short through the deepest part of the gel. Only then is there
a frame in which the dome can be fitted, detections can be merged, and features
can be compared between organoids that were never in the same photograph.

Detection itself is unchanged from the single-field pipeline, deliberately. It
is the part that was already validated against the raw images, and running it
per tile on each tile's own pixels is what keeps every number traceable back to
a photograph that actually exists.
"""
from __future__ import annotations

import csv
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import blend as blendmod
from . import dedup as dedupmod
from . import dome_global as domemod
from . import features as featuremod
from . import mosaic as mosaicmod
from . import register as registermod
from .config import Acquisition, Params
from .keyence import read_group_metadata


def _log(msg: str = "") -> None:
    print(msg, flush=True)


def _bar(done: int, total: int, width: int = 28) -> str:
    filled = int(width * done / max(1, total))
    return "[" + "#" * filled + "." * (width - filled) + f"] {done}/{total}"


def _inline(msg: str) -> None:
    sys.stdout.write("\r  " + msg + "   ")
    sys.stdout.flush()


@dataclass
class MosaicResult:
    mosaic: mosaicmod.Mosaic
    registration: registermod.Registration
    dome_fit: domemod.GlobalFit
    organoids: list[dedupmod.Organoid]
    report: dedupmod.MergeReport
    acq: Acquisition
    outdir: Path
    rows: list[dict]


def analyse_tiles(mosaic: mosaicmod.Mosaic, substrate: float, params: Params,
                  outdir: Path, gpu: bool = True, use_cache: bool = True,
                  min_sharpness: float = 0.25, flat_field=None
                  ) -> dict[str, list[dict]]:
    """Run the validated single-field measurement on each tile in turn.

    One tile is resident at a time. The full mosaic volume is 750 million
    voxels; a single tile is 82 megabytes, and there is never a reason to hold
    more than one, because detection is per tile by design.
    """
    from .pipeline import run as run_tile

    per_tile: dict[str, list[dict]] = {}
    for n, tile in enumerate(mosaic, start=1):
        tile_out = outdir / "tiles" / tile.name
        result_path = tile_out / "organoids.json"
        _log(f"\n  ---- tile {n}/{len(mosaic)}: {tile.name} "
             f"(row {tile.row}, col {tile.col}) ----")
        if use_cache and result_path.exists():
            # What a cached tile result depends on. The substrate is not enough:
            # switching from the projection-only pass to the per-slice one
            # changes what an outline means, and a cache keyed on depth alone
            # would hand back the old answer under the new setting without
            # saying so.
            payload = json.loads(result_path.read_text())
            cached = payload.get("params") or {}
            matches = (payload.get("substrate_slice") == int(round(substrate))
                       and cached.get("mode") == params.mode
                       and cached.get("detector") == params.detector
                       and cached.get("min_diameter_px") == params.min_diameter_px
                       and cached.get("max_diameter_px") == params.max_diameter_px
                       and bool(payload.get("flat_fielded")) == (flat_field is not None))
            if matches:
                per_tile[tile.name] = payload["organoids"]
                _log(f"  {len(per_tile[tile.name])} organoids (cached)")
                continue
            _log(f"  cached result was measured with different settings "
                 f"(mode {cached.get('mode')} vs {params.mode}); re-measuring")
        run_tile(tile.folder, tile_out, params=params, gpu=gpu,
                 use_cache=use_cache, build_html=False,
                 min_sharpness=min_sharpness, substrate_override=substrate,
                 flat_field=flat_field)
        if result_path.exists():
            per_tile[tile.name] = json.loads(result_path.read_text())["organoids"]
        else:
            # A tile can legitimately hold nothing -- the corners of this scan
            # fall outside the droplet. One empty field must not take the other
            # fourteen down with it.
            per_tile[tile.name] = []
            _log("  no measurements written for this tile; treating it as empty")
    return per_tile


def extract_features(mosaic: mosaicmod.Mosaic, slices: blendmod.MosaicSlices,
                     organoids: list[dedupmod.Organoid], dome, substrate: float,
                     acq: Acquisition, progress=None) -> list[dict]:
    """Measure appearance for every organoid, on the tile that was elected.

    Appearance is measured on the elected view's own tile, at that view's own
    equatorial slice, and never on a composite. The slice is read once per
    (tile, depth) pair rather than once per organoid, which turns what would be
    a thousand file reads into a few hundred.
    """
    points = np.array([[o.elected.x_px, o.elected.y_px,
                        o.elected.z_slice * mosaic_anisotropy(acq)]
                       for o in organoids], dtype=float)
    radii = np.array([o.elected.radius_px for o in organoids], dtype=float)

    by_frame: dict[tuple[str, int], list[int]] = {}
    for i, o in enumerate(organoids):
        by_frame.setdefault((o.elected.tile, int(round(o.elected.z_slice))),
                            []).append(i)

    rows: list[dict] = [dict() for _ in organoids]
    done = 0
    for (tile_name, z), indices in sorted(by_frame.items()):
        tile = mosaic.by_name(tile_name)
        frame = slices.tile_slice(tile, min(max(z, 0), slices.depth - 1))
        for i in indices:
            o = organoids[i]
            source = o.elected.source
            lx, ly = o.elected.x_px - tile.x0, o.elected.y_px - tile.y0
            near = [(p[0] - tile.x0, p[1] - tile.y0, r)
                    for p, r in zip(points[:, :2], radii)
                    if 0 < np.hypot(p[0] - o.elected.x_px, p[1] - o.elected.y_px)
                    < 3.0 * o.elected.radius_px + 40.0]
            row = featuremod.measure(
                frame, lx, ly, o.elected.radius_px,
                source.get("radial_profile_px") or [o.elected.radius_px],
                float(source.get("area_px2", 0.0)), neighbours=near)
            row.values.update(featuremod.spatial_features(
                o.elected.x_px, o.elected.y_px, o.elected.z_slice,
                o.elected.radius_px, dome, substrate, mosaic_anisotropy(acq)))
            row.values.update(featuremod.neighbourhood_features(i, points, radii))
            if row.missing:
                row.values["missing_blocks"] = ";".join(row.missing)
            rows[i] = row.values
            done += 1
            if progress:
                progress(done, len(organoids))
    return rows


def mosaic_anisotropy(acq: Acquisition) -> float:
    return acq.anisotropy


def run(dataset: str | Path, outdir: str | Path, params: Params | None = None,
        gpu: bool = True, use_cache: bool = True, min_sharpness: float = 0.25,
        calibration: dict | None = None,
        build_viewer: bool = True) -> MosaicResult:
    """The whole mosaic analysis, from tile folders to a feature matrix."""
    params = params or Params(mode="edf", fit_dome=False)
    dataset, outdir = Path(dataset), Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    _log("=" * 74)
    _log("[1/7] Tile geometry")
    acq, meta = read_group_metadata(dataset)
    for field_name, value in (calibration or {}).items():
        setattr(acq, field_name, value)
        if field_name in ("px_um", "z_um"):
            setattr(acq, f"{field_name}_source", "user")
    mosaic = mosaicmod.build(dataset, acq)
    if mosaic is None:
        raise ValueError(f"{dataset} does not describe a tiled acquisition")
    _log("  " + mosaic.describe(acq).replace("\n", "\n  "))

    _log("\n[2/7] Illumination")
    flat = blendmod.estimate_flat_field(
        mosaic, progress=lambda n, t: _inline(_bar(n, t)))
    sys.stdout.write("\n")
    _log(f"  flat field from {flat.n_frames} frames: the illumination varies "
         f"{flat.span_before:.0f} grey levels across a frame, {flat.span_after:.0f} "
         f"after correction")
    tiles = blendmod.MosaicSlices(mosaic, flat, blend=False)

    _log("\n[3/7] Refining the tile offsets against the pixels")
    mosaic, registration = registermod.register(
        mosaic, tiles, progress=lambda n, t: _inline(_bar(n, t)))
    sys.stdout.write("\n")
    _log("  " + registration.describe().replace("\n", "\n  "))
    if not registration.reliable:
        _log("  !! the mosaic geometry is not trustworthy; everything below "
             "inherits that")

    _log("\n[4/7] The glass plane, and the Matrigel dome")
    substrate = domemod.find_substrate(
        mosaic, tiles, progress=lambda n, t: _inline(_bar(n, t)))
    sys.stdout.write("\n")
    _log("  " + substrate.describe().replace("\n", "\n  "))
    fit = domemod.fit(mosaic, tiles, substrate, acq.anisotropy,
                      progress=lambda n, t: _inline(_bar(n, t)))
    sys.stdout.write("\n")
    dome = fit.dome
    if dome is None:
        _log("  no dome could be fitted; border distances will not be reported")
    else:
        _log(f"  axis ({dome.cx_px:.0f}, {dome.cy_px:.0f}) px, agreed by "
             f"{len([r for r in fit.rings if r.used])} slices to "
             f"({fit.axis_scatter_px[0]:.1f}, {fit.axis_scatter_px[1]:.1f}) px")
        _log(f"  contact radius {dome.contact_radius_px:.0f} px, cap residual "
             f"{fit.cap_residual_px:.1f} px rms")
        if acq.calibrated:
            a = dome.contact_radius_px * acq.px_um
            h = dome.height_slices * acq.z_um
            _log(f"  = {2 * a / 1000:.2f} mm across, {h / 1000:.2f} mm tall, "
                 f"{np.pi * h / 6.0 * (3 * a * a + h * h) / 1e9:.0f} ul")

    _log("\n[5/7] Measuring each tile")
    per_tile = analyse_tiles(mosaic, substrate.slice_index, params, outdir,
                             gpu=gpu, use_cache=use_cache,
                             min_sharpness=min_sharpness, flat_field=flat.gain)

    _log("\n[6/7] Merging the overlaps")
    organoids, report = dedupmod.merge(mosaic, per_tile, acq)
    _log("  " + report.describe().replace("\n", "\n  "))
    if dome is not None:
        inside = np.mean([
            float(dome.distance_px(o.elected.x_px, o.elected.y_px,
                                   o.elected.z_slice)) > 0
            for o in organoids])
        _log(f"  {100 * inside:.0f}% of them fall inside the fitted droplet "
             f"(organoids grow in the gel, so this should be nearly all of them)")

    _log("\n[7/7] Features and outputs")
    rows = extract_features(mosaic, tiles, organoids, dome,
                            substrate.slice_index, acq,
                            progress=lambda n, t: _inline(_bar(n, t)))
    sys.stdout.write("\n")
    merged = [{**o.to_dict(), **r} for o, r in zip(organoids, rows)]
    _write(merged, organoids, mosaic, registration, fit, report, acq,
           params, outdir)

    result = MosaicResult(mosaic, registration, fit, organoids, report, acq,
                          outdir, merged)
    if build_viewer:
        from . import viewer_mosaic

        html = viewer_mosaic.build(
            result, outdir / "viewer.html",
            progress=lambda n, t: _inline(f"viewer {_bar(n, t)}"))
        sys.stdout.write("\n")
        _log(f"  viewer.html      ({html.stat().st_size / 1e6:.1f} MB, one file, "
             f"no server)")

    _log(f"\nDone in {time.time() - started:.0f} s")
    _log("=" * 74)
    return result


def _write(rows: list[dict], organoids, mosaic, registration, fit, report,
           acq: Acquisition, params: Params, outdir: Path) -> None:
    """Write the feature matrix and everything needed to interpret it."""
    from . import __version__

    every: list[str] = []
    for row in rows:
        for key in row:
            if key not in every:
                every.append(key)
    # Keep the identity and provenance columns first: a matrix whose first
    # column is a texture descriptor invites someone to load it without ever
    # noticing which tile a row came from or whether it was clipped.
    lead = ["uid", "tile", "tile_row", "tile_col", "x_mosaic_px", "y_mosaic_px",
            "z_slice", "n_views", "views", "coverage_k", "clipped",
            "clipped_everywhere", "appearance_measurable"]
    every = [c for c in lead if c in every] + [c for c in every if c not in lead]

    # The matrix a person opens carries the columns a person reads. The other
    # hundred and twelve are texture bins and percentiles that exist because
    # they were cheap to compute; they are still measured and still written
    # alongside, but handing them to a biologist who wants to know which
    # organoids are dying is handing them a haystack.
    curated = [c for c in featuremod.VIABILITY_COLUMN_NAMES if c in every]

    def dump(path: Path, cols: list[str]) -> None:
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

    dump(outdir / "features.csv", curated)
    dump(outdir / "features_all.csv", every)
    _log(f"  features.csv     ({len(rows)} organoids x {len(curated)} columns "
         f"chosen for reading -- see features.py::VIABILITY_COLUMNS for why "
         f"each one is there)")
    _log(f"  features_all.csv (the same rows with all {len(every)} columns, for "
         f"model training)")

    n_views = 0
    with (outdir / "views.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["uid", "tile", "row", "col", "local_oid", "x_mosaic_px",
                         "y_mosaic_px", "z_slice", "radius_px", "focus_sharpness",
                         "circularity", "clipped", "edge_distance_px", "elected"])
        for o in organoids:
            for v in o.views:
                writer.writerow([o.uid, v.tile, v.row, v.col, v.local_oid,
                                 round(v.x_px, 3), round(v.y_px, 3),
                                 round(v.z_slice, 3), round(v.radius_px, 3),
                                 round(v.focus_sharpness, 4),
                                 round(v.circularity, 4), int(v.clipped),
                                 round(v.edge_distance_px, 2),
                                 int(v is o.elected)])
                n_views += 1
    repeats = n_views - len(organoids)
    _log(f"  views.csv        ({n_views} sightings; {repeats} of them are a "
         f"second independent measurement of an organoid another tile also saw, "
         f"which is a free error bar rather than a discard)")

    (outdir / "mosaic.json").write_text(json.dumps({
        "version": __version__,
        "feature_version": featuremod.FEATURE_VERSION,
        "dataset": str(mosaic.tiles[0].folder.parent),
        "units": {
            "primary": "pixels (lateral) and slice indices (axial)",
            "calibrated": acq.calibrated,
            "px_um": acq.px_um, "px_um_source": acq.px_um_source,
            "z_um": acq.z_um, "z_um_source": acq.z_um_source,
            "anisotropy": round(acq.anisotropy, 4),
        },
        "acquisition": acq.to_dict(),
        "params": params.to_dict(),
        "mosaic": mosaic.to_dict(),
        "registration": registration.to_dict(),
        "dome": fit.to_dict(),
        "merge": report.to_dict(),
        "n_organoids": len(rows),
    }, indent=2), encoding="utf-8")
    _log("  mosaic.json      (geometry, dome fit, merge report, provenance)")
    mosaic.save(outdir / "tiles.json")


# --------------------------------------------------------------------------- #
# rebuilding the viewer without re-analysing
# --------------------------------------------------------------------------- #

def _load_result(dataset: Path, outdir: Path) -> MosaicResult:
    """Reassemble a finished run from what it wrote, without re-measuring.

    Everything needed is already on disk: the tile placement and the dome fit in
    mosaic.json, one row per organoid in features_all.csv. The full matrix and
    not features.csv -- that one is trimmed to the columns a person reads, and
    two of the columns it drops are ones the viewer needs and nobody reads, the
    r(theta) outline and the modelled axial extent. Reading the trimmed file
    yields a viewer with no outlines to draw, which looks like a viewer showing
    photographs rather than like a bug.
    """
    import ast
    import types

    from .dome import Dome
    from .keyence import read_group_metadata
    from .mosaic import REGISTERED

    meta = json.loads((outdir / "mosaic.json").read_text())
    acq, _ = read_group_metadata(dataset)
    for field_name, value in (meta.get("units") or {}).items():
        if field_name in ("px_um", "z_um") and value is not None:
            setattr(acq, field_name, value)

    mosaic = mosaicmod.build(dataset, acq)
    if mosaic is None:
        raise ValueError(f"{dataset} does not describe a tiled acquisition")
    placed = {t["name"]: t for t in meta["mosaic"]["tiles"]}
    for tile in mosaic:
        record = placed.get(tile.name)
        if record is None:
            raise ValueError(f"{tile.name} is not in {outdir / 'mosaic.json'}")
        tile.x0, tile.y0 = record["x0_px"], record["y0_px"]
        tile.offset_source = record.get("offset_source", REGISTERED)

    matrix = outdir / "features_all.csv"
    if not matrix.exists():
        raise FileNotFoundError(
            f"{matrix} is missing. It is written by a full run and carries the "
            f"columns the viewer draws with; features.csv is a summary and "
            f"cannot be used in its place.")

    rows: list[dict] = []
    with matrix.open(encoding="utf-8") as fh:
        for record in csv.DictReader(fh):
            row: dict = {}
            for key, raw in record.items():
                if raw is None or raw == "":
                    continue
                if key == "radial_profile_px":
                    row[key] = ast.literal_eval(raw)
                    continue
                try:
                    row[key] = float(raw)
                except ValueError:
                    row[key] = {"True": True, "False": False}.get(raw, raw)
            rows.append(row)

    dome_record = (meta.get("dome") or {}).get("dome")
    dome = (Dome(**{k: v for k, v in dome_record.items()
                    if k in Dome.__dataclass_fields__})
            if dome_record else None)
    merge = meta.get("merge") or {}
    return MosaicResult(
        mosaic=mosaic,
        registration=types.SimpleNamespace(
            residual_px=(meta.get("registration") or {}).get("residual_px", 0.0)),
        dome_fit=types.SimpleNamespace(
            dome=dome,
            substrate=types.SimpleNamespace(
                slice_index=meta["dome"]["substrate"]["slice_index"])),
        organoids=[], report=types.SimpleNamespace(
            n_sightings=merge.get("n_sightings", len(rows)),
            n_merged=merge.get("n_merged", 0)),
        acq=acq, outdir=outdir, rows=rows)


def rebuild_viewer(dataset: str | Path, outdir: str | Path,
                   progress=None) -> Path:
    """Regenerate viewer.html from a finished run, in about a minute.

    This exists because the alternative was rebuilding it by hand. A generated
    viewer is a frozen copy of the template, so every change to the template
    leaves the file on disk behind, and re-running the whole analysis to pick up
    a change to a button is absurd. Without a supported way to do it, the job
    fell to a script kept outside the repository, which promptly drifted and
    started reading the wrong file.
    """
    from . import viewer_mosaic

    dataset, outdir = Path(dataset), Path(outdir)
    result = _load_result(dataset, outdir)
    _log(f"  {len(result.rows)} organoids, {len(result.mosaic)} fields, "
         f"offsets {result.mosaic.offset_source}")
    html = viewer_mosaic.build(result, outdir / "viewer.html", progress=progress)
    _log(f"  viewer.html      ({html.stat().st_size / 1e6:.1f} MB)")
    return html
