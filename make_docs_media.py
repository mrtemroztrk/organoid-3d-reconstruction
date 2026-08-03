#!/usr/bin/env python
"""Build the animated figures used in README.md.

    ./.venv/bin/python make_docs_media.py output/4x_00009

Everything here is rendered from real pipeline output -- the same geometry the
interactive viewer draws, produced offline so the README works without anyone
having to run the tool first.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import vtk
from vtk.util import numpy_support as vnp

from jx3d.config import Params
from jx3d.qc import _draw_slice
from jx3d.stack import load_stack
from render_3d import _box_actor, _mesh_actor, _polydata


# --------------------------------------------------------------------------- #
# 3D slicer frame: raw slice textured into the volume, models clipped at it
# --------------------------------------------------------------------------- #

class SlicerScene:
    """Reusable VTK scene; only the texture and the clip plane move per frame."""

    def __init__(self, ply: Path, dims, size=(760, 620), dome=None,
                 scale: float = 1.0):
        self.dims = dims
        self.ren = vtk.vtkRenderer()
        self.ren.SetBackground(0.031, 0.043, 0.070)
        self.ren.SetBackground2(0.075, 0.098, 0.145)
        self.ren.GradientBackgroundOn()

        self.mesh = _mesh_actor(ply)
        self.clip = vtk.vtkPlane()
        self.clip.SetNormal(0, 0, 1)          # keep the half deeper than the plane
        self.mesh.GetMapper().AddClippingPlane(self.clip)
        self.ren.AddActor(self.mesh)
        self.ren.AddActor(_box_actor(dims))
        from render_3d import _dome_actor
        for a in _dome_actor(dome, dims, scale):
            self.ren.AddActor(a)

        # textured plane carrying the raw slice
        x, y, _ = dims
        src = vtk.vtkPlaneSource()
        src.SetOrigin(0, 0, 0)
        src.SetPoint1(x, 0, 0)
        src.SetPoint2(0, y, 0)
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(src.GetOutputPort())
        self.image = vtk.vtkImageData()
        self.texture = vtk.vtkTexture()
        self.texture.SetInputData(self.image)
        self.texture.InterpolateOn()
        self.plane = vtk.vtkActor()
        self.plane.SetMapper(mapper)
        self.plane.SetTexture(self.texture)
        self.plane.GetProperty().LightingOff()
        self.ren.AddActor(self.plane)

        self.win = vtk.vtkRenderWindow()
        self.win.SetOffScreenRendering(1)
        self.win.AddRenderer(self.ren)
        self.win.SetSize(*size)
        self.win.SetMultiSamples(8)

        cx, cy, cz = x / 2, y / 2, dims[2] / 2
        cam = self.ren.GetActiveCamera()
        cam.SetFocalPoint(cx, cy, cz)
        cam.SetPosition(cx, cy - max(dims) * 2.0, cz)
        cam.SetViewUp(0, 0, -1)
        cam.Azimuth(32)
        cam.Elevation(24)
        cam.OrthogonalizeViewUp()
        self.ren.ResetCamera()
        cam.Zoom(1.35)

        lk = vtk.vtkLightKit()
        lk.SetKeyLightIntensity(1.05)
        lk.SetKeyToFillRatio(2.6)
        lk.AddLightsToRenderer(self.ren)

        self.grab = vtk.vtkWindowToImageFilter()
        self.grab.SetInput(self.win)

    def frame(self, slice_img: np.ndarray, z_um: float) -> np.ndarray:
        h, w = slice_img.shape[:2]
        self.image.SetDimensions(w, h, 1)
        # Row 0 of the array becomes texture v=0, which sits at plane y=0 -- the
        # same orientation the 2D overlay uses, so the two panels line up.
        arr = vnp.numpy_to_vtk(np.ascontiguousarray(slice_img.reshape(-1, 3)[:, ::-1]),
                               deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
        arr.SetNumberOfComponents(3)
        self.image.GetPointData().SetScalars(arr)
        self.image.Modified()
        self.texture.Modified()

        self.plane.SetPosition(0, 0, z_um)
        self.clip.SetOrigin(0, 0, z_um)

        self.win.Render()
        self.grab.Modified()
        self.grab.Update()
        img = self.grab.GetOutput()
        gw, gh, _ = img.GetDimensions()
        out = vnp.vtk_to_numpy(img.GetPointData().GetScalars()).reshape(gh, gw, -1)[::-1]
        return cv2.cvtColor(out[:, :, :3].astype(np.uint8), cv2.COLOR_RGB2BGR)


class OrbitScene:
    """Same scene, camera flying around it — one frame per azimuth step."""

    def __init__(self, ply: Path, dims, substrate_um: float, size=(900, 700),
                 dome=None, scale: float = 1.0):
        from render_3d import _dome_actor, _substrate_actor

        self.dims = dims
        self.ren = vtk.vtkRenderer()
        self.ren.SetBackground(0.031, 0.043, 0.070)
        self.ren.SetBackground2(0.075, 0.098, 0.145)
        self.ren.GradientBackgroundOn()
        self.ren.AddActor(_mesh_actor(ply))
        self.ren.AddActor(_box_actor(dims))
        for a in _substrate_actor(dims, substrate_um, filled=True):
            self.ren.AddActor(a)
        for a in _dome_actor(dome, dims, scale):
            self.ren.AddActor(a)

        self.win = vtk.vtkRenderWindow()
        self.win.SetOffScreenRendering(1)
        self.win.AddRenderer(self.ren)
        self.win.SetSize(*size)
        self.win.SetMultiSamples(8)

        lk = vtk.vtkLightKit()
        lk.SetKeyLightIntensity(1.05)
        lk.SetKeyToFillRatio(2.6)
        lk.AddLightsToRenderer(self.ren)

        self.cx, self.cy, self.cz = dims[0] / 2, dims[1] / 2, dims[2] / 2
        self.radius = max(dims) * 1.45
        self.grab = vtk.vtkWindowToImageFilter()
        self.grab.SetInput(self.win)

    def frame(self, azimuth_deg: float, elevation_deg: float = 24.0) -> np.ndarray:
        a = np.radians(azimuth_deg)
        e = np.radians(elevation_deg)
        cam = self.ren.GetActiveCamera()
        cam.SetFocalPoint(self.cx, self.cy, self.cz)
        cam.SetPosition(self.cx + self.radius * np.cos(e) * np.cos(a),
                        self.cy + self.radius * np.cos(e) * np.sin(a),
                        self.cz - self.radius * np.sin(e))
        cam.SetViewUp(0, 0, -1)
        cam.OrthogonalizeViewUp()
        self.ren.ResetCameraClippingRange()

        self.win.Render()
        self.grab.Modified()
        self.grab.Update()
        img = self.grab.GetOutput()
        gw, gh, _ = img.GetDimensions()
        out = vnp.vtk_to_numpy(img.GetPointData().GetScalars()).reshape(gh, gw, -1)[::-1]
        return cv2.cvtColor(out[:, :, :3].astype(np.uint8), cv2.COLOR_RGB2BGR)


def _curve_panel(curve: np.ndarray, z_now: int, z_peak: float, z0: int,
                 w: int, h: int) -> np.ndarray:
    """Little plot of one organoid's rim sharpness against Z."""
    p = np.zeros((h, w, 3), dtype=np.uint8)
    p[:] = (24, 17, 11)
    pad_l, pad_b, pad_t = 44, 26, 18
    lo, hi = float(curve.min()), float(curve.max())
    span = max(hi - lo, 1e-9)

    def xy(i, v):
        x = pad_l + i / max(1, len(curve) - 1) * (w - pad_l - 12)
        y = h - pad_b - (v - lo) / span * (h - pad_b - pad_t)
        return int(x), int(y)

    cv2.line(p, (pad_l, pad_t - 6), (pad_l, h - pad_b), (70, 70, 80), 1)
    cv2.line(p, (pad_l, h - pad_b), (w - 10, h - pad_b), (70, 70, 80), 1)

    pts = np.array([xy(i, v) for i, v in enumerate(curve)], dtype=np.int32)
    cv2.polylines(p, [pts], False, (225, 208, 77), 2, cv2.LINE_AA)

    xp, _ = xy(z_peak - z0, lo)
    cv2.line(p, (xp, pad_t - 6), (xp, h - pad_b), (110, 230, 110), 1, cv2.LINE_AA)
    cv2.putText(p, "focal plane", (xp + 5, pad_t + 6), cv2.FONT_HERSHEY_SIMPLEX,
                0.36, (110, 230, 110), 1, cv2.LINE_AA)

    i_now = int(np.clip(z_now - z0, 0, len(curve) - 1))
    xn, yn = xy(i_now, curve[i_now])
    cv2.line(p, (xn, pad_t - 6), (xn, h - pad_b), (80, 80, 230), 1, cv2.LINE_AA)
    cv2.circle(p, (xn, yn), 4, (80, 80, 240), -1, cv2.LINE_AA)

    cv2.putText(p, "rim sharpness", (6, pad_t + 2), cv2.FONT_HERSHEY_SIMPLEX,
                0.38, (170, 190, 215), 1, cv2.LINE_AA)
    cv2.putText(p, "Z slice", (w // 2 - 20, h - 8), cv2.FONT_HERSHEY_SIMPLEX,
                0.38, (170, 190, 215), 1, cv2.LINE_AA)
    return p


def _label(img: np.ndarray, text: str, sub: str = "") -> np.ndarray:
    h = img.shape[0]
    cv2.rectangle(img, (0, h - (34 if sub else 22)), (img.shape[1], h), (16, 11, 7), -1)
    cv2.putText(img, text, (10, h - (22 if sub else 7)), cv2.FONT_HERSHEY_SIMPLEX,
                0.46, (240, 245, 255), 1, cv2.LINE_AA)
    if sub:
        cv2.putText(img, sub, (10, h - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.38, (150, 170, 200), 1, cv2.LINE_AA)
    return img



def _save_gif(path: Path, frames, delay_ms: int, palettesize: int = 64) -> None:
    """Write a GIF, quantised hard, with a frame delay that actually sticks.

    `duration` here is in **milliseconds**. Passing seconds silently rounds the
    delay to zero, the file ends up with no timing at all, and every browser
    falls back to its own minimum -- which is why several earlier attempts to
    slow these animations down changed nothing. The written delay is read back
    and asserted for exactly that reason.

    These are mostly greyscale micrographs, so a 64-colour palette is visually
    indistinguishable from 256 and roughly halves the file. Four full-size GIFs
    in one README is a slow page otherwise.
    """
    import imageio.v2 as imageio
    from PIL import Image

    imageio.mimsave(path, frames, format="GIF", duration=int(delay_ms), loop=0,
                    palettesize=palettesize)

    with Image.open(path) as im:
        im.seek(min(1, im.n_frames - 1))
        written = im.info.get("duration", 0)
    if not written:
        raise RuntimeError(f"{path}: frame delay was not stored (got {written!r})")
    print(f"  {path}  ({path.stat().st_size / 1e6:.1f} MB, "
          f"{written} ms/frame, {im.n_frames * written / 1000:.1f} s loop)")


def _focus_gif(stack, organoids, params, docs, imageio, n_frames,
               crop_px: int = 260, panel_w: int = 900) -> None:
    """Zoom on one organoid while Z sweeps, next to its live sharpness curve."""
    from jx3d.focus import contour_band, focus_curve
    from jx3d.reconstruct import _band_from_contour

    # a big, well-isolated, confidently-focused organoid
    cand = [o for o in organoids if o.diameter_px > 30 and o.focus_sharpness > 0.5]
    o = max(cand or organoids, key=lambda x: x.focus_sharpness)

    theta = np.linspace(0, 2 * np.pi, params.n_theta, endpoint=False)
    r_px = np.asarray(o.radial_profile_px)
    poly = np.stack([o.x_px + r_px * np.cos(theta),
                     o.y_px + r_px * np.sin(theta)], axis=1).astype(np.float32)

    mask = np.zeros(stack.data.shape[1:], dtype=np.uint8)
    cv2.fillPoly(mask, [poly.astype(np.int32)], 1)
    band = contour_band(mask, params.focus_band_px)

    half = crop_px // 2
    x0 = int(np.clip(o.x_px - half, 0, stack.width - crop_px))
    y0 = int(np.clip(o.y_px - half, 0, stack.height - crop_px))

    z_lo = max(0, o.best_slice - 26)
    z_hi = min(stack.depth, o.best_slice + 27)
    zwin = range(z_lo, z_hi)
    curve = focus_curve(stack.data, band, zwin)

    zs = np.linspace(z_lo, z_hi - 1, n_frames).round().astype(int)
    frames = []
    for z in zs:
        crop = stack.data[int(z), y0:y0 + crop_px, x0:x0 + crop_px]
        img = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
        img = cv2.resize(img, (panel_w // 2, panel_w // 2), interpolation=cv2.INTER_NEAREST)

        k = (panel_w // 2) / crop_px
        if abs(int(z) - o.best_slice) <= 1:
            pts = ((poly - [x0, y0]) * k).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(img, [pts], True, (60, 230, 255), 2, cv2.LINE_AA)

        img = _label(img, f"organoid #{o.oid} · {o.diameter_px:.0f} px across"
                          f" · slice Z{int(z) + 1:03d}",
                     "sharp rim only near its own equator")
        panel = _curve_panel(curve, int(z), o.z_slice, z_lo,
                             panel_w - panel_w // 2, panel_w // 2)
        frames.append(cv2.cvtColor(np.hstack([img, panel]), cv2.COLOR_BGR2RGB))

    _save_gif(docs / "focus.gif", frames, 150, palettesize=64)


def _edf_gif(stack, params, docs, imageio, prof, meta,
             imageio_frames: int = 28, width: int = 840) -> None:
    """Left: slices sweeping. Right: the all-in-focus image building up."""
    from jx3d.focus import tenengrad

    z_limit = meta["substrate_slice"] - params.substrate_margin_slices + 1
    zs = list(range(0, z_limit))
    keep = set(np.linspace(0, len(zs) - 1, imageio_frames).round().astype(int))

    h, w = stack.data.shape[1:]
    best_val = np.full((h, w), -np.inf, dtype=np.float32)
    edf = np.zeros((h, w), dtype=np.uint8)

    pane = width // 2
    frames = []
    for i, z in enumerate(zs):
        sl = stack.data[z]
        s = cv2.GaussianBlur(tenengrad(sl), (0, 0), params.edf_smooth_sigma)
        better = s > best_val
        best_val[better] = s[better]
        edf[better] = sl[better]
        if i not in keep:
            continue

        ph = int(h * pane / w)
        left = cv2.resize(cv2.cvtColor(sl, cv2.COLOR_GRAY2BGR), (pane, ph),
                          interpolation=cv2.INTER_AREA)
        right = cv2.resize(cv2.cvtColor(edf, cv2.COLOR_GRAY2BGR), (pane, ph),
                           interpolation=cv2.INTER_AREA)
        left = _label(left, f"raw slice Z{z + 1:03d}",
                      "most organoids here are out of focus")
        right = _label(right, "all-in-focus projection, building up",
                       "sharpest slice per pixel — every organoid crisp at once")
        frames.append(cv2.cvtColor(np.hstack([left, right]), cv2.COLOR_BGR2RGB))
        print(f"\r  edf {len(frames)}/{imageio_frames}", end="", flush=True)
    print()

    for _ in range(6):                      # hold on the finished projection
        frames.append(frames[-1])
    _save_gif(docs / "edf.gif", frames, 150, palettesize=64)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Build README figures")
    ap.add_argument("outdir")
    ap.add_argument("--docs", default="docs")
    ap.add_argument("--frames", type=int, default=26)
    ap.add_argument("--width", type=int, default=900)
    a = ap.parse_args(argv)

    import imageio.v2 as imageio

    outdir = Path(a.outdir)
    docs = Path(a.docs)
    docs.mkdir(parents=True, exist_ok=True)

    meta = json.loads((outdir / "organoids.json").read_text())
    units = meta["units"]
    aniso = units["anisotropy"]
    scale = units["px_um"] if units["calibrated"] else 1.0
    CAL = units["calibrated"]
    params = Params(**{k: v for k, v in meta["params"].items()
                       if k in Params.__dataclass_fields__})

    stack = load_stack(Path("BK52_WT_9805_B") / meta["dataset"])
    from jx3d.reconstruct import Organoid
    organoids = [Organoid(**{k: v for k, v in o.items()
                             if k in Organoid.__dataclass_fields__})
                 for o in meta["organoids"]]
    for o in organoids:
        o.z_extent_slices = tuple(o.z_extent_slices)

    prof = np.load(outdir / "focus_profile.npy")
    dims = (stack.width * scale, stack.height * scale,
            len(prof) * aniso * scale)
    dome = meta.get("dome")

    lo = min(o.best_slice for o in organoids)
    hi = max(o.best_slice for o in organoids)
    zs = np.linspace(lo, hi, a.frames).round().astype(int)

    scene = SlicerScene(outdir / "organoids.ply", dims, dome=dome, scale=scale)

    pane_w = a.width // 2
    frames = []
    for i, z in enumerate(zs):
        # thicker strokes: these frames are halved before they land in the GIF
        left = _draw_slice(stack, organoids, int(z), params,
                           lw=4, lw_cross=2, cross_alpha=0.8, label_scale=1.6)
        n_focus = sum(1 for o in organoids if abs(o.best_slice - int(z)) <= 1)
        left = cv2.resize(left, (pane_w, int(left.shape[0] * pane_w / left.shape[1])),
                          interpolation=cv2.INTER_AREA)
        zlab = (f"Raw slice Z{z + 1:03d}"
                + (f"  ({z * units['z_um']:.0f} um deep)" if CAL else ""))
        left = _label(left, zlab,
                      f"solid = measured here ({n_focus})   faint = model cross-section")

        right = scene.frame(cv2.cvtColor(stack.data[int(z)], cv2.COLOR_GRAY2BGR),
                            float(z) * aniso * scale)
        right = cv2.resize(right, (pane_w, left.shape[0]), interpolation=cv2.INTER_AREA)
        right = _label(right, "3D reconstruction, clipped at the same plane",
                       "the photograph sits at its own depth inside the volume")

        frames.append(cv2.cvtColor(np.hstack([left, right]), cv2.COLOR_BGR2RGB))
        print(f"\r  frame {i + 1}/{len(zs)}", end="", flush=True)
    print()

    _save_gif(docs / "slicer.gif", frames, 160, palettesize=64)

    # single representative still, for the top of the README
    mid = frames[len(frames) // 2]
    cv2.imwrite(str(docs / "slicer.png"), cv2.cvtColor(mid, cv2.COLOR_RGB2BGR))

    # ------------------------------------------------------------ orbit.gif
    substrate_um = meta["substrate_slice"] * aniso * scale
    orbit = OrbitScene(outdir / "organoids.ply", dims, substrate_um,
                       dome=dome, scale=scale)
    frames = []
    n_orb = 72
    for i in range(n_orb):
        az = 360.0 * i / n_orb
        f = orbit.frame(az, 22.0)
        ow = 680
        f = cv2.resize(f, (ow, int(f.shape[0] * ow / f.shape[1])),
                       interpolation=cv2.INTER_AREA)
        cal_txt = (f"{units['px_um']:.2f} µm/px · {units['z_um']:.0f} µm/slice"
                   if CAL else "uncalibrated — pixels and slices")
        f = _label(f, f"{len(organoids)} organoids · {cal_txt}",
                   "colour = depth in the dome · translucent shell = fitted Matrigel surface")
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        print(f"\r  orbit {i + 1}/{n_orb}", end="", flush=True)
    print()
    # 72 frames over a full turn at 300 ms each: 21.6 s per revolution, 5 deg
    # per step. Slow enough to follow an individual organoid all the way round.
    _save_gif(docs / "orbit.gif", frames, 300, palettesize=96)

    # ------------------------------------------------------------- focus.gif
    # The core idea, on one organoid: its rim is crisp only near its own
    # equator. That single fact is the entire depth cue.
    _focus_gif(stack, organoids, params, docs, imageio, a.frames)

    # --------------------------------------------------------------- edf.gif
    _edf_gif(stack, params, docs, imageio, prof, meta, imageio_frames=28)

    # copy the static figures the pipeline already produced
    for name, dest in [("qc_edf.png", "recall.png"),
                       ("render_3d.png", "render_3d.png"),
                       ("qc_focus.png", "focus_profile.png"),
                       ("edf_depth.png", "depth_map.png"),
                       ("qc_slices.png", "qc_slices.png")]:
        src = outdir / name
        if not src.exists():
            continue
        img = cv2.imread(str(src))
        scale = min(1.0, 1400 / img.shape[1])
        if scale < 1.0:
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(docs / dest), img, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        print(f"  {docs / dest}  ({(docs / dest).stat().st_size / 1e6:.1f} MB)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
