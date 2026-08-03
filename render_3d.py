#!/usr/bin/env python
"""Render the reconstruction to still images from several angles -- no browser.

    ./.venv/bin/python render_3d.py output/4x_00009

Writes <outdir>/render_3d.png (2x2 panel) plus the individual views.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import vtk
from vtk.util import numpy_support as vnp


# --------------------------------------------------------------------------- #
# scene setup
# --------------------------------------------------------------------------- #

def _mesh_actor(ply_path: Path) -> vtk.vtkActor:
    reader = vtk.vtkPLYReader()
    reader.SetFileName(str(ply_path))
    reader.Update()

    normals = vtk.vtkPolyDataNormals()
    normals.SetInputConnection(reader.GetOutputPort())
    normals.SplittingOff()
    normals.ConsistencyOn()
    normals.Update()

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(normals.GetOutputPort())
    mapper.SetScalarModeToUsePointData()
    mapper.ScalarVisibilityOn()

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    p = actor.GetProperty()
    p.SetInterpolationToPhong()
    p.SetSpecular(0.35)
    p.SetSpecularPower(28)
    p.SetDiffuse(0.85)
    p.SetAmbient(0.18)
    return actor


def _box_actor(dims, colour=(0.22, 0.28, 0.42)) -> vtk.vtkActor:
    x, y, z = dims
    cube = vtk.vtkCubeSource()
    cube.SetBounds(0, x, 0, y, 0, z)
    edges = vtk.vtkFeatureEdges()
    edges.SetInputConnection(cube.GetOutputPort())
    edges.BoundaryEdgesOn()
    edges.FeatureEdgesOff()
    edges.ManifoldEdgesOff()
    edges.NonManifoldEdgesOff()

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(edges.GetOutputPort())
    mapper.ScalarVisibilityOff()   # otherwise the actor colour is ignored and it draws red
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    p = actor.GetProperty()
    p.SetColor(*colour)
    p.SetLineWidth(1.4)
    p.LightingOff()            # LightKit's warm key light would tint the box red
    return actor


def _substrate_actor(dims, z_um: float, filled: bool = True) -> list:
    """Plane marking the glass surface (fill optional, outline always)."""
    x, y, _ = dims
    plane = vtk.vtkPlaneSource()
    plane.SetOrigin(0, 0, z_um)
    plane.SetPoint1(x, 0, z_um)
    plane.SetPoint2(0, y, z_um)
    plane.Update()
    out = []

    if filled:
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(plane.GetOutputPort())
        a = vtk.vtkActor()
        a.SetMapper(mapper)
        a.GetProperty().SetColor(1.0, 0.74, 0.36)
        a.GetProperty().SetOpacity(0.06)
        a.GetProperty().LightingOff()
        out.append(a)

    edges = vtk.vtkFeatureEdges()
    edges.SetInputConnection(plane.GetOutputPort())
    edges.BoundaryEdgesOn()
    edges.FeatureEdgesOff()
    em = vtk.vtkPolyDataMapper()
    em.SetInputConnection(edges.GetOutputPort())
    ea = vtk.vtkActor()
    ea.SetMapper(em)
    ea.GetProperty().SetColor(1.0, 0.74, 0.36)
    ea.GetProperty().SetOpacity(0.75)
    ea.GetProperty().SetLineWidth(1.6)
    ea.GetProperty().LightingOff()
    out.append(ea)
    return out


def _scale_bar(dims, length_um: float = 500.0) -> list:
    """Scale bar along the front lower edge of the volume."""
    x, y, z = dims
    pts = vtk.vtkPoints()
    pts.InsertNextPoint(x * 0.06, y * 1.02, z * 1.03)
    pts.InsertNextPoint(x * 0.06 + length_um, y * 1.02, z * 1.03)
    line = vtk.vtkCellArray()
    line.InsertNextCell(2)
    line.InsertCellPoint(0)
    line.InsertCellPoint(1)
    pd = vtk.vtkPolyData()
    pd.SetPoints(pts)
    pd.SetLines(line)

    tube = vtk.vtkTubeFilter()
    tube.SetInputData(pd)
    tube.SetRadius(max(dims) * 0.004)
    tube.SetNumberOfSides(12)

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(tube.GetOutputPort())
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(0.92, 0.94, 1.0)
    actor.GetProperty().LightingOff()

    label = vtk.vtkBillboardTextActor3D()
    label.SetInput(f"{length_um:.0f} um")
    label.SetPosition(x * 0.06 + length_um * 0.5, y * 1.02, z * 1.10)
    label.GetTextProperty().SetFontSize(22)
    label.GetTextProperty().SetColor(0.85, 0.89, 0.98)
    label.GetTextProperty().SetJustificationToCentered()
    return [actor, label]


def _caption(text: str, w: int, h: int) -> vtk.vtkTextActor:
    t = vtk.vtkTextActor()
    t.SetInput(text)
    t.SetDisplayPosition(16, h - 34)
    tp = t.GetTextProperty()
    tp.SetFontSize(21)
    tp.SetColor(0.90, 0.94, 1.0)
    tp.SetBold(True)
    return t


def render_view(ply: Path, dims, substrate_um: float, caption: str,
                azimuth: float, elevation: float, size=(1100, 850),
                zoom: float = 1.0, parallel: bool = False,
                filled_substrate: bool = True) -> np.ndarray:
    ren = vtk.vtkRenderer()
    ren.SetBackground(0.031, 0.043, 0.070)
    ren.SetBackground2(0.075, 0.098, 0.145)
    ren.GradientBackgroundOn()

    ren.AddActor(_mesh_actor(ply))
    ren.AddActor(_box_actor(dims))
    for a in _substrate_actor(dims, substrate_um, filled=filled_substrate):
        ren.AddActor(a)
    for a in _scale_bar(dims):
        ren.AddActor(a)
    ren.AddViewProp(_caption(caption, *size))

    win = vtk.vtkRenderWindow()
    win.SetOffScreenRendering(1)
    win.AddRenderer(ren)
    win.SetSize(*size)
    win.SetMultiSamples(8)

    cx, cy, cz = dims[0] / 2, dims[1] / 2, dims[2] / 2
    cam = ren.GetActiveCamera()
    cam.SetFocalPoint(cx, cy, cz)
    cam.SetPosition(cx, cy - max(dims) * 2.0, cz)
    cam.SetViewUp(0, 0, -1)          # +Z is depth, pointing down on screen
    cam.Azimuth(azimuth)
    cam.Elevation(elevation)
    cam.OrthogonalizeViewUp()
    if parallel:
        cam.ParallelProjectionOn()
    ren.ResetCamera()
    cam.Zoom(zoom)

    ren.SetTwoSidedLighting(True)
    ren.SetUseDepthPeeling(False)
    lk = vtk.vtkLightKit()
    lk.SetKeyLightIntensity(1.05)
    lk.SetKeyToFillRatio(2.6)
    lk.SetKeyToHeadRatio(3.2)
    lk.AddLightsToRenderer(ren)

    win.Render()
    grab = vtk.vtkWindowToImageFilter()
    grab.SetInput(win)
    grab.SetScale(1)
    grab.Update()

    img = grab.GetOutput()
    w, h, _ = img.GetDimensions()
    arr = vnp.vtk_to_numpy(img.GetPointData().GetScalars())
    return arr.reshape(h, w, -1)[::-1]


def _polydata(verts: np.ndarray, faces: np.ndarray) -> vtk.vtkPolyData:
    pts = vtk.vtkPoints()
    pts.SetData(vnp.numpy_to_vtk(np.ascontiguousarray(verts, dtype=np.float64), deep=True))
    cells = np.hstack([np.full((faces.shape[0], 1), 3, dtype=np.int64), faces]).ravel()
    ca = vtk.vtkCellArray()
    ca.SetCells(faces.shape[0],
                vnp.numpy_to_vtk(np.ascontiguousarray(cells), deep=True,
                                 array_type=vtk.VTK_ID_TYPE))
    pd = vtk.vtkPolyData()
    pd.SetPoints(pts)
    pd.SetPolys(ca)
    return pd


def render_closeups(organoids: list[dict], params, n: int = 4,
                    size=(560, 560)) -> np.ndarray:
    """Render the few largest organoids one by one, close up.

    Shows how much of the reconstruction is measurement: the bumps around the
    equator are the genuinely measured r(theta) outline, while the taper towards
    the poles is the sphericity assumption.
    """
    from matplotlib import colormaps
    from jx3d.reconstruct import Organoid, spheroid_mesh

    cmap = colormaps.get_cmap("turbo")
    zs = [o["z_um"] for o in organoids]
    lo, hi = min(zs), max(zs)
    span = max(hi - lo, 1e-6)

    picked = sorted(organoids, key=lambda o: -o["diameter_um"])[:n]
    tiles = []
    for o in picked:
        org = Organoid(**{k: v for k, v in o.items()
                          if k in Organoid.__dataclass_fields__})
        verts, faces = spheroid_mesh(org, params["n_phi"])
        verts = verts - verts.mean(axis=0)

        mapper = vtk.vtkPolyDataMapper()
        normals = vtk.vtkPolyDataNormals()
        normals.SetInputData(_polydata(verts, faces))
        normals.SplittingOff()
        normals.Update()
        mapper.SetInputConnection(normals.GetOutputPort())

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        p = actor.GetProperty()
        p.SetColor(*cmap((org.z_um - lo) / span)[:3])
        p.SetInterpolationToPhong()
        p.SetSpecular(0.45)
        p.SetSpecularPower(35)
        p.SetAmbient(0.16)

        ren = vtk.vtkRenderer()
        ren.SetBackground(0.031, 0.043, 0.070)
        ren.SetBackground2(0.085, 0.105, 0.155)
        ren.GradientBackgroundOn()
        ren.AddActor(actor)

        cap = vtk.vtkTextActor()
        cap.SetInput(f"#{org.oid}   {org.diameter_um:.0f} um\n"
                     f"depth {org.z_um:.0f} um  ·  Z{org.best_slice + 1}\n"
                     f"circularity {org.circularity:.2f}")
        cap.SetDisplayPosition(14, size[1] - 66)
        cap.GetTextProperty().SetFontSize(18)
        cap.GetTextProperty().SetColor(0.88, 0.92, 1.0)
        ren.AddViewProp(cap)

        win = vtk.vtkRenderWindow()
        win.SetOffScreenRendering(1)
        win.AddRenderer(ren)
        win.SetSize(*size)
        win.SetMultiSamples(8)

        cam = ren.GetActiveCamera()
        cam.SetFocalPoint(0, 0, 0)
        cam.SetPosition(0, -1, 0)
        cam.SetViewUp(0, 0, -1)
        cam.Azimuth(28)
        cam.Elevation(20)
        cam.OrthogonalizeViewUp()
        ren.ResetCamera()
        cam.Zoom(1.5)

        lk = vtk.vtkLightKit()
        lk.SetKeyLightIntensity(1.15)
        lk.SetKeyToFillRatio(2.2)
        lk.AddLightsToRenderer(ren)

        win.Render()
        grab = vtk.vtkWindowToImageFilter()
        grab.SetInput(win)
        grab.Update()
        img = grab.GetOutput()
        w, h, _ = img.GetDimensions()
        arr = vnp.vtk_to_numpy(img.GetPointData().GetScalars()).reshape(h, w, -1)[::-1]
        tiles.append(arr[:, :, :3].astype(np.uint8))

    row = np.hstack(tiles[:2]) if len(tiles) >= 2 else tiles[0]
    if len(tiles) >= 4:
        row = np.vstack([row, np.hstack(tiles[2:4])])
    return row


def _colorbar(width: int, lo_um: float, hi_um: float, height: int = 54) -> np.ndarray:
    """Depth colour scale (turbo), labelled in µm."""
    import cv2
    from matplotlib import colormaps

    cmap = colormaps.get_cmap("turbo")
    bar = np.zeros((height, width, 3), dtype=np.uint8)
    bar[:] = (18, 12, 8)
    x0, x1, y0, y1 = 130, width - 130, 12, 30
    grad = np.linspace(0, 1, x1 - x0)
    rgb = (np.array([cmap(t)[:3] for t in grad]) * 255).astype(np.uint8)
    bar[y0:y1, x0:x1] = rgb[None, :, ::-1]
    cv2.rectangle(bar, (x0, y0), (x1, y1), (90, 90, 90), 1)

    f, s, c = cv2.FONT_HERSHEY_SIMPLEX, 0.48, (215, 225, 240)
    cv2.putText(bar, "depth", (18, y1), f, s, c, 1, cv2.LINE_AA)
    cv2.putText(bar, f"{lo_um:.0f} um (dome top)", (x0, y1 + 18), f, 0.42, c, 1, cv2.LINE_AA)
    txt = f"{hi_um:.0f} um (near glass)"
    (tw, _), _ = cv2.getTextSize(txt, f, 0.42, 1)
    cv2.putText(bar, txt, (x1 - tw, y1 + 18), f, 0.42, c, 1, cv2.LINE_AA)
    return bar


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Render the 3D reconstruction to still images")
    ap.add_argument("outdir", help="pipeline output folder containing organoids.ply")
    ap.add_argument("--size", type=int, nargs=2, default=[1100, 850])
    a = ap.parse_args(argv)

    outdir = Path(a.outdir)
    ply = outdir / "organoids.ply"
    meta = json.loads((outdir / "organoids.json").read_text())
    acq = meta["acquisition"]

    # volume bounds, from the image dimensions
    prof = np.load(outdir / "focus_profile.npy")
    dims = (960 * acq["px_um"], 720 * acq["px_um"], len(prof) * acq["z_um"])
    substrate_um = meta["substrate_slice"] * acq["z_um"]
    n = meta["n_organoids"]

    views = [
        ("3/4 perspective", 34, 22, 1.30, False, True),
        (f"Top view (microscope view) - {n} organoids", 0, 89.9, 1.45, True, False),
        ("Side view - depth through the dome", 90, 0, 1.55, True, False),
    ]

    import cv2
    tiles = []
    for name, az, el, zoom, par, fill in views:
        img = render_view(ply, dims, substrate_um, name, az, el,
                          size=tuple(a.size), zoom=zoom, parallel=par,
                          filled_substrate=fill)
        bgr = cv2.cvtColor(img[:, :, :3].astype(np.uint8), cv2.COLOR_RGB2BGR)
        slug = name.split()[0].lower().replace("/", "-")
        cv2.imwrite(str(outdir / f"render_{slug}.png"), bgr)
        tiles.append(bgr)
        print(f"  {name}")

    close = render_closeups(meta["organoids"], meta["params"], n=4,
                            size=(a.size[0] // 2, a.size[1] // 2))
    close = cv2.cvtColor(close, cv2.COLOR_RGB2BGR)
    close = cv2.resize(close, tuple(a.size), interpolation=cv2.INTER_AREA)
    h_c = close.shape[0]
    cv2.rectangle(close, (0, h_c - 30), (close.shape[1], h_c), (16, 11, 7), -1)
    cv2.putText(close, "4 largest organoids - measured equatorial outline, revolved",
                (16, h_c - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (240, 245, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(outdir / "render_closeups.png"), close)
    tiles.append(close)
    print("  Close-ups (4 largest organoids)")

    h, w = tiles[0].shape[:2]
    zs = [o["z_um"] for o in meta["organoids"]]
    bar = _colorbar(w * 2 + 6, min(zs), max(zs))
    grid = np.zeros((h * 2 + 6 + bar.shape[0], w * 2 + 6, 3), dtype=np.uint8)
    grid[:] = (18, 12, 8)
    for i, t in enumerate(tiles):
        r, c = divmod(i, 2)
        grid[r * (h + 6):r * (h + 6) + h, c * (w + 6):c * (w + 6) + w] = t
    grid[-bar.shape[0]:] = bar
    cv2.imwrite(str(outdir / "render_3d.png"), grid)
    print(f"\n{outdir / 'render_3d.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
