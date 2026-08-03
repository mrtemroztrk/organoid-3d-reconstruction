"""Quality-control figures: does the reconstruction match the raw image?

The whole point of the redesign is that every number must be checkable by eye,
so the pipeline always writes a montage of raw slices with the measured outlines
drawn on top. If an outline does not sit on a crisp dark rim, that organoid is
wrong -- no need to open the 3D viewer to find out.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .reconstruct import Organoid


def _draw_slice(stack, organoids: list[Organoid], z: int, params,
                lw: int = 2, lw_cross: int = 1, cross_alpha: float = 0.55,
                label_scale: float = 1.0) -> np.ndarray:
    """Raw slice with the reconstruction's outlines drawn on it.

    Line weights are adjustable because the same figure is used at full
    resolution (QC montage) and downscaled (README animation), and a 1 px line
    at 55% alpha disappears entirely once the image is halved.
    """
    acq = stack.acq
    rgb = cv2.cvtColor(stack.data[z], cv2.COLOR_GRAY2BGR)
    fs = 0.55 * label_scale
    n_focus = 0

    theta = np.linspace(0, 2 * np.pi, params.n_theta, endpoint=False)
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    for o in organoids:
        dz = z - o.z_slice
        if o.radius_z_slices <= 0 or abs(dz) >= o.radius_z_slices:
            continue
        s = float(np.sqrt(1.0 - (dz / o.radius_z_slices) ** 2))
        r = np.asarray(o.radial_profile_px) * s
        pts = np.stack([o.x_px + r * cos_t, o.y_px + r * sin_t], axis=1)
        pts = pts.astype(np.int32).reshape(-1, 1, 2)

        in_focus = abs(o.best_slice - z) <= 1
        if in_focus:
            n_focus += 1
            cv2.polylines(rgb, [pts], True, (60, 230, 255), lw, cv2.LINE_AA)
            cv2.putText(rgb, str(o.oid), (int(o.x_px) - 6, int(o.y_px) + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4 * label_scale, (60, 230, 255),
                        max(1, lw - 1), cv2.LINE_AA)
        else:
            # model cross-section: visible, but muted enough not to be confused
            # with the outline measured at the focal plane
            overlay = rgb.copy()
            cv2.polylines(overlay, [pts], True, (200, 120, 60), lw_cross, cv2.LINE_AA)
            cv2.addWeighted(overlay, cross_alpha, rgb, 1.0 - cross_alpha, 0, rgb)

    label = f"Z{z + 1:03d}   in focus: {n_focus}"
    if acq.calibrated:
        label += f"   ({z * acq.z_um:.0f} um deep)"
    bh = int(26 * label_scale)
    cv2.rectangle(rgb, (0, 0), (rgb.shape[1], bh), (20, 20, 20), -1)
    cv2.putText(rgb, label, (8, int(18 * label_scale)), cv2.FONT_HERSHEY_SIMPLEX, fs,
                (255, 255, 255), 1, cv2.LINE_AA)
    return rgb


def montage(stack, organoids: list[Organoid], params, output: str | Path,
            n_cols: int = 3, n_rows: int = 3, scale: float = 0.6) -> Path:
    """Grid of evenly spaced slices with reconstructed cross-sections drawn on."""
    if not organoids:
        raise ValueError("no organoids to draw")

    # Sample the depth range that actually contains organoids.
    lo = min(o.best_slice for o in organoids)
    hi = max(o.best_slice for o in organoids)
    zs = np.linspace(lo, hi, n_cols * n_rows).round().astype(int)

    tiles = []
    for z in zs:
        img = _draw_slice(stack, organoids, int(z), params)
        tiles.append(cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA))

    h, w = tiles[0].shape[:2]
    grid = np.zeros((h * n_rows, w * n_cols, 3), dtype=np.uint8)
    for i, t in enumerate(tiles):
        r, c = divmod(i, n_cols)
        grid[r * h:(r + 1) * h, c * w:(c + 1) * w] = t

    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), grid)
    return out


def edf_overlay(edf: np.ndarray, organoids: list[Organoid], acq, params,
                output: str | Path) -> Path:
    """All-in-focus projection with every measured outline drawn on it.

    This is the recall check. On the projection every organoid shows its crisp
    rim at the same time, so an un-outlined rim here is a genuine miss -- unlike
    on a single raw slice, where most objects are legitimately out of focus.
    """
    rgb = cv2.cvtColor(edf, cv2.COLOR_GRAY2BGR)
    theta = np.linspace(0, 2 * np.pi, params.n_theta, endpoint=False)
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    for o in organoids:
        r = np.asarray(o.radial_profile_px)
        pts = np.stack([o.x_px + r * cos_t, o.y_px + r * sin_t], axis=1)
        cv2.polylines(rgb, [pts.astype(np.int32).reshape(-1, 1, 2)], True,
                      (60, 230, 255), 2, cv2.LINE_AA)

    cv2.rectangle(rgb, (0, 0), (rgb.shape[1], 26), (20, 20, 20), -1)
    cv2.putText(rgb, f"All-in-focus projection  |  {len(organoids)} organoids",
                (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), rgb)
    return out


def focus_plot(profile: np.ndarray, substrate: int, organoids: list[Organoid],
               output: str | Path) -> Path:
    """Focus profile of the stack with the substrate plane and organoid depths."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 3.2), dpi=130)
    z = np.arange(len(profile))
    ax.plot(z, profile, color="#4dd0e1", lw=1.4, label="focus (Tenengrad)")
    ax.axvline(substrate, color="#ffb74d", ls="--", lw=1.2,
               label=f"glass surface (Z{substrate + 1})")
    if organoids:
        depths = [o.best_slice for o in organoids]
        ax.hist(depths, bins=min(40, len(profile)), alpha=.35, color="#66bb6a",
                weights=np.full(len(depths), profile.max() / max(1, len(depths)) * 4),
                label="organoid focal slices")
    ax.set_xlabel("Z slice")
    ax.set_ylabel("edge energy")
    ax.set_title("Z-stack focus profile")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=.15)
    fig.tight_layout()
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out
