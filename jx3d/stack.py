"""Loading a Keyence Z-stack folder."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tifffile as tiff

from .config import Acquisition
from .keyence import read_group_metadata

_Z_RE = re.compile(r"_Z(\d+)_")


@dataclass
class ZStack:
    data: np.ndarray          # (Z, Y, X) uint8
    z_indices: list[int]      # 1-based Z number printed in each filename
    files: list[str]
    acq: Acquisition
    meta: dict
    name: str

    @property
    def depth(self) -> int:
        return self.data.shape[0]

    @property
    def height(self) -> int:
        return self.data.shape[1]

    @property
    def width(self) -> int:
        return self.data.shape[2]

    def z_um(self, z: float) -> float:
        """Slice index (0-based, fractional allowed) -> depth in um."""
        return float(z) * self.acq.z_um

    def describe(self) -> str:
        a = self.acq
        lines = [f"{self.name}: {self.depth} slices  {self.width}x{self.height} px",
                 f"  objective {a.objective}" + (f" (NA {a.na})" if a.na else "")]
        for ln in a.describe_scale().splitlines():
            lines.append("  " + ln)
        if a.calibrated:
            lines.append(f"  field     {self.width * a.px_um / 1000:.2f}"
                         f" x {self.height * a.px_um / 1000:.2f} mm"
                         f"   depth {self.depth * a.z_um / 1000:.2f} mm")
        lines.append(f"  anisotropy {a.anisotropy:.2f} slice/px ({a.anisotropy_source})")
        dof = a.depth_of_field_um
        lines.append(f"  depth of field ~{a.depth_of_field_slices:.1f} slices"
                     + (f" (~{dof:.0f} um)" if dof else "")
                     + f"  [{a.depth_of_field_source}]")
        return "\n".join(lines)


def load_stack(folder: str | Path, channel: str | None = None) -> ZStack:
    """Read every *.tif in `folder`, ordered by the _Z### token in the name."""
    folder = Path(folder)
    pattern = f"*{channel}*.tif" if channel else "*.tif"
    files = sorted(folder.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No TIFF files matching {pattern!r} in {folder}")

    def z_of(p: Path) -> int:
        m = _Z_RE.search(p.name)
        return int(m.group(1)) if m else 0

    files.sort(key=z_of)
    z_indices = [z_of(p) for p in files]

    first = tiff.imread(files[0])
    if first.ndim == 3:
        first = first.mean(axis=2)
    h, w = first.shape[:2]

    data = np.empty((len(files), h, w), dtype=np.uint8)
    for i, p in enumerate(files):
        img = tiff.imread(p)
        if img.ndim == 3:
            img = img.mean(axis=2)
        if img.shape[:2] != (h, w):
            raise ValueError(f"{p.name} is {img.shape[:2]}, expected {(h, w)}")
        data[i] = img.astype(np.uint8, copy=False)

    acq, meta = read_group_metadata(folder, image_width=w)
    return ZStack(
        data=data,
        z_indices=z_indices,
        files=[str(p) for p in files],
        acq=acq,
        meta=meta,
        name=folder.name,
    )
