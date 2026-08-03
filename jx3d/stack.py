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
        return (
            f"{self.name}: {self.depth} slices  {self.width}x{self.height} px\n"
            f"  objective {a.objective} (NA {a.na})\n"
            f"  lateral   {a.px_um:.4f} um/px  -> field {self.width * a.px_um / 1000:.2f}"
            f" x {self.height * a.px_um / 1000:.2f} mm\n"
            f"  axial     {a.z_um:.2f} um/slice -> depth {self.depth * a.z_um / 1000:.2f} mm"
            f"  (anisotropy {a.anisotropy:.2f}x)\n"
            f"  depth of field ~{a.depth_of_field_um:.0f} um "
            f"(~{a.depth_of_field_um / a.z_um:.1f} slices)"
        )


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
