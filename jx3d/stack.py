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


SKIP_DIRS = {".git", ".venv", "venv", "output", "docs", "__pycache__",
             "node_modules", "site-packages"}

DEFAULT_HINT = "4x_00009"
"""Which stack `run.py` reaches for when no folder is given.

The dataset this package was developed against has fifteen positions in one
folder; 4x_00009 is the one with organoids spread through the full depth of the
dome, so it is the useful default to open. Any other folder can be passed
explicitly, or picked from the control panel.
"""


def is_stack_dir(d: Path) -> bool:
    """Does this folder hold a Z-stack? (at least a few *_Z###_*.tif files)"""
    if not d.is_dir():
        return False
    n = 0
    for p in d.glob("*.tif"):
        if _Z_RE.search(p.name):
            n += 1
            if n >= 3:
                return True
    return False


def discover_stacks(root: str | Path = ".", max_depth: int = 3) -> list[Path]:
    """Every Z-stack folder at or below `root`, breadth-first, sorted by name."""
    root = Path(root)
    found: list[Path] = []
    frontier = [(root, 0)]
    while frontier:
        d, depth = frontier.pop(0)
        if is_stack_dir(d):
            found.append(d)
            continue                     # a stack has no stacks inside it
        if depth >= max_depth:
            continue
        try:
            kids = sorted(p for p in d.iterdir() if p.is_dir())
        except OSError:
            continue
        for k in kids:
            if k.name.startswith(".") or k.name in SKIP_DIRS:
                continue
            frontier.append((k, depth + 1))
    return sorted(found, key=lambda p: p.name)


def default_stack(root: str | Path = ".", hint: str = DEFAULT_HINT) -> Path | None:
    """The stack to open when the user did not name one.

    Prefers a folder matching `hint`; otherwise the first one found, so the
    package still does something sensible on a dataset it has never seen.
    """
    stacks = discover_stacks(root)
    if not stacks:
        return None
    for p in stacks:
        if hint in p.name:
            return p
    return stacks[0]


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
