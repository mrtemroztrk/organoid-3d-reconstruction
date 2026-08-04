"""Where each tile sits in the assembled mosaic.

A tiled acquisition is fifteen separate Z-stacks that together cover one
specimen. Before anything can be measured across the whole dome, one question
has to be answered precisely: for a pixel in tile 7, where is it in the
specimen? Everything downstream -- the dome fit, the distance of an organoid to
the droplet border, the decision that two detections in two tiles are the same
organoid -- is a statement about that shared frame, and is only as good as it.

This module answers it, and records how it knows. Offsets begin as the stage
positions the microscope wrote into each frame, which is the instrument's own
account of where it moved. `jx3d.register` then refines them against the pixels
in the overlapping strips. Each tile carries the provenance of its own offset,
so a mosaic assembled from metadata alone can never be mistaken for one that was
checked against the images.

Units, as everywhere in this package, are pixels for lateral distances and slice
indices for depth. Stage coordinates are nanometres, and are converted once, on
the way in, using the calibration the instrument recorded.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .config import Acquisition
from .keyence import TileLayout, read_stage_location, read_tile_layout

STAGE = "stage-metadata"
"""Offsets taken from the per-frame stage positions, unchecked against pixels."""

REGISTERED = "phase-correlation"
"""Offsets refined by correlating the overlapping strips."""

NOMINAL = "declared-layout"
"""Offsets from an assumed regular grid, because the instrument recorded none."""


@dataclass
class Tile:
    """One Z-stack and its place in the mosaic."""

    name: str
    folder: Path
    row: int
    col: int

    x0: float
    """Left edge of this tile in mosaic pixels; column 0 sits at x0 = 0."""
    y0: float
    """Top edge of this tile in mosaic pixels; row 0 sits at y0 = 0."""

    width: int
    height: int

    stage_nm: tuple[int, int, int] | None = None
    """Raw stage position, as recorded, kept so the derivation stays auditable."""

    offset_source: str = STAGE

    @property
    def index(self) -> int:
        """The number the operator sees, parsed out of a name like 4x_00007."""
        digits = "".join(ch for ch in self.name.split("_")[-1] if ch.isdigit())
        return int(digits) if digits else 0

    def to_global(self, x_px, y_px):
        """Tile-local pixel coordinates into mosaic coordinates."""
        return np.asarray(x_px) + self.x0, np.asarray(y_px) + self.y0

    def to_local(self, x_px, y_px):
        return np.asarray(x_px) - self.x0, np.asarray(y_px) - self.y0

    def contains(self, x_px, y_px, margin: float = 0.0) -> np.ndarray:
        """Does this mosaic coordinate fall inside the tile's own frame?"""
        x = np.asarray(x_px, dtype=float)
        y = np.asarray(y_px, dtype=float)
        return ((x >= self.x0 + margin) & (x <= self.x0 + self.width - margin) &
                (y >= self.y0 + margin) & (y <= self.y0 + self.height - margin))

    def edge_distance(self, x_px, y_px) -> np.ndarray:
        """Distance from a mosaic coordinate to the nearest edge of this tile.

        Negative outside. This is what tells a detection near a seam apart from
        one safely inside the frame, and therefore which measurements of it can
        be trusted.
        """
        x = np.asarray(x_px, dtype=float)
        y = np.asarray(y_px, dtype=float)
        return np.minimum(np.minimum(x - self.x0, self.x0 + self.width - x),
                          np.minimum(y - self.y0, self.y0 + self.height - y))

    def to_dict(self) -> dict:
        return {
            "name": self.name, "index": self.index,
            "row": self.row, "col": self.col,
            "x0_px": round(self.x0, 3), "y0_px": round(self.y0, 3),
            "width_px": self.width, "height_px": self.height,
            "stage_nm": list(self.stage_nm) if self.stage_nm else None,
            "offset_source": self.offset_source,
        }


@dataclass
class Mosaic:
    """The tile grid, and the frame every measurement is finally expressed in."""

    tiles: list[Tile]
    n_rows: int
    n_cols: int
    layout_source: str
    """How the row/column assignment was established."""

    notes: list[str] = field(default_factory=list)
    """Anything a reader needs to know that the numbers do not say themselves."""

    # ------------------------------------------------------------------ shape
    @property
    def tile_width(self) -> int:
        return self.tiles[0].width

    @property
    def tile_height(self) -> int:
        return self.tiles[0].height

    @property
    def width(self) -> int:
        return int(np.ceil(max(t.x0 + t.width for t in self.tiles)))

    @property
    def height(self) -> int:
        return int(np.ceil(max(t.y0 + t.height for t in self.tiles)))

    @property
    def offset_source(self) -> str:
        sources = {t.offset_source for t in self.tiles}
        return sources.pop() if len(sources) == 1 else "mixed"

    def __len__(self) -> int:
        return len(self.tiles)

    def __iter__(self):
        return iter(self.tiles)

    # ----------------------------------------------------------------- lookup
    def by_name(self, name: str) -> Tile | None:
        return next((t for t in self.tiles if t.name == name), None)

    def at(self, row: int, col: int) -> Tile | None:
        return next((t for t in self.tiles if t.row == row and t.col == col), None)

    def neighbour_pairs(self) -> list[tuple[Tile, Tile, str]]:
        """Adjacent tiles that share an overlap, as (left/upper, right/lower, axis).

        Only the two grid directions are returned. Diagonal neighbours also
        overlap at the corners, but their shared area is the intersection of two
        already-constrained pairs and adds no independent information to the
        global solve, while being the smallest and least reliable patch to
        correlate.
        """
        pairs: list[tuple[Tile, Tile, str]] = []
        for t in self.tiles:
            right = self.at(t.row, t.col + 1)
            if right is not None:
                pairs.append((t, right, "x"))
            below = self.at(t.row + 1, t.col)
            if below is not None:
                pairs.append((t, below, "y"))
        return pairs

    def overlap_px(self) -> tuple[float, float]:
        """Median overlap between neighbours, in pixels, as (along x, along y)."""
        dx = [b.x0 - a.x0 for a, b, axis in self.neighbour_pairs() if axis == "x"]
        dy = [b.y0 - a.y0 for a, b, axis in self.neighbour_pairs() if axis == "y"]
        return (self.tile_width - float(np.median(dx)) if dx else 0.0,
                self.tile_height - float(np.median(dy)) if dy else 0.0)

    def covering(self, x_px, y_px, margin: float = 0.0) -> list[Tile]:
        """Every tile whose frame contains this mosaic coordinate."""
        return [t for t in self.tiles if bool(t.contains(x_px, y_px, margin))]

    # ----------------------------------------------------------------- report
    def describe(self, acq: Acquisition | None = None) -> str:
        ox, oy = self.overlap_px()
        lines = [
            f"{len(self.tiles)} tiles, {self.n_rows} x {self.n_cols} "
            f"({self.layout_source})",
            f"  canvas    {self.width} x {self.height} px"
            + (f"  = {self.width * acq.px_um / 1000:.2f} x "
               f"{self.height * acq.px_um / 1000:.2f} mm"
               if acq is not None and acq.px_um else ""),
            f"  tile      {self.tile_width} x {self.tile_height} px",
            f"  overlap   {ox:.1f} px ({100 * ox / self.tile_width:.1f}%) in x, "
            f"{oy:.1f} px ({100 * oy / self.tile_height:.1f}%) in y",
            f"  offsets   {self.offset_source}",
        ]
        for r in range(self.n_rows):
            row = [self.at(r, c) for c in range(self.n_cols)]
            lines.append("  " + "  ".join(
                (f"{t.index:>3d}" if t else "  -") for t in row))
        lines.extend("  ! " + n for n in self.notes)
        return "\n".join(lines)

    def to_dict(self) -> dict:
        ox, oy = self.overlap_px()
        return {
            "n_tiles": len(self.tiles),
            "n_rows": self.n_rows,
            "n_cols": self.n_cols,
            "layout_source": self.layout_source,
            "offset_source": self.offset_source,
            "canvas_width_px": self.width,
            "canvas_height_px": self.height,
            "tile_width_px": self.tile_width,
            "tile_height_px": self.tile_height,
            "overlap_x_px": round(ox, 3),
            "overlap_y_px": round(oy, 3),
            "notes": list(self.notes),
            "tiles": [t.to_dict() for t in self.tiles],
        }

    def save(self, path: str | Path) -> Path:
        out = Path(path)
        out.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return out


# --------------------------------------------------------------------------- #
# building the grid
# --------------------------------------------------------------------------- #

def _axis_offsets(stage: np.ndarray, index: np.ndarray, nm_per_px: float,
                  axis: str, notes: list[str]) -> np.ndarray:
    """Stage coordinates along one axis into mosaic pixel offsets.

    Which way the stage runs relative to the image is a property of how the
    camera is mounted, not something to assume. It is settled here by asking the
    tile map, which was written independently of the stage log: if the tiles the
    file list calls column 0 are the ones at the highest stage X, then stage X
    runs opposite to image x. Only the direction is taken from the map; the
    spacing still comes from the stage, so an irregular scan stays irregular.
    """
    origin = np.polyfit(index, stage, 1)[0] if len(set(index.tolist())) > 1 else -1.0
    descending = origin < 0
    zero = stage.max() if descending else stage.min()
    offsets = np.abs(stage - zero) / nm_per_px
    notes.append(f"stage {axis.upper()} {'decreases' if descending else 'increases'} "
                 f"as mosaic {axis} increases")
    return offsets


def build(dataset: str | Path, acq: Acquisition,
          stack_names: list[str] | None = None,
          layout: TileLayout | None = None) -> Mosaic | None:
    """Assemble the tile grid for a tiled dataset, or None if it is not one.

    `acq` supplies the lateral calibration that turns stage nanometres into
    pixels. Without it the stage log cannot be used at all, and the grid falls
    back to an assumed regular overlap, which is reported as an assumption.
    """
    dataset = Path(dataset)
    notes: list[str] = []

    if layout is None:
        layout, info = read_tile_layout(dataset)
        if layout is None:
            return None
        if "warning" in info:
            notes.append(info["warning"])

    names = [n for n in layout.cells if stack_names is None or n in stack_names]
    if not names:
        return None

    first = sorted((dataset / names[0]).glob("*.tif"))
    if not first:
        raise FileNotFoundError(f"{dataset / names[0]} holds no TIFFs")
    import tifffile as tiff
    probe = tiff.imread(first[0])
    height, width = probe.shape[:2]

    stage = {n: read_stage_location(sorted((dataset / n).glob("*.tif"))[0])
             for n in names}
    have_stage = all(v is not None for v in stage.values())

    if have_stage and acq.px_um:
        nm_per_px = acq.px_um * 1000.0
        cols = np.array([layout.cells[n][1] for n in names])
        rows = np.array([layout.cells[n][0] for n in names])
        sx = np.array([stage[n][0] for n in names], dtype=float)
        sy = np.array([stage[n][1] for n in names], dtype=float)
        xs = _axis_offsets(sx, cols, nm_per_px, "x", notes)
        ys = _axis_offsets(sy, rows, nm_per_px, "y", notes)
        source = STAGE

        depths = {stage[n][2] for n in names}
        if len(depths) == 1:
            notes.append("every tile was captured at the same stage Z, so the "
                         "slice index means the same depth in all of them")
        else:
            span = (max(depths) - min(depths)) / 1000.0
            notes.append(f"stage Z differs between tiles by up to {span:.1f} um "
                         f"- depth is NOT directly comparable across tiles")
    else:
        # No usable stage log. Fall back to the overlap a BZ-X scan is normally
        # set up with, and say so: this is the one path where the geometry is
        # assumed rather than recorded.
        notes.append("no usable stage log; assuming a regular 30% overlap")
        xs = np.array([layout.cells[n][1] * width * 0.70 for n in names])
        ys = np.array([layout.cells[n][0] * height * 0.70 for n in names])
        source = NOMINAL

    tiles = [
        Tile(name=n, folder=dataset / n,
             row=layout.cells[n][0], col=layout.cells[n][1],
             x0=float(x), y0=float(y), width=int(width), height=int(height),
             stage_nm=stage[n], offset_source=source)
        for n, x, y in zip(names, xs, ys)
    ]
    tiles.sort(key=lambda t: (t.row, t.col))
    return Mosaic(tiles=tiles, n_rows=layout.n_rows, n_cols=layout.n_cols,
                  layout_source=layout.source, notes=notes)
