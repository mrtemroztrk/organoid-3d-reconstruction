"""Read acquisition geometry out of a Keyence BZ-X dataset.

Two files carry it. The .gci group file is a plain zip of small properties.xml
documents describing the whole acquisition -- objective, Z pitch, and, for a
tiled scan, the shape of the tile grid and which image belongs in which cell.
Each individual TIFF then carries its own copy of the state the stage was in
when that frame was taken, hidden in a Keyence-private EXIF MakerNote.

Everything here reads the instrument's own record. Where a value cannot be
found the caller is told, so that a missing calibration or a missing tile map
becomes a stated fact rather than a silent default.
"""
from __future__ import annotations

import re
import struct
import zipfile
from dataclasses import dataclass
from pathlib import Path

from .config import Acquisition

def _decode_double(raw: str | None) -> float | None:
    """Keyence stores System.Double fields as the raw IEEE-754 bit pattern in an
    Int64. Verified against two fields whose values are also written in plain
    text in the lens name: NumericalAperture decodes to 0.2 and WorkingDistance
    to 20.0, matching "PlanApo 4x 0.20/20.00mm".
    """
    if raw is None:
        return None
    try:
        return struct.unpack("<d", struct.pack("<q", int(raw)))[0]
    except (ValueError, struct.error):
        return None


# Fallback only. The instrument writes its own lateral calibration into the
# .gci; this table (from the documented BZ-X field of view for a 960 px frame)
# is used when that field is missing.
_PX_UM_960 = {
    2: 7.5472,
    4: 3.7736,
    10: 1.5094,
    20: 0.7547,
    40: 0.3774,
    60: 0.2516,
    100: 0.1509,
}


def _find_gci(folder: Path) -> Path | None:
    hits = sorted(folder.glob("*.gci")) + sorted(folder.parent.glob("*.gci"))
    return hits[0] if hits else None


def _xml_value(text: str, tag: str) -> str | None:
    m = re.search(rf"<{tag}\b[^>]*>(.*?)</{tag}>", text, re.S)
    return m.group(1) if m else None


def read_group_metadata(stack_folder: str | Path, image_width: int = 960) -> tuple[Acquisition, dict]:
    """Return (Acquisition, raw_info). Falls back to defaults when no .gci."""
    folder = Path(stack_folder)
    acq = Acquisition()
    info: dict = {"source": None}

    gci = _find_gci(folder)
    if gci is None:
        info["warning"] = ("no .gci found - stack is uncalibrated, results will "
                           "be reported in pixels and slices")
        return acq, info
    info["source"] = str(gci)

    with zipfile.ZipFile(gci) as zf:
        def read(name: str) -> str:
            try:
                return zf.read(name).decode("utf-8", "replace")
            except KeyError:
                return ""

        stack_xml = read("GroupFileProperty/Stack/properties.xml")
        lens_xml = read("GroupFileProperty/Lens/properties.xml")
        image_xml = read("GroupFileProperty/Image/properties.xml")

    # --- Z pitch: stored in units of 0.1 um ---
    pitch = _xml_value(stack_xml, "Pitch")
    total = _xml_value(stack_xml, "TotalNumber")
    if pitch is not None:
        # Keyence stores the stack pitch as an integer in units of 0.1 um.
        acq.z_um = int(pitch) / 10.0
        acq.z_um_source = "keyence-stack-pitch"
        info["stack_pitch_raw"] = int(pitch)
    if total is not None:
        info["stack_total"] = int(total)

    # --- lateral scale, as recorded by the instrument ---
    # <Calibration> is nanometres per pixel. This is a value the microscope
    # wrote, not something inferred from the objective, so it is preferred over
    # the lookup table below.
    calib_nm = _decode_double(_xml_value(image_xml, "Calibration"))
    if calib_nm and 1.0 < calib_nm < 1e6:
        acq.px_um = calib_nm / 1000.0
        acq.px_um_source = "keyence-calibration"
        info["calibration_nm_per_px"] = calib_nm

    # --- objective: Magnification is stored x100 (400 -> 4x) ---
    mag_raw = _xml_value(lens_xml, "Magnification")
    lens_name = _xml_value(lens_xml, "LensName")
    if lens_name:
        acq.objective = lens_name.split(":")[0].strip()
        info["lens_name"] = lens_name
    if mag_raw is not None:
        mag = int(mag_raw) // 100
        info["magnification"] = mag
        table_px = (_PX_UM_960[mag] * (960.0 / image_width)
                    if mag in _PX_UM_960 else None)
        if acq.px_um is None and table_px is not None:
            acq.px_um = table_px
            acq.px_um_source = "keyence-lens-table"
        elif acq.px_um is not None and table_px is not None:
            # Both available: record how far apart they are. A large
            # disagreement means the frame was cropped or binned and the table
            # no longer applies.
            info["lens_table_px_um"] = round(table_px, 5)
            info["calibration_vs_table_pct"] = round(
                100.0 * abs(acq.px_um - table_px) / table_px, 3)
        elif acq.px_um is None:
            info["warning"] = (f"objective {mag}x not in the calibration table "
                               f"and no <Calibration> field; lateral scale unknown")

    na = _decode_double(_xml_value(lens_xml, "NumericalAperture"))
    if na and 0.01 < na < 2.0:
        acq.na = na
    elif lens_name:
        m = re.search(r"(\d+(?:\.\d+)?)\s*x\s+(\d*\.\d+)", lens_name)
        if m:
            acq.na = float(m.group(2))
    wd = _decode_double(_xml_value(lens_xml, "WorkingDistance"))
    if wd:
        info["working_distance_mm"] = wd

    return acq, info


# --------------------------------------------------------------------------- #
# Tiled ("image joint") acquisitions
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class TileLayout:
    """Which captured stack belongs in which cell of a tiled scan.

    `cells` maps a stack folder name such as ``4x_00007`` to its ``(row, col)``,
    with row 0 at the top and column 0 at the left of the assembled mosaic.
    """

    n_rows: int
    n_cols: int
    cells: dict[str, tuple[int, int]]
    source: str
    """Where the layout came from: 'keyence-filelist' or 'keyence-imagejoint'."""

    def name_at(self, row: int, col: int) -> str | None:
        for name, rc in self.cells.items():
            if rc == (row, col):
                return name
        return None

    def describe(self) -> str:
        lines = [f"{self.n_rows} x {self.n_cols} tiles ({self.source})"]
        for r in range(self.n_rows):
            lines.append("  " + "  ".join(
                (self.name_at(r, c) or "--").rjust(9) for c in range(self.n_cols)))
        return "\n".join(lines)


def _read_7bit_length(buf: bytes, off: int) -> tuple[int, int]:
    """Decode a .NET BinaryWriter length prefix; returns (length, new offset).

    Lengths below 128 occupy a single byte, which is every case in practice
    here, but the format is variable-width and reading only the first byte
    would desynchronise the whole record stream on a longer name.
    """
    value = 0
    shift = 0
    while True:
        b = buf[off]
        off += 1
        value |= (b & 0x7F) << shift
        if not b & 0x80:
            return value, off
        shift += 7


def _parse_file_list(blob: bytes) -> dict[str, tuple[int, int]]:
    """Decode GroupFileProperty/ImageList/FileList into {stack name: (row, col)}.

    The blob is a .NET-serialised record array: a 32-bit count, then one record
    per captured image holding the channel name, four 32-bit indices, the file
    name, and a double. Only the third and fourth indices are needed; they are
    the tile's row and column.

    That reading is not inferred from the field order alone. It reproduces both
    the operator's own sketch of the scan and, independently, the ordering of
    the per-image stage coordinates: the tile the file list calls column 1 is
    the tile whose stage X lies between the other two.
    """
    n_records = struct.unpack_from("<I", blob, 0)[0]
    off = 4
    cells: dict[str, tuple[int, int]] = {}
    for _ in range(n_records):
        length, off = _read_7bit_length(blob, off)
        off += length                                   # channel name
        _, _, row, col = struct.unpack_from("<4i", blob, off)
        off += 16
        length, off = _read_7bit_length(blob, off)
        name = blob[off:off + length].decode("utf-8", "replace")
        off += length + 8                               # name, then the double
        cells.setdefault(name.split("_Z")[0], (int(row), int(col)))
    return cells


def read_tile_layout(folder: str | Path) -> tuple[TileLayout | None, dict]:
    """The tile grid of a stitched acquisition, or None if it was not tiled.

    `folder` may be the dataset directory or any stack folder inside it.
    """
    info: dict = {}
    gci = _find_gci(Path(folder))
    if gci is None:
        info["warning"] = "no .gci found - the tile layout is unknown"
        return None, info
    info["source"] = str(gci)

    with zipfile.ZipFile(gci) as zf:
        def read(name: str) -> bytes:
            try:
                return zf.read(name)
            except KeyError:
                return b""

        joint = read("GroupFileProperty/ImageJoint/properties.xml").decode(
            "utf-8", "replace")
        file_list = read("GroupFileProperty/ImageList/FileList")

    if not joint or (_xml_value(joint, "Enabled") or "").strip() != "True":
        info["warning"] = "the .gci does not describe a tiled acquisition"
        return None, info

    rows = _xml_value(joint, "Row")
    cols = _xml_value(joint, "Column")
    if rows is None or cols is None:
        info["warning"] = "ImageJoint carries no Row/Column"
        return None, info
    n_rows, n_cols = int(rows), int(cols)
    info["image_joint_rows"] = n_rows
    info["image_joint_columns"] = n_cols
    info["stitching_type"] = _xml_value(joint, "OperationType")

    if not file_list:
        info["warning"] = ("ImageJoint declares the grid shape but the file "
                           "list is missing, so which stack sits in which cell "
                           "is unknown")
        return None, info

    cells = _parse_file_list(file_list)
    if len(cells) != n_rows * n_cols:
        info["warning"] = (f"the file list names {len(cells)} stacks but the "
                           f"grid is {n_rows}x{n_cols}")
    return TileLayout(n_rows, n_cols, cells, "keyence-filelist"), info


# --------------------------------------------------------------------------- #
# Per-image stage position
# --------------------------------------------------------------------------- #

_MAKERNOTE_XML_TAG = 0x0800
"""MakerNote entry holding a UTF-8 properties document for that single frame."""


def _makernote_xml(path: Path) -> str | None:
    """The per-frame properties XML embedded in one TIFF, if it is there.

    The MakerNote is a little-endian TIFF IFD of its own, but its value offsets
    are relative to the start of the *file*, not to the note. Resolving them
    against the note would read whatever happens to lie at that offset inside
    it, which is why the file is reopened here rather than the note alone being
    parsed.
    """
    import tifffile as tiff

    with tiff.TiffFile(path) as tf:
        exif = tf.pages[0].tags.get("ExifTag")
        note = (exif.value or {}).get("MakerNote") if exif is not None else None
    if not note or not note.startswith(b"KmsFile"):
        return None

    (n_entries,) = struct.unpack_from("<H", note, 8)
    for i in range(n_entries):
        base = 10 + i * 12
        tag, _, count = struct.unpack_from("<HHI", note, base)
        if tag != _MAKERNOTE_XML_TAG:
            continue
        (offset,) = struct.unpack_from("<I", note, base + 8)
        with open(path, "rb") as fh:
            fh.seek(offset)
            blob = fh.read(count)
        start = blob.find(b"<?xml")
        if start < 0:
            return None
        return blob[start:].decode("utf-8", "replace")
    return None


def read_stage_location(tif_path: str | Path) -> tuple[int, int, int] | None:
    """Stage position when this frame was captured, as (X, Y, Z) nanometres.

    This is the only place the relative position of one tile to another is
    recorded, and it is recorded per frame rather than per stack, so it is read
    from an image rather than from the group file.
    """
    xml = _makernote_xml(Path(tif_path))
    if xml is None:
        return None
    got = []
    for axis in ("X", "Y", "Z"):
        raw = _xml_value(xml, f"StageLocation{axis}")
        if raw is None:
            return None
        got.append(int(raw))
    return tuple(got)                                    # type: ignore[return-value]


def read_edge_points(folder: str | Path) -> list[tuple[int, int, int]]:
    """Stage points the operator drove to when setting up the scan.

    These are the requested bounds of the tiled region, expressed as where the
    outermost tiles should be *centred*, and the instrument then rounds the grid
    up to whole tiles. On this dataset that reading checks out to within a
    pixel: the first tile centre lands 0.4 px from the first recorded point, and
    the last overshoots by 17 px across and 422 px down, which is exactly the
    slack from rounding 1.97 columns up to 3 and 3.17 rows up to 5.

    It is tempting to treat them as the operator's opinion of where the droplet
    ends, and so as an independent check on a fitted dome. They are not, and the
    numbers say so plainly: they sit a millimetre inside the fitted contact
    circle, and their stage Z is the depth of slice 60 rather than of the glass.
    They describe the region someone asked the microscope to cover, nothing more.

    Unused slots are stored as all-zero and are dropped here.
    """
    gci = _find_gci(Path(folder))
    if gci is None:
        return []
    points: list[tuple[int, int, int]] = []
    with zipfile.ZipFile(gci) as zf:
        names = set(zf.namelist())
        for i in range(32):
            name = f"GroupFileProperty/ImageJoint/EdgePoint{i}/properties.xml"
            if name not in names:
                break
            xml = zf.read(name).decode("utf-8", "replace")
            xyz = tuple(int(_xml_value(xml, a) or 0) for a in ("X", "Y", "Z"))
            if any(xyz):
                points.append(xyz)                       # type: ignore[arg-type]
    return points
