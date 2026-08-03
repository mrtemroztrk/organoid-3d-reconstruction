"""Read acquisition geometry out of a Keyence BZ-X .gci group file.

A .gci is a plain zip of small properties.xml documents. We only need the
Z-stack pitch and the objective, but the whole tree is dumped for reference.
"""
from __future__ import annotations

import re
import struct
import zipfile
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
