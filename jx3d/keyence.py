"""Read acquisition geometry out of a Keyence BZ-X .gci group file.

A .gci is a plain zip of small properties.xml documents. We only need the
Z-stack pitch and the objective, but the whole tree is dumped for reference.
"""
from __future__ import annotations

import re
import zipfile
from pathlib import Path

from .config import Acquisition

# Lateral sampling of a saved 960 px wide BZ-X frame, per objective.
# Derived from the documented field of view (4x -> 3.62 x 2.72 mm).
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

    # --- objective: Magnification is stored x100 (400 -> 4x) ---
    mag_raw = _xml_value(lens_xml, "Magnification")
    lens_name = _xml_value(lens_xml, "LensName")
    if lens_name:
        acq.objective = lens_name.split(":")[0].strip()
        info["lens_name"] = lens_name
    if mag_raw is not None:
        mag = int(mag_raw) // 100
        info["magnification"] = mag
        if mag in _PX_UM_960:
            # The .gci records the objective but not the pixel size, so this
            # comes from Keyence's documented BZ-X field of view for that
            # objective. It is a lookup, not a measurement -- hence the explicit
            # source tag on the result.
            acq.px_um = _PX_UM_960[mag] * (960.0 / image_width)
            acq.px_um_source = "keyence-lens-table"
        else:
            info["warning"] = (f"objective {mag}x not in the calibration table; "
                               f"lateral scale unknown")

    # NA is stored as a raw IEEE-754 bit pattern in an Int64 field, so it is not
    # worth decoding; infer it from the objective name instead.
    if lens_name:
        m = re.search(r"(\d+(?:\.\d+)?)\s*x\s+(\d*\.\d+)", lens_name)
        if m:
            acq.na = float(m.group(2))

    return acq, info
