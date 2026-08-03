"""Acquisition geometry and analysis parameters.

Everything physical lives here so the rest of the pipeline can work in
micrometres instead of pixels/slices.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict


@dataclass
class Acquisition:
    """Physical geometry of one Z-stack.

    Defaults match the Keyence BZ-X dataset in BK52_WT_9805_B (PlanApo 4x /
    NA 0.20, 960x720 saved frames, stack pitch 100). They are overridden by
    whatever `jx3d.keyence.read_group_metadata` can recover from the .gci.
    """

    px_um: float = 3.7736
    """Lateral sampling. At 4x the BZ-X field of view is 3.62 x 2.72 mm; over a
    960 px wide frame that is 3.77 um/px."""

    z_um: float = 10.0
    """Axial step between slices. The .gci stores <Pitch>100</Pitch> in units of
    0.1 um -> 10 um. 119 slices then span 1.19 mm, which is the right order for
    a Matrigel dome."""

    objective: str = "PlanApo 4x"
    na: float = 0.20
    wavelength_um: float = 0.55
    """Mean wavelength of the brightfield transmitted light, used only to
    estimate depth of field."""

    @property
    def anisotropy(self) -> float:
        """z_um / px_um -- how much taller a voxel is than it is wide."""
        return self.z_um / self.px_um

    @property
    def depth_of_field_um(self) -> float:
        """Approximate total depth of field.

        DOF = lambda*n/NA^2 + n*e/(M*NA), with the detector term folded in via
        px_um. At NA 0.20 this lands around 50 um -- i.e. ~5 slices. This is
        precisely why the stack is NOT an optical section: every frame contains
        light from the whole dome, and a naive 3D threshold smears each organoid
        into a column along Z.
        """
        n = 1.0
        return self.wavelength_um * n / (self.na ** 2) + n * self.px_um / self.na

    def to_dict(self) -> dict:
        d = asdict(self)
        d["anisotropy"] = round(self.anisotropy, 4)
        d["depth_of_field_um"] = round(self.depth_of_field_um, 2)
        return d


@dataclass
class Params:
    """Analysis knobs. Sizes are in micrometres; the code converts to pixels."""

    # --- detection ---
    mode: str = "both"
    """Where objects are detected.

    'edf'    -- once on the all-in-focus projection. Best recall: every organoid
                shows a crisp rim there simultaneously, so the segmenter gets its
                single best look at all of them at once. One segmenter call.
    'slices' -- independently on each Z slice, then stitched across Z. Separates
                organoids that sit at the same (x, y) but different depths, which
                the projection merges into one blob.
    'both'   -- union of the two, deduplicated. Default.
    """

    detector: str = "cellpose"
    """'cellpose' (Cellpose-SAM) or 'classical' (edge/watershed fallback)."""

    edf_smooth_sigma: float = 6.0
    """Pooling width for the sharpness map before the per-pixel depth argmax.
    Roughly a rim width; too small and the depth map is noise."""

    expected_diameter_um: float = 150.0
    """Rough organoid size, used as Cellpose's diameter hint."""

    min_diameter_um: float = 30.0
    max_diameter_um: float = 600.0
    """Objects outside this range are debris or merged clumps."""

    min_circularity: float = 0.55
    """4*pi*A/P^2. Organoids are round; stringy Matrigel texture is not."""

    z_step: int = 1
    """Segment every Nth slice. 2 halves runtime with little loss, since the
    depth of field already spans ~5 slices."""

    # --- substrate / range ---
    substrate_margin_slices: int = 3
    """Detections at or below the glass plane are debris on the dish."""

    # --- Z linking ---
    link_max_center_shift: float = 0.6
    """Max centroid drift between consecutive slices, as a fraction of radius."""

    link_max_radius_ratio: float = 2.0
    """Max radius ratio between linked detections on consecutive slices."""

    min_track_slices: int = 2
    """A real organoid is visible on more than one slice. Singletons are noise."""

    # --- reconstruction ---
    focus_band_px: int = 4
    """Half-width of the contour band used to measure edge sharpness."""

    n_theta: int = 48
    n_phi: int = 24
    """Angular resolution of the reconstructed spheroid surface."""

    axial_ratio: float = 1.0
    """rz / r_xy. 1.0 = spherical. Organoids in Matrigel are near-spherical;
    lower this if yours are visibly flattened against the glass."""

    def to_dict(self) -> dict:
        return asdict(self)
