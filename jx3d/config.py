"""Acquisition geometry and analysis parameters.

**Units policy.** Every measurement this package produces is primarily in
*pixels* and *slice indices*, because those are what the image files actually
contain. Micrometres are emitted only when a calibration was genuinely
recovered, and every calibrated value carries the source it came from. Nothing
is ever converted using a guessed scale: an uncalibrated stack yields pixel
measurements and says so, rather than quietly inventing a physical size.

This matters because these numbers are meant for scientific comparison. A
plausible-looking micrometre figure derived from an assumed pixel size is worse
than no figure at all.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict


UNKNOWN = "unknown"


@dataclass
class Acquisition:
    """Physical geometry of one Z-stack, with provenance for every scale.

    `px_um` / `z_um` are None until something authoritative supplies them --
    normally the Keyence .gci group file, otherwise the user on the command
    line. They are never defaulted to a plausible number.
    """

    px_um: float | None = None
    """Lateral sampling, micrometres per pixel. None = not calibrated."""

    z_um: float | None = None
    """Axial spacing, micrometres per slice. None = not calibrated."""

    px_um_source: str = UNKNOWN
    """'keyence-lens-table', 'user', or 'unknown'."""

    z_um_source: str = UNKNOWN
    """'keyence-stack-pitch', 'user', or 'unknown'."""

    objective: str = UNKNOWN
    na: float | None = None
    wavelength_um: float = 0.55
    """Mean wavelength of the transmitted light; only used to estimate the
    depth of field, and only when the stack is calibrated."""

    assumed_anisotropy: float = 1.0
    """Fallback slice-spacing / pixel-spacing ratio, used *only* when the stack
    is uncalibrated. The reconstruction needs this ratio to relate an
    organoid's lateral radius to its extent in Z; with no calibration there is
    nothing to derive it from, so it is declared here as an assumption and
    reported as one. Override with --anisotropy."""

    assumed_dof_slices: float = 3.0
    """Fallback depth of field, in slices, for an uncalibrated stack."""

    # ------------------------------------------------------------------ scale
    @property
    def calibrated(self) -> bool:
        return self.px_um is not None and self.z_um is not None

    @property
    def anisotropy(self) -> float:
        """Slice spacing divided by pixel spacing."""
        if self.calibrated:
            return self.z_um / self.px_um
        return self.assumed_anisotropy

    @property
    def anisotropy_source(self) -> str:
        return "calibration" if self.calibrated else "assumed"

    def to_um(self, value_px: float | None) -> float | None:
        """Lateral pixels -> micrometres, or None if uncalibrated."""
        if value_px is None or self.px_um is None:
            return None
        return value_px * self.px_um

    def slices_to_um(self, value_slices: float | None) -> float | None:
        if value_slices is None or self.z_um is None:
            return None
        return value_slices * self.z_um

    # ------------------------------------------------------------------ optics
    @property
    def depth_of_field_um(self) -> float | None:
        """DOF = lambda*n/NA^2 + n*e/(M*NA), detector term folded in via px_um.

        At NA 0.20 this lands around 35 um -- several slices -- which is exactly
        why the stack is not an optical section: every frame carries light from
        the whole sample, and a naive 3D threshold smears each organoid into a
        column along Z.
        """
        if self.px_um is None or not self.na:
            return None
        n = 1.0
        return self.wavelength_um * n / (self.na ** 2) + n * self.px_um / self.na

    @property
    def depth_of_field_slices(self) -> float:
        dof = self.depth_of_field_um
        if dof is None or self.z_um is None:
            return self.assumed_dof_slices
        return dof / self.z_um

    @property
    def depth_of_field_source(self) -> str:
        return "optics" if self.depth_of_field_um is not None else "assumed"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["calibrated"] = self.calibrated
        d["anisotropy"] = round(self.anisotropy, 4)
        d["anisotropy_source"] = self.anisotropy_source
        dof = self.depth_of_field_um
        d["depth_of_field_um"] = round(dof, 2) if dof is not None else None
        d["depth_of_field_slices"] = round(self.depth_of_field_slices, 2)
        d["depth_of_field_source"] = self.depth_of_field_source
        return d

    def describe_scale(self) -> str:
        if self.calibrated:
            return (f"lateral   {self.px_um:.4f} um/px  ({self.px_um_source})\n"
                    f"axial     {self.z_um:.2f} um/slice ({self.z_um_source})")
        return ("NOT CALIBRATED -- results are reported in pixels and slices.\n"
                f"assumed anisotropy {self.assumed_anisotropy:.3f} slice/px "
                f"(set --anisotropy, or --px-size/--z-step to calibrate)")


@dataclass
class Params:
    """Analysis knobs.

    Sizes are given in *pixels*, so the pipeline behaves identically whether or
    not a calibration is available. `run.py` converts micrometre-valued options
    to pixels up front when the stack is calibrated.
    """

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

    expected_diameter_px: float = 40.0
    min_diameter_px: float = 8.0
    max_diameter_px: float = 160.0
    """Objects outside this range are debris or merged clumps."""

    min_circularity: float = 0.55
    """4*pi*A/P^2. Organoids are round; stringy Matrigel texture is not."""

    z_step: int = 1
    """Segment every Nth slice, for the per-slice path."""

    cellprob_threshold: float = -2.0
    """How confident Cellpose must be that a pixel belongs to a cell.

    Lowering it below zero admits fainter objects and grows the masks it already
    has. In a Matrigel dome the faint objects are the deep ones -- there is a
    millimetre of gel above them -- so this is the knob that decides whether the
    bottom of the droplet is measured or missed. It is also the knob that
    invents objects if pushed too far, which is why it is chosen by sweeping it
    and watching the fraction of outlines that enclose nothing."""

    flow_threshold: float = 0.6
    """How far a mask's flow field may depart from a well-formed cell before it
    is rejected. Raising it keeps more irregular shapes, at the cost of keeping
    more debris. Raised from the 0.4 default because organoids in a dome are not
    all tidy spheres, and the outlines this admits are the ragged ones that
    `shape_suspect` already marks, so they arrive labelled rather than
    unnoticed."""

    # --- substrate / range ---
    substrate_margin_slices: int = 3
    """Detections at or below the glass plane are debris on the dish."""

    # --- Z linking ---
    link_max_center_shift: float = 0.6
    link_max_radius_ratio: float = 2.0
    min_track_slices: int = 2

    # --- reconstruction ---
    focus_band_px: int = 4
    """Half-width of the contour band used to measure edge sharpness."""

    n_theta: int = 48
    n_phi: int = 24
    """Angular resolution of the reconstructed spheroid surface."""

    axial_ratio: float = 1.0
    """rz / r_xy in isotropic units. 1.0 = spherical."""

    # --- dome ---
    fit_dome: bool = True
    dome_sigma: float = 28.0
    dome_threshold_k: float = 0.8
    """How far above the slice median the interface band must rise."""

    dome_x_min_frac: float = 0.0
    """Ignore the leftmost fraction of each row when tracing the interface.
    Useful when the droplet edge is known to lie on one side of the field; the
    RANSAC consensus usually makes it unnecessary."""
    """Smoothing scale for the interface ridge. Must be well above organoid
    size so the trace follows the droplet edge, not individual organoids."""

    def to_dict(self) -> dict:
        return asdict(self)
