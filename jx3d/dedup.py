"""One row per organoid, from fifteen overlapping views of the same droplet.

Nearly half the mosaic is seen by more than one tile -- 51.3 % of the canvas by
exactly one, 40.8 % by two and 7.9 % by four -- so an organoid in an overlap is
detected twice, or at a corner up to four times. Counting it once is not a
tidying step. It is the difference between a catalogue of organoids and a
catalogue of sightings, and with this geometry roughly half of all raw
detections are repeats.

Two decisions shape everything here.

**Detection happens on each tile's own pixels, and merging happens afterwards in
mosaic coordinates.** The alternative -- blend the tiles into one image and
segment that once -- fails on the thing this project is for. In an overlap an
organoid sits near the right edge of one frame and the left edge of the next, on
opposite sides of the illumination field, so a blend gives it pixels no camera
recorded, weighted differently across the object. That is a position-dependent
bias injected into exactly the texture numbers meant to carry the viability
signal, and it cannot be checked against any raw image, because there is none it
came from.

**When several views survive, one is elected rather than averaged.** Averaging
two outlines produces a shape neither camera saw, and averaging texture across
two tiles imaged at different stage positions mixes the very thing being
measured. The unelected views are kept instead of discarded, which turns the
overlap into something better than a nuisance: the same organoid measured twice,
independently, is a free repeatability estimate for every feature -- an error
bar that a single field of view simply cannot produce.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import Acquisition
from .mosaic import Mosaic, Tile


@dataclass
class Sighting:
    """One tile's measurement of one organoid, in mosaic coordinates."""

    tile: str
    row: int
    col: int
    local_oid: int

    x_px: float
    y_px: float
    z_slice: float
    radius_px: float
    focus_sharpness: float
    circularity: float

    edge_distance_px: float
    """How far this detection sat from its own tile's frame edge. Negative or
    small means the outline was cut off and every shape measurement from it is
    a measurement of the cut, not of the organoid."""

    clipped: bool = False
    source: dict = field(default_factory=dict)
    """The full v1 measurement, carried through untouched."""

    @property
    def key(self) -> tuple[str, int]:
        return (self.tile, self.local_oid)


@dataclass
class Organoid:
    """One physical organoid, and every view anyone had of it."""

    uid: int
    elected: Sighting
    views: list[Sighting]

    coverage_k: int
    """How many tiles' frames contain this position -- not how many saw it.
    Reported because the outer rim of the mosaic is singly covered and can never
    be cross-confirmed, so any analysis wanting an unbiased subsample has to be
    able to ask for k >= 2."""

    n_views: int = 1
    lateral_spread_px: float = 0.0
    axial_spread_slices: float = 0.0
    radius_ratio: float = 1.0

    single_view_in_overlap: bool = False
    """Inside a doubly-covered region but found by only one tile. A real object
    the other tile's segmenter missed, not a merge failure -- and never a reason
    to drop it."""
    clipped_everywhere: bool = False
    radius_disagreement: bool = False
    same_tile_twice: bool = False
    guarantee_void: bool = False
    """Larger than half the overlap, so no tile is guaranteed to hold it whole."""

    def to_dict(self) -> dict:
        d = dict(self.elected.source)
        d.update({
            "uid": self.uid,
            "tile": self.elected.tile,
            "tile_row": self.elected.row,
            "tile_col": self.elected.col,
            "x_mosaic_px": round(self.elected.x_px, 3),
            "y_mosaic_px": round(self.elected.y_px, 3),
            "n_views": self.n_views,
            "views": ";".join(sorted(v.tile for v in self.views)),
            "coverage_k": self.coverage_k,
            "lateral_spread_px": round(self.lateral_spread_px, 3),
            "axial_spread_slices": round(self.axial_spread_slices, 3),
            "radius_ratio": round(self.radius_ratio, 4),
            "clipped": bool(self.elected.clipped),
            "clipped_everywhere": self.clipped_everywhere,
            "single_view_in_overlap": self.single_view_in_overlap,
            "radius_disagreement": self.radius_disagreement,
            "same_tile_twice": self.same_tile_twice,
            "guarantee_void": self.guarantee_void,
        })
        return d


@dataclass
class MergeReport:
    """What the merge did, in enough detail to argue with."""

    n_sightings: int
    n_organoids: int
    n_merged: int
    pair_match_rate: dict[str, float] = field(default_factory=dict)
    pair_residual_px: dict[str, float] = field(default_factory=dict)
    pair_shared: dict[str, int] = field(default_factory=dict)
    """Detections lying in each seam's shared region. A seam with almost none
    has nothing to say about whether the tiles are placed correctly, and must
    not be allowed to vote on it."""
    lateral_px: tuple[float, float, float] = (0.0, 0.0, 0.0)
    axial_slices: tuple[float, float, float] = (0.0, 0.0, 0.0)
    lateral_gate_px: float = 0.0
    axial_gate_slices: float = 0.0
    notes: list[str] = field(default_factory=list)

    @property
    def duplicate_fraction(self) -> float:
        return self.n_merged / max(1, self.n_sightings)

    min_shared_to_judge: int = 6
    """Detections a seam needs before its match rate means anything."""

    def informative_seams(self) -> dict[str, float]:
        """Seams carrying enough detections for their match rate to be evidence.

        Three of the fifteen fields here lie beyond the edge of the droplet and
        contain almost nothing, so the seams touching them match nothing --
        not because the tiles are misplaced but because there was nothing there
        to match. Counting those as failures would condemn a perfectly good
        mosaic, so a seam has to carry some detections before it gets a vote.
        """
        return {k: v for k, v in self.pair_match_rate.items()
                if self.pair_shared.get(k, 0) >= self.min_shared_to_judge}

    @property
    def reliable(self) -> bool:
        """Were the tiles placed well enough for the merge to mean anything?

        The evidence is the residual of the matches, not how many were made. If
        two tiles were misplaced, the pairs that did match would match badly, so
        a small residual is what says the frame is sound.

        How *many* matched is a different question with a different answer, and
        conflating the two would raise a false alarm here. The seam between the
        two central fields matches only a third of its detections while placing
        those matches to half a pixel: the tiles are in the right place and the
        segmenter simply found different objects on either side of it, which is
        what happens under the thickest gel where contrast is worst. Those
        objects are real, they are kept, and they are flagged
        `single_view_in_overlap` -- they are not unmerged duplicates.
        """
        judged = self.informative_seams()
        if not judged:
            return True
        return max(self.pair_residual_px[k] for k in judged) <= 4.0

    @property
    def recall_asymmetry(self) -> list[str]:
        """Populated seams where the two tiles largely disagreed on what was there.

        Reported rather than treated as a fault. It bounds how much of the
        catalogue rests on a single segmentation, which a downstream model has a
        right to know.
        """
        return sorted(k for k, v in self.informative_seams().items() if v < 0.50)

    def describe(self) -> str:
        lines = [
            f"{self.n_sightings} sightings -> {self.n_organoids} organoids "
            f"({self.n_merged} merged away, {100 * self.duplicate_fraction:.0f}% "
            f"of raw detections were repeats)",
            f"  agreement between two views of one organoid: "
            f"lateral median {self.lateral_px[0]:.2f} px "
            f"(p90 {self.lateral_px[1]:.2f}, worst {self.lateral_px[2]:.2f}), "
            f"axial median {self.axial_slices[0]:.2f} slices "
            f"(p90 {self.axial_slices[1]:.2f}, worst {self.axial_slices[2]:.2f})",
            f"  gates: {self.lateral_gate_px:.1f} px lateral, "
            f"{self.axial_gate_slices:.2f} slices axial",
        ]
        judged = self.informative_seams()
        skipped = len(self.pair_match_rate) - len(judged)
        if judged:
            worst = min(judged, key=lambda k: judged[k])
            lines.append(f"  weakest populated seam {worst}: "
                         f"{100 * judged[worst]:.0f}% of its "
                         f"{self.pair_shared.get(worst, 0)} shared detections "
                         f"matched")
        if judged:
            worst_res = max(self.pair_residual_px[k] for k in judged)
            lines.append(f"  where two tiles agreed an object was there, they "
                         f"placed it to within {worst_res:.2f} px at every seam, "
                         f"which is what says the tiles are in the right place")
        if skipped:
            lines.append(f"  {skipped} seam(s) had too few detections to judge "
                         f"- they lie beyond the edge of the specimen, so "
                         f"matching nothing there means nothing")
        for seam in self.recall_asymmetry:
            lines.append(f"  seam {seam} matched only "
                         f"{100 * self.pair_match_rate[seam]:.0f}% of its "
                         f"detections: the two fields found different objects "
                         f"there, so more of that region rests on one "
                         f"segmentation than on two")
        if not self.reliable:
            lines.append("  !! matched objects disagree on where they are; the "
                         "tiles are not placed well enough to merge on")
        lines.extend("  " + n for n in self.notes)
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "n_sightings": self.n_sightings,
            "n_organoids": self.n_organoids,
            "n_merged": self.n_merged,
            "duplicate_fraction": round(self.duplicate_fraction, 4),
            "lateral_agreement_px": [round(v, 4) for v in self.lateral_px],
            "axial_agreement_slices": [round(v, 4) for v in self.axial_slices],
            "lateral_gate_px": round(self.lateral_gate_px, 3),
            "axial_gate_slices": round(self.axial_gate_slices, 3),
            "pair_match_rate": {k: round(v, 4) for k, v in self.pair_match_rate.items()},
            "pair_residual_px": {k: round(v, 4) for k, v in self.pair_residual_px.items()},
            "pair_shared": dict(self.pair_shared),
            "informative_seams": len(self.informative_seams()),
            "recall_asymmetry_seams": self.recall_asymmetry,
            "reliable": self.reliable,
            "notes": list(self.notes),
        }


# --------------------------------------------------------------------------- #
# building sightings
# --------------------------------------------------------------------------- #

def sightings_from_tile(tile: Tile, records: list[dict],
                        clip_margin_px: float = 2.0) -> list[Sighting]:
    """Lift one tile's v1 measurements into mosaic coordinates.

    Clipping is decided from geometry -- how far the outline reaches towards the
    frame edge -- and never from circularity. A disc cut in half still scores
    0.72 for circularity and sails through the 0.55 shape filter, so a shape
    test cannot see clipping at all; it only sees a slightly rounder object.
    """
    out: list[Sighting] = []
    for rec in records:
        r_max = max(rec.get("radial_profile_px") or [rec["radius_px"]])
        gx = float(rec["x_px"]) + tile.x0
        gy = float(rec["y_px"]) + tile.y0
        reach = float(tile.edge_distance(gx, gy))
        out.append(Sighting(
            tile=tile.name, row=tile.row, col=tile.col,
            local_oid=int(rec["oid"]),
            x_px=gx, y_px=gy, z_slice=float(rec["z_slice"]),
            radius_px=float(rec["radius_px"]),
            focus_sharpness=float(rec.get("focus_sharpness", 0.0)),
            circularity=float(rec.get("circularity", 0.0)),
            edge_distance_px=reach,
            clipped=reach < r_max + clip_margin_px,
            source=rec,
        ))
    return out


# --------------------------------------------------------------------------- #
# matching
# --------------------------------------------------------------------------- #

def _gate(a: Sighting, b: Sighting, lateral_floor: float, lateral_frac: float,
          axial_slices: float) -> float | None:
    """Cost of calling two sightings the same organoid, or None if they cannot be.

    The axial gate is deliberately tight, and tighter than the intra-tile merge
    in `reconstruct`. There, a generous axial window exists to reunite one
    organoid whose focus curve split into two humps. Here it would do real
    damage: two different organoids at the same (x, y) and different depths are
    an ordinary configuration in a droplet more than a millimetre deep, and a
    loose gate would fuse them into one and delete a real object.

    Two independent focus sweeps of the same rim cannot be expected to agree
    better than the depth over which that rim stays sharp, so the gate is one
    depth of field and no more.
    """
    lateral = float(np.hypot(a.x_px - b.x_px, a.y_px - b.y_px))
    limit = lateral_floor + lateral_frac * max(a.radius_px, b.radius_px)
    if lateral > limit:
        return None
    if abs(a.z_slice - b.z_slice) > axial_slices:
        return None
    ratio = max(a.radius_px, b.radius_px) / max(min(a.radius_px, b.radius_px), 1e-6)
    # Radius guards the match, it does not make it. Refusing a good lateral and
    # axial match because the two outlines disagree in size would produce a
    # double count, which is the error being eliminated here; only a difference
    # too large to be one object at all is disqualifying.
    if ratio > 2.0 and not (a.clipped or b.clipped):
        return None
    return lateral / max(limit, 1e-6) + 0.5 * float(np.log(ratio))


def _assign(left: list[Sighting], right: list[Sighting], lateral_floor: float,
            lateral_frac: float, axial_slices: float) -> list[tuple[int, int, float]]:
    """Best one-to-one pairing between two tiles' sightings in their overlap.

    One-to-one matters. Greedy nearest-neighbour will happily attach two
    detections in one tile to a single detection in the other, which is how a
    cluster of neighbouring organoids collapses into one.
    """
    if not left or not right:
        return []
    big = 1e6
    cost = np.full((len(left), len(right)), big)
    for i, a in enumerate(left):
        for j, b in enumerate(right):
            c = _gate(a, b, lateral_floor, lateral_frac, axial_slices)
            if c is not None:
                cost[i, j] = c

    from scipy.optimize import linear_sum_assignment

    rows, cols = linear_sum_assignment(cost)
    return [(int(i), int(j), float(cost[i, j]))
            for i, j in zip(rows, cols) if cost[i, j] < big]


class _Union:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, i: int) -> int:
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, i: int, j: int) -> None:
        a, b = self.find(i), self.find(j)
        if a != b:
            self.parent[b] = a


def merge(mosaic: Mosaic, per_tile: dict[str, list[dict]], acq: Acquisition,
          lateral_floor_px: float = 4.0, lateral_frac: float = 0.15,
          axial_dof_multiple: float = 1.0) -> tuple[list[Organoid], MergeReport]:
    """Fifteen lists of per-tile measurements into one catalogue of organoids."""
    sightings: list[Sighting] = []
    for tile in mosaic:
        sightings.extend(sightings_from_tile(tile, per_tile.get(tile.name, [])))

    axial_gate = axial_dof_multiple * acq.depth_of_field_slices
    index = {s.key: i for i, s in enumerate(sightings)}
    union = _Union(len(sightings))

    matched_pairs: list[tuple[Sighting, Sighting]] = []
    match_rate: dict[str, float] = {}
    residual: dict[str, float] = {}
    shared: dict[str, int] = {}

    for a, b, _axis in mosaic.neighbour_pairs():
        # Only sightings inside the shared region can possibly correspond, and
        # restricting to it keeps the match rate an honest denominator.
        left = [s for s in sightings if s.tile == a.name and bool(b.contains(s.x_px, s.y_px))]
        right = [s for s in sightings if s.tile == b.name and bool(a.contains(s.x_px, s.y_px))]
        pairs = _assign(left, right, lateral_floor_px, lateral_frac, axial_gate)
        for i, j, _cost in pairs:
            union.union(index[left[i].key], index[right[j].key])
            matched_pairs.append((left[i], right[j]))
        label = f"{a.index}-{b.index}"
        denom = len(left) + len(right)
        shared[label] = denom
        match_rate[label] = (2.0 * len(pairs) / denom) if denom else 1.0
        residual[label] = (float(np.median([np.hypot(left[i].x_px - right[j].x_px,
                                                     left[i].y_px - right[j].y_px)
                                            for i, j, _ in pairs])) if pairs else 0.0)

    groups: dict[int, list[Sighting]] = {}
    for i, s in enumerate(sightings):
        groups.setdefault(union.find(i), []).append(s)

    organoids: list[Organoid] = []
    for members in groups.values():
        organoids.append(_elect(members, mosaic, lateral_floor_px, lateral_frac))
    organoids.sort(key=lambda o: (o.elected.y_px, o.elected.x_px))
    for n, o in enumerate(organoids, start=1):
        o.uid = n

    lat = np.array([np.hypot(a.x_px - b.x_px, a.y_px - b.y_px)
                    for a, b in matched_pairs]) if matched_pairs else np.zeros(1)
    ax = np.array([abs(a.z_slice - b.z_slice)
                   for a, b in matched_pairs]) if matched_pairs else np.zeros(1)

    report = MergeReport(
        n_sightings=len(sightings),
        n_organoids=len(organoids),
        n_merged=len(sightings) - len(organoids),
        pair_match_rate=match_rate,
        pair_residual_px=residual,
        pair_shared=shared,
        lateral_px=(float(np.median(lat)), float(np.percentile(lat, 90)), float(lat.max())),
        axial_slices=(float(np.median(ax)), float(np.percentile(ax, 90)), float(ax.max())),
        lateral_gate_px=lateral_floor_px,
        axial_gate_slices=axial_gate,
    )
    n_single = sum(1 for o in organoids if o.single_view_in_overlap)
    if n_single:
        report.notes.append(
            f"{n_single} organoids sit in a doubly-covered region but were found "
            f"by one tile only; they are kept, and flagged, because the other "
            f"tile's segmenter missing them does not make them unreal")
    n_clip = sum(1 for o in organoids if o.clipped_everywhere)
    if n_clip:
        report.notes.append(
            f"{n_clip} organoids are cut off in every view they have, so their "
            f"shape and texture describe the cut rather than the object")
    return organoids, report


def _elect(members: list[Sighting], mosaic: Mosaic, lateral_floor: float,
           lateral_frac: float) -> Organoid:
    """Choose the view to report, and record how much the views disagreed.

    An unclipped view always beats a clipped one: a truncated outline gives a
    confident, wrong area and a confident, wrong circularity, and no weighting
    of it against a whole view is defensible. Among comparable views the one
    furthest from its own frame edge wins, because that is the one least likely
    to be losing part of the object to vignetting or to the edge of the field.
    """
    xs = np.array([s.x_px for s in members])
    ys = np.array([s.y_px for s in members])
    zs = np.array([s.z_slice for s in members])
    radii = np.array([s.radius_px for s in members])

    ranked = sorted(members, key=lambda s: (s.clipped, -s.edge_distance_px,
                                            -s.focus_sharpness))
    elected = ranked[0]

    cx, cy = float(np.mean(xs)), float(np.mean(ys))
    coverage = len(mosaic.covering(cx, cy))
    tiles_seen = {s.tile for s in members}

    o = Organoid(
        uid=0, elected=elected, views=list(members),
        coverage_k=coverage,
        n_views=len(members),
        lateral_spread_px=float(np.hypot(xs - cx, ys - cy).max()) if len(members) > 1 else 0.0,
        axial_spread_slices=float(zs.max() - zs.min()) if len(members) > 1 else 0.0,
        radius_ratio=float(radii.max() / max(radii.min(), 1e-6)),
    )
    o.single_view_in_overlap = coverage > 1 and len(tiles_seen) == 1
    o.clipped_everywhere = all(s.clipped for s in members)
    o.radius_disagreement = o.radius_ratio > 1.25
    o.same_tile_twice = len(tiles_seen) < len(members)

    overlap_x, overlap_y = mosaic.overlap_px()
    r_max = max(max(s.source.get("radial_profile_px") or [s.radius_px]) for s in members)
    o.guarantee_void = r_max > 0.5 * min(overlap_x, overlap_y)
    return o
