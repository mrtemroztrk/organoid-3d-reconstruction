"""Link per-slice detections into 3D objects.

Organoids embedded in Matrigel do not move, so the same organoid appears at
essentially the same (x, y) on every slice where it is visible -- it just grows
blurrier and slightly larger away from its focal plane. Linking is therefore a
lateral-proximity assignment problem between consecutive segmented slices,
solved optimally per slice pair with the Hungarian algorithm.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from .config import Params
from .detect import Detection


@dataclass
class Track:
    """One candidate organoid, seen on a contiguous-ish run of slices."""

    tid: int
    dets: list[Detection] = field(default_factory=list)

    @property
    def z_first(self) -> int:
        return self.dets[0].z

    @property
    def z_last(self) -> int:
        return self.dets[-1].z

    @property
    def n_slices(self) -> int:
        return len(self.dets)

    @property
    def cx(self) -> float:
        return float(np.mean([d.cx for d in self.dets]))

    @property
    def cy(self) -> float:
        return float(np.mean([d.cy for d in self.dets]))

    def det_at(self, z: int) -> Detection | None:
        for d in self.dets:
            if d.z == z:
                return d
        return None


_UNMATCHED = 1e6


def _cost(a: Detection, b: Detection, p: Params) -> float:
    """Assignment cost between a detection on slice z and one on slice z'."""
    ref_r = 0.5 * (a.radius_px + b.radius_px)
    if ref_r <= 0:
        return _UNMATCHED

    dist = float(np.hypot(a.cx - b.cx, a.cy - b.cy)) / ref_r
    if dist > p.link_max_center_shift:
        return _UNMATCHED

    ratio = max(a.radius_px, b.radius_px) / max(1e-6, min(a.radius_px, b.radius_px))
    if ratio > p.link_max_radius_ratio:
        return _UNMATCHED

    return dist + 0.5 * np.log(ratio)


def link_tracks(per_slice: list[list[Detection]], params: Params,
                max_gap: int = 2) -> list[Track]:
    """Greedy-optimal chaining of detections across Z.

    `max_gap` is in processed-slice units: a track survives that many slices
    without a detection before it is closed (a merged clump or a momentary
    Cellpose miss should not split one organoid into two objects).
    """
    tracks: list[Track] = []
    active: list[Track] = []      # tracks still open
    next_id = 1

    processed = [z for z, dets in enumerate(per_slice) if dets]
    for z in processed:
        dets = per_slice[z]

        # close tracks that have gone quiet
        still_active = []
        for t in active:
            gap = sum(1 for zz in processed if t.z_last < zz < z)
            if gap <= max_gap:
                still_active.append(t)
        active = still_active

        if not active:
            for d in dets:
                t = Track(tid=next_id, dets=[d])
                next_id += 1
                tracks.append(t)
                active.append(t)
            continue

        cost = np.full((len(active), len(dets)), _UNMATCHED, dtype=np.float64)
        for i, t in enumerate(active):
            last = t.dets[-1]
            for j, d in enumerate(dets):
                cost[i, j] = _cost(last, d, params)

        rows, cols = linear_sum_assignment(cost)
        matched_dets: set[int] = set()
        for i, j in zip(rows, cols):
            if cost[i, j] >= _UNMATCHED:
                continue
            active[i].dets.append(dets[j])
            matched_dets.add(j)

        for j, d in enumerate(dets):
            if j in matched_dets:
                continue
            t = Track(tid=next_id, dets=[d])
            next_id += 1
            tracks.append(t)
            active.append(t)

    return [t for t in tracks if t.n_slices >= params.min_track_slices]


def drop_substrate_tracks(tracks: list[Track], z_substrate: int,
                          margin: int) -> tuple[list[Track], int]:
    """Remove objects that live at or below the dish surface.

    Debris settled on the glass is sharp, round and plentiful, and it is not
    biology. Anything whose whole extent sits below the substrate plane goes.
    """
    keep, dropped = [], 0
    limit = z_substrate - margin
    for t in tracks:
        if t.z_first >= limit:
            dropped += 1
            continue
        keep.append(t)
    return keep, dropped
