"""Why are survivors not confirmed? (substep 3.1b, part A). Harness-side: reads simulator ground truth and the
Layer1234 pipeline's Layer 4 state. Never imported by library layers.

Exactly one label per survivor instance that is not CONFIRMED in the true cell, checked in this order:
  confirmed_wrong_cell  - its main track is CONFIRMED/UPDATED in another cell
  split                 - two or more live tracks each hold >= l4.provisional_min_obs of its observations
  never_associated      - none of its accepted observations is held by any track (all rejected or ambiguous)
  blocked               - the main track carries a BLOCK-severity warning
  min_obs               - main track has fewer than l4.confirm_min_obs observations
  span                  - first-to-last span below l4.confirm_min_span_s
  prob_below            - top-cell probability below l4.confirm_prob (probability and edge distance recorded)
  other                 - anything else (e.g. UNRELIABLE)
The main track is the one holding most of the survivor's observations (live tracks preferred over dropped ones).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .types import TagStatus

LABELS = ("confirmed_wrong_cell", "split", "never_associated", "blocked", "min_obs", "span", "prob_below", "other")
LENGTH_ARTIFACTS = ("min_obs", "span")  # would disappear with a longer look at the survivor
CONFIRMED = (TagStatus.CONFIRMED, TagStatus.UPDATED)


@dataclass
class SurvivorDiag:
    scenario: str
    survivor_id: int
    confirmed_correct: bool
    label: Optional[str]
    n_detections: int
    n_accepted: int
    top_prob: Optional[float] = None
    edge_distance_m: Optional[float] = None
    main_status: Optional[str] = None


@dataclass
class DiagSummary:
    survivors: List[SurvivorDiag] = field(default_factory=list)
    statuses: Counter = field(default_factory=Counter)  # final track statuses + "DROPPED_TIMEOUT" + "MERGED"


def diagnose_run(placed, frames, pipeline, summary: DiagSummary) -> None:
    l4, p = pipeline.l4, pipeline.l4.p
    sid_of = {(d.image_stamp_ns, d.detection_id): sid
              for fr in frames for d, sid in zip(fr.detections, fr.det_survivor_ids)}
    n_det = Counter(sid for fr in frames for sid in fr.det_survivor_ids if sid is not None)
    accepted: Dict[int, List[int]] = {}
    for o in pipeline.observations():
        sid = sid_of.get((o.capture_stamp_ns, o.detection_id))
        if sid is not None and o.failure is None and o.cell is not None:
            accepted.setdefault(sid, []).append(o.observation_id)
    holder: Dict[int, object] = {}
    for t in list(l4.dropped) + list(l4.tracks.values()):  # live tracks last: they win
        for it in t.items:
            holder[it.obs.observation_id] = t
    for t in l4.tracks.values():
        summary.statuses[t.status.name] += 1
    for t in l4.dropped:
        last = t.history[-1][1] if t.history else ""
        summary.statuses["MERGED" if last == "merged_into" else "DROPPED_TIMEOUT" if last == "dropped" else
                         "EVICTED"] += 1

    for truth in placed.truths():
        sid = truth.survivor_id
        ids = accepted.get(sid, [])
        counts = Counter(id(holder[i]) for i in ids if i in holder)
        by_id = {id(holder[i]): holder[i] for i in ids if i in holder}
        live = {id(t) for t in l4.tracks.values()}
        main = None
        if counts:
            main = by_id[max(counts, key=lambda k: (k in live, counts[k]))]
        d = SurvivorDiag(placed.name, sid, False, None, n_det.get(sid, 0), len(ids))
        if main is not None:
            d.main_status = main.status.name
        if main is not None and main.status in CONFIRMED and tuple(main.confirmed_cell) == tuple(truth.cell):
            d.confirmed_correct = True
        elif main is not None and main.status in CONFIRMED:
            d.label = "confirmed_wrong_cell"
        elif sum(1 for k, n in counts.items() if k in live and n >= p.provisional_min_obs) >= 2:
            d.label = "split"
        elif main is None:
            d.label = "never_associated"
        elif l4.blocked(main):
            d.label = "blocked"
        elif main.n_frames < p.confirm_min_obs:
            d.label = "min_obs"
        elif (main.last_seen_ns - main.first_seen_ns) < p.confirm_min_span_s * 1e9:
            d.label = "span"
        elif main.top()[1] < p.confirm_prob:
            d.label = "prob_below"
            d.top_prob = float(main.top()[1])
            d.edge_distance_m = float(l4.reg.boundary_distance_m(l4.reg.map_to_grid(main.x)))
        else:
            d.label = "other"
        summary.survivors.append(d)


def format_diag(summary: DiagSummary, scenarios: List[str]) -> str:
    head = (f"{'scenario':<29}{'n':>4}{'ok':>4}" + "".join(f"{lab[:10]:>11}" for lab in LABELS)
            + f"{'obs min/med/max':>17}{'p_top med':>10}{'edge med':>9}")
    lines = [head, "-" * len(head)]
    for s in scenarios + ["ALL"]:
        ds = [d for d in summary.survivors if s == "ALL" or d.scenario == s]
        if not ds:
            continue
        c = Counter(d.label for d in ds if not d.confirmed_correct)
        obs = [d.n_accepted for d in ds]
        pb = [d.top_prob for d in ds if d.top_prob is not None]
        ed = [d.edge_distance_m for d in ds if d.edge_distance_m is not None]
        lines.append(f"{s:<29}{len(ds):>4}{sum(d.confirmed_correct for d in ds):>4}"
                     + "".join(f"{c.get(lab, 0):>11}" for lab in LABELS)
                     + f"{f'{min(obs)}/{int(np.median(obs))}/{max(obs)}':>17}"
                     + (f"{np.median(pb):>10.2f}" if pb else f"{'-':>10}")
                     + (f"{np.median(ed):>9.2f}" if ed else f"{'-':>9}"))
    return "\n".join(lines)
