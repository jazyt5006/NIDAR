"""Golden outputs of the Layers 1-5 pipeline (substep 3.4, gate P1: behaviour must not change while it is optimised).

Harness side (uses the simulator). Run:
    PYTHONPATH=src python -m survivor_tagging.golden record     (writes tests/golden/layer5_golden.json.gz; only BEFORE a
                                                                  change that must not alter behaviour)
    PYTHONPATH=src python -m survivor_tagging.golden check      (all scenarios, 3 seeds, 2 offsets; ~3 min; exit 1 on diff)

Per run (scenario, seed, body offset, recompute mode) the record holds: finalize() tags, the published (GCS) tags with
tag revisions and cell probabilities, the final-tag sets of two policies, track / recompute / telemetry counts and the
score fields. Pooled evaluation numbers are computed from the per-run scores with harness.summarise. Comparison:
decisions (cells, statuses, counts, ids, revisions, flags) must be EXACTLY equal; floats within 1e-9. Floats are stored
rounded to 12 decimals. SIMULATION tested only.
"""

from __future__ import annotations

import gzip
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .harness import Score, ScenarioResult, run_events, summarise
from .pipeline import tag_result
from .scenarios import CORRECTION_SCENARIOS, all_scenarios
from .sim import build_events, place_survivors
from .sim_pipelines import layer5_factory

GOLDEN_PATH = Path(__file__).resolve().parents[2] / "tests" / "golden" / "layer5_golden.json.gz"
SEEDS = (0, 1, 2)
OFFSETS = (0.0, 0.2)
TOL = 1e-9
ROUND = 12


def _r(x: Any) -> Any:
    if isinstance(x, float):
        return round(x, ROUND)
    if isinstance(x, (list, tuple)):
        return [_r(v) for v in x]
    if hasattr(x, "tolist"):
        return _r(x.tolist())
    return x


def _tagrow(t) -> list:
    return [t.survivor_id, None if t.cell is None else list(t.cell), t.status.name,
            None if t.p_map_m is None else _r([float(v) for v in t.p_map_m]),
            None if t.top_prob is None else _r(float(t.top_prob)), t.cell_changes, t.first_stamp_ns, t.confirm_stamp_ns]


def _pubrow(pt) -> list:
    t = pt.tag
    return [pt.survivor_id, pt.tag_revision, pt.removed, t.status.name, None if t.cell is None else list(t.cell),
            _r([float(v) for v in t.p_map_m]), [[list(c), _r(float(p))] for c, p in t.cell_probs],
            t.observation_count, _r(float(t.sigma_horizontal_m)), _r(float(t.localisation_quality)),
            [r.value for r in t.reasons], len(pt.history), t.cell_changes]


def capture_run(name: str, seed: int, off: float, mode: str) -> dict:
    scn = all_scenarios(with_two_close=True)[name]
    placed = place_survivors(scn, seed)
    ov = {"l1.body_depth_offset_m": off}
    if mode != "full":
        ov["l5.recompute_mode"] = mode
    pl = layer5_factory(ov)(placed)
    res = run_events(placed, build_events(placed, seed), pl, seed)
    sc = res.score
    return dict(
        tags=[_tagrow(t) for t in res.tags],
        pub=[_pubrow(p) for p in pl.published()],
        final_ev=[_tagrow(t) for t in pl.final_tags("expected_value", 0.667)],
        final_co=[_tagrow(t) for t in pl.final_tags("confirmed_only")],
        n_tracks=len(pl.l4.tracks), n_recomputes=len(pl.recomputes), n_msgs=len(pl.sent),
        msg_bytes=sum(len(m.payload) for m in pl.sent), map_rev=pl.map_revision,
        score=dict(n_truth=sc.n_truth, n_correct=sc.n_correct, n_missed=sc.n_missed, n_false=sc.n_false,
                   n_duplicate=sc.n_duplicate, n_correct_confirmed=sc.n_correct_confirmed,
                   n_false_confirmed=sc.n_false_confirmed, n_false_provisional=sc.n_false_provisional,
                   n_flicker=sc.n_flicker, errors_m=_r(sc.errors_m), confirm_times_s=_r(sc.confirm_times_s),
                   confirm_from_first_obs_s=_r(sc.confirm_from_first_obs_s)))


def _task(args) -> List[Tuple[str, dict]]:
    name, off, mode, seeds = args
    return [(f"{mode}|{off}|{name}|{s}", capture_run(name, s, off, mode)) for s in seeds]


def _jobs(seeds: Tuple[int, ...] = SEEDS) -> List[tuple]:
    names = list(all_scenarios(with_two_close=True))
    jobs = [(n, off, "full", seeds) for n in names for off in OFFSETS]
    jobs += [(n, off, "affected_tracks", seeds) for n in CORRECTION_SCENARIOS for off in OFFSETS]
    return jobs


def pooled(runs: Dict[str, dict]) -> Dict[str, dict]:
    """Pooled evaluation rows (harness.summarise over all scenarios and seeds) per (mode, offset)."""
    out: Dict[str, dict] = {}
    groups: Dict[str, List[ScenarioResult]] = {}
    for key, r in runs.items():
        mode, off, name, seed = key.split("|")
        sc = Score(**r["score"])
        groups.setdefault(f"{mode}|{off}", []).append(ScenarioResult(name, int(seed), sc, [], 0.0))
    for g, results in groups.items():
        row = asdict(summarise("ALL", results))
        out[g] = {k: (None if isinstance(v, float) and math.isnan(v) else _r(v)) for k, v in row.items()}
    return out


def capture(parallel: bool = True, seeds: Tuple[int, ...] = SEEDS, jobs=None) -> Dict[str, Any]:
    """Runs the golden set (default: all scenarios, `seeds`, both offsets, plus affected_tracks for the correction
    scenarios). `jobs` (name, offset, mode, seeds) restricts it."""
    jobs = _jobs(seeds) if jobs is None else jobs
    runs: Dict[str, dict] = {}
    if parallel:
        with ProcessPoolExecutor(max(1, (os.cpu_count() or 2) - 1)) as ex:
            for part in ex.map(_task, jobs, chunksize=1):
                runs.update(part)
    else:
        for j in jobs:
            runs.update(_task(j))
    return {"runs": runs, "pooled": pooled(runs)}


def diff(a: Any, b: Any, path: str = "") -> List[str]:
    """Differences between two golden structures: exact for everything except floats (|a - b| <= TOL)."""
    if isinstance(a, bool) or isinstance(b, bool) or a is None or b is None or isinstance(a, str) or isinstance(b, str):
        return [] if a == b else [f"{path}: {a!r} != {b!r}"]
    if isinstance(a, dict) and isinstance(b, dict):
        out = [f"{path}: keys differ {sorted(set(a) ^ set(b))[:3]}"] if set(a) != set(b) else []
        for k in a:
            if k in b:
                out += diff(a[k], b[k], f"{path}/{k}")
        return out
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return [f"{path}: length {len(a)} != {len(b)}"]
        return [d for i, (x, y) in enumerate(zip(a, b)) for d in diff(x, y, f"{path}[{i}]")]
    if isinstance(a, int) and isinstance(b, int):
        return [] if a == b else [f"{path}: {a} != {b}"]
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return [] if abs(a - b) <= TOL else [f"{path}: {a} != {b} (> {TOL})"]
    return [f"{path}: type {type(a).__name__} vs {type(b).__name__}"]


def load() -> Dict[str, Any]:
    with gzip.open(GOLDEN_PATH, "rt", encoding="utf-8") as f:
        return json.load(f)


def save(data: Dict[str, Any]) -> None:
    GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(GOLDEN_PATH, "wt", encoding="utf-8", compresslevel=9) as f:
        json.dump(data, f, separators=(",", ":"), sort_keys=True)


def main(argv: List[str]) -> int:  # pragma: no cover
    if argv and argv[0] == "record":
        save(capture())
        print(f"saved {GOLDEN_PATH} ({GOLDEN_PATH.stat().st_size / 1024:.0f} KB)")
        return 0
    if argv and argv[0] == "check":
        gold = load()
        new = json.loads(json.dumps(capture()))
        d = diff(gold, new)
        print(f"golden check: {len(gold['runs'])} runs, {len(d)} differences")
        for line in d[:20]:
            print("  ", line)
        return 1 if d else 0
    print(__doc__)
    return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main(sys.argv[1:]))
