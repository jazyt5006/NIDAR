"""Substep 3.4: a full recompute can run on a worker thread from a snapshot and be swapped in atomically.

Design (pipeline.RebuildJob, Layer5Pipeline.job_sink / complete_rebuild): the pipeline thread takes the snapshot, any
thread runs job.run() (touching only the snapshot and read-only configuration), the pipeline thread installs the result.
While the job runs, the live path keeps updating the old state. Observations accepted meanwhile are replayed into the
new state at completion, so the end state equals the one a synchronous recompute gives. SIMULATION scenario.
"""
import threading

import numpy as np

from survivor_tagging.pipeline import RebuildJob
from survivor_tagging.scenarios import all_scenarios
from survivor_tagging.sim import build_events, place_survivors
from survivor_tagging.sim_pipelines import layer5_factory

SCN = "repeated_corrections"  # two history corrections (2.0 s, 4.0 s) with observations after each of them


def _events(seed=0):
    placed = place_survivors(all_scenarios()[SCN], seed)
    return placed, build_events(placed, seed)


def _feed_one(pl, ev):
    if ev.kind == "pose":
        pl.process_pose(ev.payload, ev.arrival_ns)
    elif ev.kind == "frame":
        pl.process_frame(ev.payload.detections, ev.payload.depth, ev.payload.intrinsics, ev.arrival_ns)
    elif ev.kind == "correction":
        pl.process_correction(ev.payload, ev.arrival_ns)


def _sync_pipeline(placed, events):
    pl = layer5_factory()(placed)
    for ev in events:
        _feed_one(pl, ev)
    return pl


def _summary(tags):
    return sorted((t.cell, t.status.name, tuple(np.round(np.asarray(t.p_map_m, dtype=float)[:2], 9))) for t in tags)


def _same(a, b):
    assert len(a) == len(b)
    for (ca, sa, pa), (cb, sb, pb) in zip(a, b):
        assert (ca, sa) == (cb, sb)
        assert np.allclose(pa, pb, atol=1e-9, rtol=0)


class _Worker:
    """Runs job.run() on its own thread, but only after go() (so the test controls how long the job is in flight)."""

    def __init__(self, job: RebuildJob):
        self.job, self._go = job, threading.Event()
        self.thread = threading.Thread(target=self._main, daemon=True)
        self.thread.start()

    def _main(self):
        self._go.wait(10)
        try:
            self.job.run()
        except BaseException:  # noqa: BLE001 - kept on job.error
            pass

    def go(self):
        self._go.set()
        self.thread.join(30)
        assert not self.thread.is_alive() and self.job.error is None


def test_async_recompute_equals_synchronous_and_does_not_block_live_updates():
    placed, events = _events()
    expected = _sync_pipeline(placed, events)
    pl = layer5_factory()(placed)
    jobs, workers = [], []
    pl.job_sink = jobs.append
    frames_while_pending = 0
    for ev in events:
        n_jobs = len(jobs)
        _feed_one(pl, ev)
        if len(jobs) > n_jobs:
            workers.append(_Worker(jobs[-1]))  # blocked until go(): the job is "in flight"
        elif workers and ev.kind == "frame":
            frames_while_pending += 1
            assert all(w.thread.is_alive() for w in workers)  # the live path ran while the workers were still pending
    assert len(jobs) == 2 and frames_while_pending > 0
    old_l4 = pl.l4
    for w in workers:
        w.go()
    assert pl.complete_rebuild(jobs[0]) is False  # superseded by the second correction
    assert pl.l4 is old_l4
    assert pl.complete_rebuild(jobs[1]) is True
    assert pl.l4 is not old_l4
    _same(_summary(pl.finalize()), _summary(expected.finalize()))
    _same(_summary(pl.final_tags("expected_value", 0.667)), _summary(expected.final_tags("expected_value", 0.667)))
    assert pl.map_revision == expected.map_revision


def test_job_reads_only_its_snapshot():
    """A job started at the first correction gives the same Layer 4 state as the synchronous rebuild at that moment,
    although the live pipeline keeps changing (more frames, a second correction) before the job runs."""
    placed, events = _events()
    first = next(i for i, e in enumerate(events) if e.kind == "correction")
    sync = _sync_pipeline(placed, events[:first + 1])
    pl = layer5_factory()(placed)
    jobs = []
    pl.job_sink = jobs.append
    for ev in events:
        _feed_one(pl, ev)
    job = jobs[0]
    assert job.state is None
    st = job.run()  # run late, after the live state moved on
    a = {t.track_id: t for t in sync.l4.tracks.values()}
    b = {t.track_id: t for t in st.l4.tracks.values()}
    assert sorted(a) == sorted(b)
    for k in a:
        assert a[k].status == b[k].status and a[k].n_frames == b[k].n_frames
        assert np.allclose(a[k].x, b[k].x, atol=1e-9, rtol=0) and np.allclose(a[k].P, b[k].P, atol=1e-12, rtol=0)
        assert a[k].top()[0] == b[k].top()[0]


def test_new_state_shares_no_mutable_object_with_the_live_state():
    placed, events = _events()
    pl = layer5_factory()(placed)
    jobs = []
    pl.job_sink = jobs.append
    for ev in events:
        _feed_one(pl, ev)
        if jobs:
            break
    st = jobs[0].run()
    live_obs = {id(it.obs) for t in pl.l4.tracks.values() for it in t.items} | {id(o) for o in pl.derived.values()}
    new_obs = {id(it.obs) for t in st.l4.tracks.values() for it in t.items} | {id(o) for o in st.derived.values()}
    assert new_obs and not (live_obs & new_obs)
    assert st.l4 is not pl.l4 and st.derived is not pl.derived


def test_concurrent_live_updates_while_the_job_runs_do_not_disturb_either_side():
    """The worker really runs concurrently with live updates (no gate); neither side raises and the result equals the
    synchronous one."""
    placed, events = _events(seed=1)
    expected = _sync_pipeline(placed, events)
    pl = layer5_factory()(placed)
    jobs = []
    pl.job_sink = jobs.append
    thread = None
    for ev in events:
        n = len(jobs)
        _feed_one(pl, ev)
        if len(jobs) > n:
            thread = threading.Thread(target=jobs[-1].run, daemon=True)
            thread.start()
            if len(jobs) == 1:
                continue
    thread.join(60)
    assert jobs[-1].error is None and jobs[-1].state is not None
    assert pl.complete_rebuild(jobs[-1])
    _same(_summary(pl.finalize()), _summary(expected.finalize()))


def test_inline_mode_is_unchanged_when_no_sink_is_set():
    placed, events = _events()
    pl = _sync_pipeline(placed, events)
    assert pl.job_sink is None and len(pl.recomputes) >= 2 and all(r.mode == "full" for r in pl.recomputes)
