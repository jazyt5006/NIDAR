"""Layer 5: correction handling, revisions, survivor identity (substep 3.2). Unit + SIMULATION tests."""

from dataclasses import replace
from functools import lru_cache

import numpy as np
import pytest

from survivor_tagging.config import ConfigError
from survivor_tagging.experiments_layer5 import feed, r1_compare
from survivor_tagging.output import TagFlags, decode_snapshot, decode_tag_record, tag_record
from survivor_tagging.scenarios import (LAYER5_SCENARIOS, all_scenarios, make_scenario, with_correction_mode,
                                        without_correction_events)
from survivor_tagging.sim import Closure, build_events, place_survivors
from survivor_tagging.sim_pipelines import layer5_factory
from survivor_tagging.transforms import camera_to_map
from survivor_tagging.types import FailureReason, PoseCorrection, TagStatus

S = 1_000_000_000
F = FailureReason


@lru_cache(maxsize=None)
def stream(name, seed=0, mode=None, missing=False):
    scn = all_scenarios()[name]
    if mode is not None:
        scn = with_correction_mode(scn, mode)
    if missing:
        scn = without_correction_events(scn)
    placed = place_survivors(scn, seed)
    return placed, tuple(build_events(placed, seed))


def run(name, seed=0, overrides=None, mode=None, until_ns=None, events=None, sink=None):
    placed, ev = stream(name, seed, mode)
    ev = events if events is not None else ev
    pl = layer5_factory(overrides, sink=sink)(placed)
    feed(pl, [e for e in ev if until_ns is None or e.arrival_ns <= until_ns])
    return placed, pl


def corrections(name, seed=0, mode=None):
    return [e for e in stream(name, seed, mode)[1] if e.kind == "correction"]


def pre_closure_obs(pl, closure_s):
    return [o for o in pl.observations() if o.failure is None and o.p_map_m is not None
            and o.capture_stamp_ns < closure_s * S]


# --- spec 12.5 cases (one named test each) ------------------------------------------------------------------------

def test_125_slam_map_correction():
    """frame_correction source: the map jump moves exactly the observations captured before the correction."""
    (ce,) = corrections("frame_correction")
    _, before = run("frame_correction", until_ns=ce.arrival_ns - 1)
    old = {o.observation_id: o for o in pre_closure_obs(before, 2.5)}
    _, pl = run("frame_correction", until_ns=ce.arrival_ns)
    assert pl.outcomes[-1][1].applied and pl.map_revision == 1
    shift = ce.payload.T_new_old[:3, 3]
    new = {o.observation_id: o for o in pl.observations()}
    assert old
    for oid, o in old.items():
        n = new[oid]
        assert np.allclose(n.p_map_m - o.p_map_m, shift, atol=1e-9)
        assert F.FRAME_CORRECTION_APPLIED in n.warnings and n.pose_revision == 1
        assert np.allclose(n.cov_map_m2, o.cov_map_m2 * pl.l5.frame_inflation, rtol=1e-6, atol=1e-15)
    after = [o for o in new.values() if o.failure is None and o.p_map_m is not None and o.capture_stamp_ns >= 2.5 * S]
    assert after and all(F.FRAME_CORRECTION_APPLIED not in o.warnings for o in after)


def test_125_historical_pose_correction():
    """history source: pre-closure observations are re-derived from the corrected capture-time poses."""
    placed, pl = run("loop_closure")
    pre = pre_closure_obs(pl, 1.5)
    assert pre and all(o.pose_revision == 1 and F.POSE_NOT_CORRECTED not in o.warnings for o in pre)
    _, before = run("loop_closure", until_ns=corrections("loop_closure")[0].arrival_ns - 1)
    truth = np.asarray(placed.truths()[0].centre_map_xy)
    err_after = np.median([np.linalg.norm(o.p_map_m[:2] - truth) for o in pre])
    err_before = np.median([np.linalg.norm(o.p_map_m[:2] - truth) for o in pre_closure_obs(before, 1.5)])
    assert err_after < err_before - 0.1  # the 0.3 m pose error is gone (residual 0.02 m SIM)


def test_125_grid_registration_update():
    placed, ev = stream("registration_update")
    upd = next(e for e in ev if e.kind == "registration")
    _, pl = run("registration_update", until_ns=upd.arrival_ns)
    assert pl.grid.version == 1 and pl.outcomes[-1][1].applied
    for o in pl.observations():
        if o.failure is None and o.p_map_m is not None:
            assert o.registration_version == 1
            assert o.cell == pl.grid.cell_index(pl.grid.map_to_grid(o.p_map_m[:2]))
    assert all(t.registration_version == 1 for t in pl.l4.tags())
    snap = next(m for m in pl.sent if m.kind == "snapshot" and m.stamp_ns == upd.arrival_ns)
    assert decode_snapshot(snap.payload).reg_rev == 1  # snapshot sent immediately on the reg_rev change


def test_125_recalculation_of_survivor_positions():
    """After the second correction every stored observation equals camera point x effective pose, by hand."""
    _, pl = run("repeated_corrections")
    assert [r.cause for r in pl.recomputes] == ["correction r1", "correction r2"]
    n = 0
    for rec in pl.store:
        eff = pl.corrections.effective(rec)
        o = pl.derived[rec.observation_id]
        p, cov = camera_to_map(rec.p_camera_m, rec.cov_camera_m2, eff.lookup.pose, pl.l2.extrinsic)
        assert np.array_equal(o.p_map_m, p) and np.array_equal(o.cov_map_m2, cov)
        if rec.capture_pose_ns < 4.0 * S:
            assert o.pose_revision == 2
            n += 1
    assert n > 0
    for t in pl.l4.tracks.values():  # Layer 4 holds exactly the re-derived observations
        assert all(it.obs is pl.derived[it.obs.observation_id] for it in t.items)


def _reassign_scenario():
    """SIM: as loop_closure_late with a 0.9 m pose error along grid +y and the survivor at grid (3.7, 3.5), so the
    estimate (front surface ~0.25 m nearer the camera) sits mid-cell in x and the tag is confirmed in the wrong cell
    (3, 4) before the correction; it belongs in (3, 3)."""
    base = all_scenarios()["loop_closure_late"]
    yaw = base.registration.yaw_rad  # grid = Rz(yaw) map, so grid +y in the map is (sin yaw, cos yaw)
    tr = replace(base.trajectory, loop_closure_jump_m=(0.9 * np.sin(yaw), 0.9 * np.cos(yaw), 0.0))
    surv = (replace(base.survivors[0], grid_xy=(3.7, 3.5)),)
    return replace(base, name="reassign", trajectory=tr, survivors=surv, placement="nominal")


def test_125_reassignment_to_a_different_cell():
    scn = _reassign_scenario()
    ev = build_events(scn, 1)
    ce = next(e for e in ev if e.kind == "correction")
    pl = layer5_factory()(scn)
    feed(pl, [e for e in ev if e.arrival_ns < ce.arrival_ns])
    (pt0,) = pl.published()
    assert pt0.tag.status is TagStatus.CONFIRMED and pt0.tag.cell == (3, 4) and pt0.tag_revision == 0
    feed(pl, [e for e in ev if e.arrival_ns >= ce.arrival_ns])
    (pt,) = pl.published()
    assert pt.survivor_id == pt0.survivor_id
    assert pt.tag.status is TagStatus.UPDATED and pt.tag.cell == (3, 3) and pt.tag_revision == 1
    (ch,) = pt.history
    assert (ch.old_cell, ch.new_cell, ch.cause, ch.map_revision, ch.tag_revision) == ((3, 4), (3, 3), "correction r1",
                                                                                      1, 1)
    # The map_rev change triggers a snapshot at once (send rule 2); it already carries the UPDATED tag.
    at_event = [m for m in pl.sent if m.stamp_ns == ce.arrival_ns]
    recs = [r for m in at_event for r in (decode_snapshot(m.payload).records if m.kind == "snapshot"
                                          else (decode_tag_record(m.payload),))]
    assert any(r.status == TagStatus.UPDATED and r.cell == (3, 3) and r.map_rev == 1 for r in recs)


def test_125_live_output_consistency():
    """Every telemetry message equals the encoding of the published tags at the moment it was sent."""
    log = []
    holder = {}

    def sink(m):
        pl = holder["pl"]
        log.append((m, {pt.survivor_id: pt for pt in pl.published()}, pl.map_revision, pl.grid.version))

    placed, ev = stream("loop_closure_late")
    pl = layer5_factory(sink=sink)(placed)
    holder["pl"] = pl
    feed(pl, ev)
    pl.finalize()
    assert any(m.kind == "snapshot" for m, *_ in log) and any(m.kind == "update" for m, *_ in log)
    for m, pub, map_rev, reg_rev in log:
        recs = decode_snapshot(m.payload).records if m.kind == "snapshot" else (decode_tag_record(m.payload),)
        if m.kind == "snapshot":
            snap = decode_snapshot(m.payload)
            assert (snap.map_rev, snap.reg_rev) == (map_rev & 0xFF, reg_rev & 0xFF)
        for r in recs:
            if r.flags & TagFlags.REMOVED:
                continue
            exp = tag_record(pub[r.survivor_id], r.seq, pl.telemetry.t0_ns, pl.grid.valid, pl.telemetry_params.confirm_prob)
            assert (r.status, r.cell, r.flags, r.map_rev, r.reg_rev) == (exp.status, exp.cell, exp.flags, exp.map_rev,
                                                                        exp.reg_rev)
            assert r.grid_xy_m == pytest.approx(exp.grid_xy_m, abs=0.006)


def test_125_missing_correction_event():
    """A pose_revision bump without a correction event: after l5.missing_correction_timeout_s older observations are
    POSE_NOT_CORRECTED (inflated); a late real event then corrects them."""
    placed, ev = stream("loop_closure", missing=True)
    ce = corrections("loop_closure")[0]
    pl = layer5_factory({"l5.missing_correction_timeout_s": 0.3})(placed)
    feed(pl, ev)
    assert [r.cause for r in pl.recomputes] == ["missing correction r1"] and pl.map_revision == 0
    pre = pre_closure_obs(pl, 1.5)
    assert pre and all(F.POSE_NOT_CORRECTED in o.warnings for o in pre)
    assert any(F.POSE_NOT_CORRECTED in pt.tag.reasons for pt in pl.published())
    out = pl.process_correction(ce.payload, ev[-1].arrival_ns + 1)  # the real event, late
    assert out.applied and pl.map_revision == 1
    assert all(F.POSE_NOT_CORRECTED not in o.warnings and o.pose_revision == 1 for o in pre_closure_obs(pl, 1.5))


def test_125_restart_without_persistence_starts_clean():
    """Persistence is not implemented: a restarted pipeline has no stored observations, so a correction for the
    old flight changes nothing and does not fail (state recovery is out of scope, see report)."""
    placed, _ = stream("loop_closure")
    pl = layer5_factory()(placed)
    out = pl.process_correction(corrections("loop_closure")[0].payload, 5 * S)
    assert out.applied and len(pl.store) == 0 and pl.published() == [] and pl.map_revision == 1


# --- decision rule R1 and correction semantics ----------------------------------------------------------------------

@pytest.mark.parametrize("name,seed", [("loop_closure", 0), ("loop_closure", 4), ("repeated_corrections", 1),
                                       ("loop_closure_late", 1)])
def test_r1_tags_equal_offline_run_with_corrected_poses(name, seed):
    """Right after each correction (offline run on the corrected poses known then) and at the end."""
    placed, ev = stream(name, seed)
    res = r1_compare(placed, ev)
    assert len(res) == 1 + sum(e.kind == "correction" for e in ev)
    assert all(eq and worst <= 1e-9 for eq, worst in res), res


def test_same_event_twice_changes_nothing():
    placed, pl = run("loop_closure")
    tags0 = [(t.cell, t.p_map_m.copy()) for t in pl.l4.tags()]
    n_rc = len(pl.recomputes)
    out = pl.process_correction(corrections("loop_closure")[0].payload, 10 * S)
    assert not out.applied and out.reason is None and out.detail == "duplicate"
    assert len(pl.recomputes) == n_rc
    tags1 = [(t.cell, t.p_map_m) for t in pl.l4.tags()]
    assert len(tags1) == len(tags0) and all(a[0] == b[0] and np.array_equal(a[1], b[1]) for a, b in zip(tags0, tags1))


def test_out_of_order_events_are_applied_by_revision():
    placed, ev = stream("repeated_corrections", 2)
    c1, c2 = (e.payload for e in ev if e.kind == "correction")
    rest = [e for e in ev if e.kind != "correction"]
    end = rest[-1].arrival_ns
    results = []
    for order in ((c1, c2), (c2, c1)):
        pl = layer5_factory()(placed)
        feed(pl, rest)
        for c in order:
            assert pl.process_correction(c, end + 1).applied
        results.append(pl)
    ta, tb = (sorted(r.l4.tags(), key=lambda t: t.first_seen_ns) for r in results)
    assert len(ta) == len(tb) and all(a.cell == b.cell and np.array_equal(a.p_map_m, b.p_map_m) for a, b in zip(ta, tb))
    assert results[0].map_revision == results[1].map_revision == 2


def test_revisions_are_monotonic_conflicts_and_stale_events_rejected():
    placed, ev = stream("repeated_corrections")
    c1, c2 = (e.payload for e in ev if e.kind == "correction")
    pl = layer5_factory({"l5.correction_history_max": 1})(placed)
    revs = []
    for e in ev:
        feed(pl, [e])
        revs.append(pl.map_revision)
    assert revs == sorted(revs) and revs[-1] == 2
    heads = [decode_snapshot(m.payload).map_rev for m in pl.sent if m.kind == "snapshot"]
    assert heads == sorted(heads)
    assert pl.corrections.baked_revision == 1  # history bound 1: revision 1 was folded in
    assert pl.process_correction(c1, ev[-1].arrival_ns).reason is F.CORRECTION_STALE
    other = replace(c2, poses=tuple(replace(p, position_m=p.position_m + 0.01) for p in c2.poses))
    assert pl.process_correction(other, ev[-1].arrival_ns).reason is F.CORRECTION_CONFLICT
    assert pl.map_revision == 2


def test_partial_history_flags_uncorrected_observations():
    _, pl = run("partial_history")
    obs = [o for o in pl.observations() if o.failure is None and o.p_map_m is not None]
    early = [o for o in obs if o.capture_stamp_ns < 0.88 * S]
    covered = [o for o in obs if 0.9 * S <= o.capture_stamp_ns < 1.48 * S]
    late = [o for o in obs if o.capture_stamp_ns >= 1.5 * S]
    assert early and covered and late
    assert all(F.POSE_NOT_CORRECTED in o.warnings and o.pose_revision in (0, 1) for o in early)
    assert all(F.POSE_NOT_CORRECTED not in o.warnings and o.pose_revision == 1 for o in covered + late)
    for o in early:  # inflated, uncorrected: same position as the live derivation, covariance x factor
        rec = pl.store.get(o.observation_id)
        p, cov = camera_to_map(rec.p_camera_m, rec.cov_camera_m2, rec.base.pose, pl.l2.extrinsic)
        assert np.array_equal(o.p_map_m, p) and np.allclose(o.cov_map_m2, cov * pl.l5.uncorrected_inflation)


def test_evicted_observations_are_flagged_and_kept_bounded():
    """Observations evicted before a recompute are replayed with their old position and flagged."""
    ce = corrections("loop_closure")[0]
    _, pl = run("loop_closure", overrides={"l5.observation_store_max": 5}, until_ns=ce.arrival_ns)
    assert len(pl.store) == 5 and pl.store.n_evicted > 0
    carried = [it.obs for t in pl.l4.tracks.values() for it in t.items if it.obs.observation_id not in pl.store]
    assert carried and all(F.OBSERVATION_NOT_REDERIVABLE in o.warnings for o in carried)
    assert any(F.OBSERVATION_NOT_REDERIVABLE in pt.tag.reasons for pt in pl.published())
    assert len(pl._carried) <= pl.l4.p.max_tracks * pl.l4.p.max_obs_per_track


def test_survivor_id_continuity_across_correction():
    ce = corrections("loop_closure_late")[0]
    _, before = run("loop_closure_late", until_ns=ce.arrival_ns - 1)
    _, after = run("loop_closure_late", until_ns=ce.arrival_ns)
    ids_b = {pt.survivor_id for pt in before.published()}
    ids_a = {pt.survivor_id for pt in after.published()}
    assert ids_b and ids_a <= ids_b  # no new id for the same survivor


def test_split_tracks_merge_after_correction():
    """loop_closure seed 0: without Layer 5 the 0.3 m pose error splits the survivor into two tags."""
    from survivor_tagging.harness import run_events
    from survivor_tagging.sim_pipelines import layer1234_factory

    placed, ev = stream("loop_closure")
    assert len(run_events(placed, ev, layer1234_factory()(placed), 0).tags) == 2
    _, pl = run("loop_closure")
    assert len(pl.finalize()) == 1 and pl.published()[0].survivor_id == 0


def test_survivor_id_rule_on_merge_and_removal_announcement():
    """TagBook: a new track holding the observations of two old survivors keeps the id it shares more
    observations with; the other id is announced once as removed (telemetry REMOVED flag)."""
    from survivor_tagging.corrections import TagBook, item_reasons
    from survivor_tagging.grid import GridRegistration
    from survivor_tagging.output import TelemetryScheduler
    from survivor_tagging.pipeline import telemetry_params
    from survivor_tagging.tracking import Layer4, Layer4Params
    from survivor_tagging.config import default_config
    from survivor_tagging.types import SurvivorObservation

    reg = GridRegistration((0.0, 0.0), 0.0, n_cells_x=6, n_cells_y=6, valid=True)
    prm = Layer4Params.from_config(default_config())

    def ob(i, t, xy):
        return SurvivorObservation(i, 0, 0.9, int(t * S), None, p_map_m=np.array([xy[0], xy[1], 0.85]),
                                   cov_map_m2=np.eye(3) * 4e-4, camera_map_m=np.array([xy[0] - 2.5, xy[1], 1.5]),
                                   pose_revision=0, extrinsic_version=1)

    old = Layer4(prm, reg)  # two tracks 0.6 m apart (a split), 4 + 3 observations
    for k in range(4):
        old.update([ob(k, 0.2 * k, (2.5, 2.5))])
    for k in range(3):
        old.update([ob(10 + k, 1.0 + 0.2 * k, (2.5, 3.1))])
    book = TagBook(4, 4)
    pubs = book.publish(old, "layer4", 0, item_reasons)
    assert len(pubs) == 2
    new = Layer4(prm, reg)  # after a correction all 7 observations agree
    for i, t in [(k, 0.2 * k) for k in range(4)] + [(10 + k, 1.0 + 0.2 * k) for k in range(3)]:
        new.update([ob(i, t, (2.5, 2.5))])
    assert len(new.tracks) == 1
    mapping = book.remap(old.tracks, new.tracks)
    assert list(mapping.values()) == [0]  # 4 shared observations beat 3
    (pub,) = book.publish(new, "correction r1", 1, item_reasons)
    assert pub.survivor_id == 0
    removed = book.pop_removed()
    assert [r.survivor_id for r in removed] == [1] and removed[0].removed
    sched = TelemetryScheduler(telemetry_params(default_config()), 0)
    sched.step(0, pubs, [], 0, 0, True)
    msgs = sched.step(2 * S, [pub], removed, 1, 0, True)
    recs = [decode_tag_record(m.payload) for m in msgs if m.kind == "update"]
    assert any(r.survivor_id == 1 and r.flags & TagFlags.REMOVED for r in recs)
    assert not sched.step(4 * S, [pub], [], 1, 0, True) or all(
        decode_tag_record(m.payload).survivor_id != 1 for m in sched.step(4 * S, [pub], [], 1, 0, True)
        if m.kind == "update")


def test_registration_update_duplicate_and_stale():
    placed, pl = run("registration_update")
    reg = pl.grid
    assert pl.process_registration(reg, 9 * S).detail == "duplicate"
    assert pl.process_registration(replace(reg, version=0), 9 * S).reason is F.CORRECTION_STALE
    assert pl.process_registration(replace(reg, yaw_rad=reg.yaw_rad + 0.1), 9 * S).reason is F.CORRECTION_CONFLICT


def test_invalid_corrections_are_rejected_explicitly():
    placed, ev = stream("loop_closure")
    c = corrections("loop_closure")[0].payload
    pl = layer5_factory()(placed)
    feed(pl, [e for e in ev if e.kind == "pose"][:5])
    bad_rev = replace(c, poses=tuple(replace(p, pose_revision=7) for p in c.poses))
    assert pl.process_correction(bad_rev, S).reason is F.CORRECTION_INVALID
    frame = PoseCorrection(1, "frame_correction", c.stamp_ns, "sim", (), np.eye(4))
    assert pl.process_correction(frame, S).reason is F.CORRECTION_SOURCE_UNAVAILABLE  # configured: history
    pl2 = layer5_factory({"l5.correction_source": "frame_correction"})(placed)
    skew = np.eye(4)
    skew[0, 1] = 0.5
    assert pl2.process_correction(PoseCorrection(1, "frame_correction", c.stamp_ns, "sim", (), skew), S).reason \
        is F.CORRECTION_INVALID
    pl3 = layer5_factory({"l5.correction_source": "none"})(placed)
    assert pl3.process_correction(c, S).reason is F.CORRECTION_SOURCE_UNAVAILABLE


def test_frame_correction_mode_is_worse_than_history_on_drift():
    """SIM: with drift inside the window the single jump leaves part of the error (C4)."""
    errs = {}
    for mode in ("history", "frame_correction"):
        placed, pl = run("frame_correction", mode=mode)
        truth = np.asarray(placed.truths()[0].centre_map_xy)
        errs[mode] = np.median([np.linalg.norm(o.p_map_m[:2] - truth) for o in pre_closure_obs(pl, 2.5)])
    assert errs["history"] < errs["frame_correction"]


def test_layer5_is_deterministic():
    a = [m.payload for m in run("repeated_corrections", 1)[1].sent]
    b = [m.payload for m in run("repeated_corrections", 1)[1].sent]
    assert a == b and len(a) > 0


@pytest.mark.parametrize("name", LAYER5_SCENARIOS)
def test_no_nan_in_layer5_outputs(name):
    _, pl = run(name)
    pl.finalize()
    for pt in pl.published():
        t = pt.tag
        assert np.all(np.isfinite(t.p_map_m)) and np.all(np.isfinite(t.cov_map_m2)) and np.all(np.isfinite(t.p_grid_m))
        assert np.isfinite(t.sigma_horizontal_m) and 0.0 <= t.localisation_quality <= 1.0
    for o in pl.observations():
        if o.p_map_m is not None:
            assert np.all(np.isfinite(o.p_map_m)) and np.all(np.isfinite(o.cov_map_m2))


def test_layer5_memory_is_bounded():
    ov = {"l5.observation_store_max": 10, "l5.correction_history_max": 1, "l5.tag_history_max": 1,
          "l5.telemetry_log_max": 5, "l5.removed_tags_max": 2}
    _, pl = run("repeated_corrections", overrides=ov)
    assert len(pl.store) <= 10 and len(pl.corrections.events) <= 1 and len(pl.sent) <= 5
    assert all(len(pt.history) <= 1 for pt in pl.published()) and len(pl.book.removed) <= 2
    assert all(len(e.corrected) <= 10 for e in pl.corrections.events.values())
    assert len(pl.derived) <= 10 and len(pl._keys) <= 10
    assert len(pl.recomputes) <= pl.recomputes.maxlen and len(pl.outcomes) <= pl.outcomes.maxlen


def test_final_tags_use_the_configured_policy_only():
    placed, pl = run("loop_closure_late")
    with pytest.raises(ConfigError):
        pl.final_tags()  # l5.end_of_mission_commit_policy unset (O2): no default is claimed as correct
    placed, pl = run("loop_closure_late", overrides={"l5.end_of_mission_commit_policy":
                                                     "confirmed_plus_provisional_top1"})
    ft = pl.final_tags()
    pub = {pt.survivor_id for pt in pl.published()}
    assert ft and {t.survivor_id for t in ft} <= pub


def test_affected_tracks_mode_merges_the_split_and_matches_full_here():
    _, full = run("loop_closure")
    _, inc = run("loop_closure", overrides={"l5.recompute_mode": "affected_tracks"})
    assert inc.recomputes[-1].mode == "affected_tracks"
    assert len(inc.published()) == len(full.published()) == 1
    assert inc.published()[0].tag.cell == full.published()[0].tag.cell
