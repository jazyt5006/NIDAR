"""Effect of the default pose covariance (pose.default_position_sigma_m 0.10, pose.default_yaw_sigma_rad 0.05) in the simulator.
Poses of these scenarios carry NO covariance in the simulator; the messages therefore have an all-zero (unknown) covariance.
Compares: default disabled (the pose claims to be exact) vs enabled. Run from the repo root:
  PYTHONPATH=src:ros2/survivor_tagging_ros python ros2/survivor_tagging_ros/tools/cov_default_effect.py > docs/reports/4.1b_cov_default.txt
SIMULATION evidence, body offset 0.0, 5 seeds."""
import sys

import numpy as np

sys.path.insert(0, "tests")
import ros_runtime_helpers as H  # noqa: E402
from survivor_tagging_ros import adapters as A  # noqa: E402

SCENARIOS = ("standing", "sitting", "near_wall", "tilt_30", "two_adjacent", "cell_boundary", "revisit_after_gap")
ON = dict(pose_default_position_sigma_m=0.10, pose_default_yaw_sigma_rad=0.05)
print("scenario                 mode      tags  top_prob(mean)  sigma_h(mean m)  CONFIRMED  correct(any)  has POSE_COV_ASSUMED")
tot = {"off": [], "on": []}
for name in SCENARIOS:
    for mode, kw in (("off", {}), ("on", ON)):
        probs, sig, conf, corr, n, flag, n_truth = [], [], 0, 0, 0, 0, 0
        for seed in range(5):
            placed, events, rt, _ = H.run_via_messages(name, seed, **kw)
            truths = {t.survivor_id: t for t in placed.truths()}
            n_truth += len(truths)
            for d in rt.latest_tags["tags"]:
                tag = A.tag_from_msg_dict(d)[0]
                n += 1
                probs.append(tag.localisation_quality)
                sig.append(tag.sigma_horizontal_m)
                conf += tag.status.name in ("CONFIRMED", "UPDATED")
                flag += "POSE_COV_ASSUMED" in [r.value for r in tag.reasons]
                near = min(truths.values(), key=lambda t: np.hypot(*(np.array(t.centre_map_xy) - tag.p_map_m[:2])))
                corr += tag.cell == near.cell
        tot[mode] += probs
        print(f"{name:<24} {mode:<8} {n:>5} {np.mean(probs):>14.3f} {np.mean(sig):>16.3f} {conf:>10} {corr:>9}/{n_truth:<3} {flag:>8}")
print(f"\nall scenarios: mean top-cell probability off {np.mean(tot['off']):.3f} -> on {np.mean(tot['on']):.3f}")
