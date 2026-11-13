"""possession_v3: v1 + team-shape sensor + Smart Tagging restarts. Settings chosen on the proxy board.

    python starter/possession_v3.py --cache tracking_v2.npz
    python evaluation/validate_submission.py evaluation/results/possession_v3.csv

Layers (each a per-frame cost before the same Viterbi decoding as v1):
  1. v1: ball proximity, ball speed, ball out of pitch, short-gap interpolation.
  2. Team shape (starter/team_shape.py): who has the ball judged from the 20 players' shape;
     works when the ball is missing. Weights learned on public Metrica sample games only.
  3. Smart Tagging (optional, see --tags): kick-off anchoring, set-piece restarts and their team.
  4. Collective player speed (optional, --speed): off by default -- neutral on the proxy board.

Why these settings: tools/proxy_board.py builds a leaderboard from public sample games with
degraded tracking (3 public games). On it, v1 scores spellF1 0.514, v1 + team shape 0.561, and
the tuned profile 0.590 (fresh degradation seeds), better than v1 in every game. The speed layer did not help there. Smart Tagging cannot be tested
there (the sample games have none); it is kept for football reasons and documented as such.
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from games import GAME, GAMES, half_video_blocks, require, source_frames  # noqa: E402
from baseline import team_of_each_track, write_submission  # noqa: E402
import possession_v1 as v1  # noqa: E402
import possession_v2 as v2  # noqa: E402
import team_shape as ts  # noqa: E402

H, A, D = v1.H, v1.A, v1.D
# Two settings profiles, both chosen on the proxy board (tools/proxy_exp.py, 3 public games):
#   tuned : random search, re-checked on fresh degradation seeds; best proxy score (0.590 vs 0.554
#           for "base"), better in every game; it reports fewer, longer possessions.
#   base  : v1 defaults + team shape; closer possession count.
PROFILES = {
    "tuned": dict(shape=0.15, v1=dict(control_radius=2.0, min_weight=0.05, max_gap=100,
                                      alive_cost_dead=1.0)),
    "base": dict(shape=0.15, v1=dict()),
}
SHAPE_WEIGHT = None  # taken from the profile unless --shape-weight is given
TAG_MODES = ("none", "kickoff", "setpieces", "all")


def filter_tags(tags, mode):
    if mode == "none":
        return []
    if mode == "kickoff":
        return [t for t in tags if t["code"] == "set pieces"
                and t["tags"].get("type", "") == "kick off (start)"]
    if mode == "setpieces":
        return [t for t in tags if t["code"] == "set pieces"]
    return tags


def build(data_dir=None, cache=None, tags_mode="setpieces", use_speed=False,
          shape_weight=None, profile="tuned"):
    prof = PROFILES[profile]
    p1, p2 = dict(v1.P, **prof["v1"]), dict(v2.P2)
    if shape_weight is None:
        shape_weight = prof["shape"]
    print("profile: {}  shape weight {}  v1 overrides {}".format(profile, shape_weight, prof["v1"]))
    meta_path, raw_path, tag_path = require(["atd_meta", "atd_raw", "smart_tagging"], GAME, data_dir)
    n_frames = source_frames("tactical", GAME)
    pitch, fps = GAMES[GAME]["pitch"], GAMES[GAME]["fps"]
    team = team_of_each_track(meta_path)
    blocks = half_video_blocks(GAME)

    if cache and os.path.isfile(cache):
        print("loading cache {}".format(cache))
        stored = np.load(cache)
        ball, players, speed = stored["ball"], stored["players"], stored["speed"]
    else:
        print("reading {} (a minute or two) ...".format(os.path.basename(raw_path)))
        ball, players, speed = v2.load_tracking_with_speed(raw_path, len(team), n_frames, pitch)
        if cache:
            np.savez_compressed(cache, ball=ball, players=players, speed=speed)

    ball_i = v1.interpolate_short_gaps(ball, p1["max_gap"])
    d_home, d_away = v1.team_distances(ball_i, players, team)
    cost = v1.build_costs(ball_i, d_home, d_away, v1.ball_speed(ball_i, fps), pitch, p1)

    if shape_weight > 0:
        ph = np.clip(ts.p_home(players, team, blocks), 1e-3, 1 - 1e-3)
        cost[:, H] += -shape_weight * np.log(ph)
        cost[:, A] += -shape_weight * np.log(1 - ph)
        graded = np.zeros(n_frames, bool)
        for _, f, l in blocks:
            graded[f:l + 1] = True
        print("team shape: says home on {:.1f}% of graded frames".format(
            100 * np.mean(ph[graded] > 0.5)))

    if use_speed:
        crowd = v2.collective_speed(speed, p2["min_players"], p2["speed_window"])
        v2.add_speed_costs(cost, ~np.isnan(ball_i[:, 0]), crowd, p2)

    tags = filter_tags(v2.load_tags(tag_path, fps), tags_mode)
    if tags:
        stats = v2.add_tag_costs(cost, tags, v2.team_lookup(meta_path), ball_i, players, team,
                                 blocks, p2)
        print("smart tagging ({}): {}".format(tags_mode, stats))

    trans = v1.transition_matrix(p1)
    labels = np.full(n_frames, "X", dtype="<U1")
    for half, first, last in blocks:
        state = v1.viterbi(cost[first:last + 1], trans)
        state = v1.bridge_short_dead(state, p1["bridge_dead_frames"])
        labels[first:last + 1] = v1.LABELS[state]
    return labels


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--cache", default=None)
    parser.add_argument("--tags", choices=TAG_MODES, default="setpieces")
    parser.add_argument("--speed", action="store_true")
    parser.add_argument("--shape-weight", type=float, default=None)
    parser.add_argument("--profile", choices=sorted(PROFILES), default="tuned")
    parser.add_argument("--out", default=os.path.join(ROOT, "evaluation", "results",
                                                      "possession_v3.csv"))
    args = parser.parse_args()

    labels = build(args.data_dir, args.cache, args.tags, args.speed, args.shape_weight, args.profile)
    write_submission(args.out, labels)
    graded = labels != "X"
    total = max(int(graded.sum()), 1)
    print("\nwrote {}".format(args.out))
    for name in ("H", "A", "D"):
        count = int(np.sum(labels == name))
        print("  {}  {:>7}  {:5.1f}% of graded".format(name, count, 100.0 * count / total))
    possessions = v1.count_possessions(labels)
    alive = int(np.sum((labels == "H") | (labels == "A")))
    if possessions:
        print("  {} possessions, mean {:.1f} s".format(possessions, alive / 25.0 / possessions))
