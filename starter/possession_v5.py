"""possession_v5: v4 + ball-centred dead-ball cues, trained on realistic degradation.

    python starter/possession_v5.py --cache tracking_v2.npz
    python evaluation/validate_submission.py evaluation/results/possession_v5.csv

What changes from v4
--------------------
1. The proxy board had a shortcut. The sample games blank the ball during most stoppages, so
   "ball missing" almost meant "dead ball" (81-98% of dead frames vs 6-12% of alive frames). In
   the competition ATD the link is much weaker (~71% before tagged set pieces vs ~19% inside
   attacking phases). tools/proxy_board.degrade_realistic() rebuilds the ball during stoppages and
   hides it at competition-like rates; the v5 model is trained on both degradations.
2. Dead-ball cues (frame_features.dead_ball_features): where the ball was last seen before it
   vanished and where it reappears (distance to the lines, the corners, the centre spot, the goal
   area), whether it was heading out, its speed when last / next seen, how long it has been
   still, how soon it is kicked, players 8-11 m from it (a wall), crowding around it, and
   collective player speed over the past / next 3 s.
3. Decoder settings re-chosen for the new model (cheaper team switch).

Leave-one-game-out (python tools/train_frame_model.py --samples ... --version v5 --check):
realistic board v3 0.473, v4 0.566, v5 0.681 (alive/dead F1 ~0.85 -> ~0.95), better than v4 on
every game; plain board v4 0.699, v5 0.717 (better on games 1 and 3, slightly worse on game 2).
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
import frame_model as fm  # noqa: E402

# Chosen on both proxy boards; every setting in the 72-point grid is within 0.017 of the best.
DECODE = dict(switch_team=10.0, enter_dead=15.0, exit_dead=10.0, bridge_dead_frames=50)
TAG_BALL_GAP = 100  # interpolation used only to find the thrower at throw-ins (same as v3)


def decode(prob, blocks, n, extra_cost=None):
    cost = -np.log(np.clip(prob, 1e-4, 1.0))
    if extra_cost is not None:
        cost = cost + extra_cost
    p = dict(v1.P, **DECODE)
    trans = v1.transition_matrix(p)
    out = np.full(n, 3, dtype=np.int8)
    for _, first, last in blocks:
        state = v1.viterbi(cost[first:last + 1], trans)
        out[first:last + 1] = v1.bridge_short_dead(state, p["bridge_dead_frames"])
    return out


def build(data_dir=None, cache=None, use_tags=True):
    meta_path, raw_path, tag_path = require(["atd_meta", "atd_raw", "smart_tagging"], GAME, data_dir)
    n_frames = source_frames("tactical", GAME)
    pitch, fps = GAMES[GAME]["pitch"], GAMES[GAME]["fps"]
    team = team_of_each_track(meta_path)
    blocks = half_video_blocks(GAME)

    if cache and os.path.isfile(cache):
        print("loading cache {}".format(cache))
        stored = np.load(cache)
        ball, players = stored["ball"], stored["players"]
    else:
        print("reading {} (a minute or two) ...".format(os.path.basename(raw_path)))
        ball, players, speed = v2.load_tracking_with_speed(raw_path, len(team), n_frames, pitch)
        if cache:
            np.savez_compressed(cache, ball=ball, players=players, speed=speed)

    model = fm.load(fm.MODEL_V5)
    prob = fm.probabilities(model, ball, players, team, blocks, pitch)

    extra = np.zeros((n_frames, 3))
    if use_tags:
        tags = [t for t in v2.load_tags(tag_path, fps) if t["code"] == "set pieces"]
        ball_i = v1.interpolate_short_gaps(ball, TAG_BALL_GAP)
        stats = v2.add_tag_costs(extra, tags, v2.team_lookup(meta_path), ball_i, players, team,
                                 blocks, dict(v2.P2))
        print("smart tagging (set pieces): {}".format(stats))

    state = decode(prob, blocks, n_frames, extra)
    labels = np.full(n_frames, "X", dtype="<U1")
    for _, first, last in blocks:
        labels[first:last + 1] = v1.LABELS[state[first:last + 1]]
    return labels


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--cache", default=None)
    parser.add_argument("--no-tags", action="store_true", help="skip the Smart Tagging layer")
    parser.add_argument("--out", default=os.path.join(ROOT, "evaluation", "results",
                                                      "possession_v5.csv"))
    args = parser.parse_args()

    labels = build(args.data_dir, args.cache, not args.no_tags)
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
