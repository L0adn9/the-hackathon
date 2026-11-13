"""possession_v4: a learned frame classifier + Viterbi + Smart Tagging restarts.

    python starter/possession_v4.py --cache tracking_v2.npz
    python evaluation/validate_submission.py evaluation/results/possession_v4.csv

What changes from v3
--------------------
v3 turned a handful of hand-set rules (nearest player, ball speed, ball out) into per-frame costs.
v4 replaces those rules with a gradient-boosted classifier (starter/frame_model.py) that reads
~50 features per frame (starter/frame_features.py): distances from the ball to both teams, how
many players are near it, ball speed and position, how long the ball has been unseen, collective
player speed, the team-shape features from v3, and the same quantities averaged over the
previous and next 0.5 s / 1 s / 3 s. The time windows are what let it judge a frame where the ball
is missing from what happened just before and just after.

It outputs P(H), P(A), P(D) per frame. The cost of a state is -log P, and the same Viterbi
decoding as v1 turns that into whole possessions. The v2 Smart Tagging set-piece layer is then
added on top, as in v3.

Training and settings: public Metrica sample games only (tools/train_frame_model.py). On the
proxy board (leave-one-game-out, a game is never scored by a model that saw it) v4 scores
spellF1 ~0.70 against 0.595 for v3, better on all three games, and 0.61 vs 0.48 when the test
tracking is degraded harder than the training tracking.
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

H, A, D = v1.H, v1.A, v1.D

# Decoder settings, chosen on the proxy board. The score there is flat around them
# (0.686 .. 0.708 over an 81-point grid), so a middle value was taken, not the single best.
DECODE = dict(switch_team=20.0, enter_dead=15.0, exit_dead=10.0, bridge_dead_frames=50)
TAG_BALL_GAP = 100  # interpolation used only to find the thrower at throw-ins (same as v3)


def costs(prob):
    return -np.log(np.clip(prob, 1e-4, 1.0))


def decode(prob, blocks, n, extra_cost=None):
    cost = costs(prob)
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

    model = fm.load()
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
                                                      "possession_v4.csv"))
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
