"""possession_v1: a first state-machine submission, built on the reference baseline.

    python starter/possession_v1.py                      # reads data/, writes evaluation/results/possession_v1.csv
    python starter/possession_v1.py --cache ball.npz     # first run parses the tracking, later runs reuse it
    python evaluation/validate_submission.py evaluation/results/possession_v1.csv

What it does, in order
----------------------
1. Reads the raw tracking (numpy only) into metres: the ball and every player slot, per frame.
2. Fills SHORT ball gaps (<= max_gap frames) by linear interpolation. Long gaps stay unknown.
3. Per frame, computes the distance from the ball to the nearest home and nearest away player,
   and the ball speed.
4. Turns that into a cost for each of three states (H, A, D). Evidence is weighted: it counts for
   a lot when a player is close to a slow ball (control) and for almost nothing when the ball is
   loose or in flight (possession stays with whoever had it).
5. Decodes each half with Viterbi. Switching state has a price, so a team change needs sustained
   evidence. This is what removes flicker and the invented micro-possessions.
6. Bridges very short dead-ball runs between the same team (an occluded ball is not a stoppage).

Everything outside the two graded halves is X.

Honest status
-------------
All constants in P are reasoned starting points. They were NOT tuned against any ground truth, and
should not be tuned against the public 5 minutes (see EVALUATION.md and RULES.md). Judge changes by
football logic and by sanity of the possession count and mean duration, not by one small score.

Not used yet (obvious next steps): Smart Tagging (restarts and team priors), player speeds,
homography, video.
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

LABELS = np.array(["H", "A", "D"])
H, A, D = 0, 1, 2

# Starting-point constants. Units: metres, frames (25 per second), and cost units.
P = dict(
    max_gap=50,              # interpolate ball gaps up to 2 s
    bounds_margin=0.5,       # ball this far outside the pitch lines counts as out of play
    softness=1.5,            # metres of "closer player" advantage that make a team ~73% likely
    control_radius=3.0,      # evidence fades when nobody is within about this distance
    min_weight=0.1,          # evidence never fades below this fraction
    fast_ball=12.0,          # m/s: a ball this fast is in flight, so control evidence is damped
    still_speed=0.5,         # m/s: a ball this slow with nobody near looks like a stoppage
    still_clearance=2.0,     # metres: "nobody near" for the stoppage cue
    alive_cost_dead=1.5,     # per-frame cost of saying D while the ball is moving or contested
    out_cost_alive=3.0,      # per-frame cost of H/A while the ball is outside the pitch
    unknown_alive_cost=0.04,  # per-frame cost of H/A while the ball is unseen (gap too long)
    switch_team=20.0,        # price of H<->A: needs sustained evidence
    enter_dead=8.0,          # price of alive -> D
    exit_dead=4.0,           # price of D -> alive
    bridge_dead_frames=100,  # D runs this short between the same team become that team (4 s)
)


def load_tracking(raw_path, n_slots, n_frames, pitch):
    """Ball (n, 2) and players (n, slots, 2) in metres, NaN where unobserved."""
    length, width = pitch
    ball = np.full((n_frames, 2), np.nan, dtype=np.float32)
    players = np.full((n_frames, n_slots, 2), np.nan, dtype=np.float32)

    with open(raw_path) as handle:
        for line in handle:
            parts = line.split(":")
            if len(parts) != 3:
                continue
            try:
                frame = int(parts[0]) - 1  # frame_id counts from 1, submissions from 0
            except ValueError:
                continue
            if frame < 0 or frame >= n_frames:
                continue

            try:
                values = parts[2].split(",")
                ball[frame, 0] = float(values[0]) * length
                ball[frame, 1] = float(values[1]) * width
            except (ValueError, IndexError):
                pass

            for slot, group in enumerate(parts[1].split(";")):
                if slot >= n_slots:
                    break
                try:
                    values = group.split(",")
                    players[frame, slot, 0] = float(values[0]) * length
                    players[frame, slot, 1] = float(values[1]) * width
                except (ValueError, IndexError):
                    pass
    return ball, players


def interpolate_short_gaps(ball, max_gap):
    """Linear fill of ball gaps no longer than max_gap frames; longer gaps stay NaN."""
    n = len(ball)
    out = ball.copy()
    known = np.flatnonzero(~np.isnan(ball[:, 0]))
    if len(known) < 2:
        return out

    position = np.arange(n)
    nxt = np.searchsorted(known, position)
    prv = nxt - 1
    inside = (prv >= 0) & (nxt < len(known))
    gap = np.full(n, np.inf)
    gap[inside] = known[nxt[inside]] - known[prv[inside]]

    fill = np.isnan(ball[:, 0]) & (gap <= max_gap)
    for axis in (0, 1):
        out[fill, axis] = np.interp(position[fill], known, ball[known, axis])
    return out


def ball_speed(ball, fps, half_window=12):
    """Ball speed in m/s from a centred difference; NaN where either end is unknown."""
    n = len(ball)
    speed = np.full(n, np.nan, dtype=np.float32)
    centre = np.arange(half_window, n - half_window)
    delta = ball[centre + half_window] - ball[centre - half_window]
    speed[centre] = np.sqrt((delta ** 2).sum(axis=1)) * fps / (2.0 * half_window)
    return speed


def team_distances(ball, players, team):
    """Distance from the ball to the nearest home and nearest away player, per frame."""
    big = 80.0
    delta = players - ball[:, None, :]
    distance = np.sqrt((delta ** 2).sum(axis=-1))
    distance = np.where(np.isnan(distance), np.inf, distance)
    d_home = np.minimum(distance[:, team == 0].min(axis=1), big)
    d_away = np.minimum(distance[:, team == 1].min(axis=1), big)
    return d_home, d_away


def build_costs(ball, d_home, d_away, speed, pitch, p):
    """(n, 3) per-frame cost of the states H, A, D. Lower is more likely."""
    length, width = pitch
    n = len(ball)
    known = ~np.isnan(ball[:, 0])

    m = p["bounds_margin"]
    with np.errstate(invalid="ignore"):
        out_of_pitch = known & ((ball[:, 0] < -m) | (ball[:, 0] > length + m)
                                | (ball[:, 1] < -m) | (ball[:, 1] > width + m))

    d_min = np.minimum(d_home, d_away)
    advantage = np.clip(d_away - d_home, -12.0, 12.0)  # > 0 means the home side is closer
    p_home = np.clip(1.0 / (1.0 + np.exp(-advantage / p["softness"])), 1e-3, 1 - 1e-3)

    weight = np.maximum(np.exp(-(d_min / p["control_radius"]) ** 2), p["min_weight"])
    with np.errstate(invalid="ignore"):
        fast = np.nan_to_num(speed, nan=0.0) > p["fast_ball"]
        still = (speed < p["still_speed"]) & (d_min > p["still_clearance"])
    weight = np.where(fast, weight * 0.3, weight)

    cost = np.zeros((n, 3), dtype=np.float64)
    cost[:, H] = -weight * np.log(p_home)
    cost[:, A] = -weight * np.log(1.0 - p_home)
    cost[:, D] = np.where(still, 0.0, p["alive_cost_dead"])

    cost[out_of_pitch, H] = p["out_cost_alive"]
    cost[out_of_pitch, A] = p["out_cost_alive"]
    cost[out_of_pitch, D] = 0.0

    unknown = ~known
    cost[unknown, H] = p["unknown_alive_cost"]
    cost[unknown, A] = p["unknown_alive_cost"]
    cost[unknown, D] = 0.0
    return cost


def transition_matrix(p):
    """price[from, to]: what it costs to move between states."""
    t = np.zeros((3, 3))
    t[H, A] = t[A, H] = p["switch_team"]
    t[H, D] = t[A, D] = p["enter_dead"]
    t[D, H] = t[D, A] = p["exit_dead"]
    return t


def viterbi(cost, trans):
    """Cheapest state path through (n, 3) costs, paying trans[from, to] on every change."""
    n = len(cost)
    back = np.zeros((n, 3), dtype=np.int8)
    delta = cost[0].copy()
    columns = np.arange(3)
    for t in range(1, n):
        candidate = delta[:, None] + trans
        best = candidate.argmin(axis=0)
        back[t] = best
        delta = candidate[best, columns] + cost[t]

    state = np.empty(n, dtype=np.int8)
    state[-1] = delta.argmin()
    for t in range(n - 1, 0, -1):
        state[t - 1] = back[t, state[t]]
    return state


def bridge_short_dead(state, max_len):
    """A short D run with the same team on both sides is an unseen ball, not a stoppage."""
    state = state.copy()
    n = len(state)
    t = 0
    while t < n:
        if state[t] != D:
            t += 1
            continue
        start = t
        while t < n and state[t] == D:
            t += 1
        end = t  # run is [start, end)
        if start > 0 and end < n and (end - start) <= max_len \
                and state[start - 1] == state[end] and state[start - 1] != D:
            state[start:end] = state[start - 1]
    return state


def count_possessions(labels):
    """Maximal runs of H or A (D is a stoppage, X is ignored)."""
    keep = labels[labels != "X"]
    if len(keep) == 0:
        return 0
    runs = 1 + int(np.sum(keep[1:] != keep[:-1]))
    runs -= int(np.sum((keep[1:] != keep[:-1]) & (keep[1:] == "D")))  # entering D starts no run
    return runs - int(keep[0] == "D")


def build(data_dir=None, cache=None, params=None):
    p = dict(P, **(params or {}))
    meta_path, raw_path = require(["atd_meta", "atd_raw"], GAME, data_dir)
    n_frames = source_frames("tactical", GAME)
    pitch = GAMES[GAME]["pitch"]
    fps = GAMES[GAME]["fps"]
    team = team_of_each_track(meta_path)

    if cache and os.path.isfile(cache):
        print("loading cache {}".format(cache))
        stored = np.load(cache)
        ball, players = stored["ball"], stored["players"]
    else:
        print("reading {} (a minute or two) ...".format(os.path.basename(raw_path)))
        ball, players = load_tracking(raw_path, len(team), n_frames, pitch)
        if cache:
            np.savez_compressed(cache, ball=ball, players=players)
            print("saved cache {}  (derived data: keep it out of git)".format(cache))

    seen = int(np.sum(~np.isnan(ball[:, 0])))
    print("ball seen on {} of {} frames ({:.1f}%)".format(seen, n_frames, 100.0 * seen / n_frames))

    ball = interpolate_short_gaps(ball, p["max_gap"])
    speed = ball_speed(ball, fps)
    d_home, d_away = team_distances(ball, players, team)
    cost = build_costs(ball, d_home, d_away, speed, pitch, p)
    trans = transition_matrix(p)

    labels = np.full(n_frames, "X", dtype="<U1")
    for half, first, last in half_video_blocks(GAME):
        print("decoding half {} (frames {}..{}) ...".format(half, first, last))
        state = viterbi(cost[first:last + 1], trans)
        state = bridge_short_dead(state, p["bridge_dead_frames"])
        labels[first:last + 1] = LABELS[state]
    return labels


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", default=None, help="defaults to ./data or MTAH_DATA_DIR")
    parser.add_argument("--out", default=os.path.join(ROOT, "evaluation", "results",
                                                      "possession_v1.csv"))
    parser.add_argument("--cache", default=None,
                        help="optional .npz to store the parsed tracking, so reruns skip parsing")
    args = parser.parse_args()

    labels = build(args.data_dir, args.cache)
    write_submission(args.out, labels)

    graded = labels != "X"
    total = max(int(graded.sum()), 1)
    print("\nwrote {}".format(args.out))
    for name in ("H", "A", "D"):
        count = int(np.sum(labels == name))
        print("  {}  {:>7}  {:5.1f}% of graded".format(name, count, 100.0 * count / total))
    possessions = count_possessions(labels)
    alive = int(np.sum((labels == "H") | (labels == "A")))
    if possessions:
        print("  {} possessions, mean {:.1f} s".format(possessions, alive / 25.0 / possessions))
    print("\nNext:  python evaluation/validate_submission.py {}".format(args.out))
