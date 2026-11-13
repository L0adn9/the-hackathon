"""possession_v2: v1 plus stoppages and restarts ("laws of the game" layer).

    python starter/possession_v2.py --cache tracking_v2.npz
    python evaluation/validate_submission.py evaluation/results/possession_v2.csv

What v2 adds on top of v1 (starter/possession_v1.py), all as extra per-frame costs before the
same Viterbi decoding:

1. Collective player speed. Every ATD player record carries a speed (m/s). At a stoppage players
   walk; in play they run. The median speed of all tracked players, smoothed over 2 s, is a ball-free
   dead-ball cue -- it still works on the 41% of frames where the ball is missing.
2. Smart Tagging set pieces. Every set piece is a restart, so the moments before it are dead ball and
   the moments after it belong to the restarting team. The tag window is ~5 s around the restart, so
   its middle is taken as the restart instant.
3. Kick-offs. Before the "Kick Off (Start)" tag of each half the ball is dead (the 2nd half is graded
   from before its kick-off).
4. Throw-ins are tagged with Team = N/A. The restarting team is taken from geometry instead: the team
   of the player nearest the ball at the restart (the thrower stands on the ball).
5. Phase tags as a weak prior: attacking phases (Build Up, Progression, ...) favour the tagged team,
   defending phases favour the other team. Deliberately weak: phases are not possessions.

All constants are reasoned starting points, not tuned on any ground truth (see docs/PLAN.md).
"""

import argparse
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from games import GAME, GAMES, half_video_blocks, require, source_frames  # noqa: E402
from baseline import team_of_each_track, team_names, write_submission  # noqa: E402
import possession_v1 as v1  # noqa: E402

H, A, D = v1.H, v1.A, v1.D

P2 = dict(
    speed_window=25,          # +/- frames for smoothing the collective speed (2 s total)
    min_players=6,            # need at least this many tracked players to trust the median
    slow_speed=1.0,           # m/s median: players walking -> stoppage cue
    active_speed=2.2,         # m/s median: players running -> play is alive
    slow_alive_cost=0.25,     # extra per-frame cost of H/A while players walk
    active_dead_cost=0.6,     # extra per-frame cost of D while players run
    unknown_active_dead=0.15,  # extra D cost on unseen-ball frames while players run
    setpiece_pre=50,          # frames before the tag start that are also treated as dead
    setpiece_dead_cost=0.6,   # per-frame cost of H/A before the restart
    setpiece_team_cost=0.8,   # per-frame cost of the non-restarting team after the restart
    setpiece_post=50,         # frames after the tag end the restart team prior still applies
    kickoff_force=5.0,        # per-frame cost of H/A before each half's kick-off
    phase_cost=0.04,          # weak per-frame prior from phase tags
)

ATTACK_CODES = {"build up", "progression", "creating chances", "attacking transition",
                "ball in final third", "ball in the box", "players in the box", "long ball"}
DEFEND_CODES = {"defending in defensive third", "defending in middle third",
                "defending in attacking third", "defensive transition"}


# ---------------------------------------------------------------- tracking with speeds

def load_tracking_with_speed(raw_path, n_slots, n_frames, pitch):
    """Ball (n, 2), players (n, slots, 2) in metres and player speed (n, slots) in m/s."""
    length, width = pitch
    ball = np.full((n_frames, 2), np.nan, dtype=np.float32)
    players = np.full((n_frames, n_slots, 2), np.nan, dtype=np.float32)
    speed = np.full((n_frames, n_slots), np.nan, dtype=np.float32)

    with open(raw_path) as handle:
        for line in handle:
            parts = line.split(":")
            if len(parts) != 3:
                continue
            try:
                frame = int(parts[0]) - 1
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
                values = group.split(",")
                try:
                    players[frame, slot, 0] = float(values[0]) * length
                    players[frame, slot, 1] = float(values[1]) * width
                except (ValueError, IndexError):
                    continue
                try:
                    speed[frame, slot] = float(values[2])
                except (ValueError, IndexError):
                    pass
    return ball, players, speed


def speed_from_positions(players, fps, half_window=5):
    """Fallback when the file carries no usable speeds: centred difference per slot."""
    n = len(players)
    out = np.full(players.shape[:2], np.nan, dtype=np.float32)
    c = np.arange(half_window, n - half_window)
    delta = players[c + half_window] - players[c - half_window]
    out[c] = np.sqrt((delta ** 2).sum(axis=-1)) * fps / (2.0 * half_window)
    return out


def collective_speed(speed, min_players, window):
    """Median player speed per frame, smoothed with a centred NaN-aware moving average."""
    observed = np.sum(~np.isnan(speed), axis=1)
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            median = np.nanmedian(speed, axis=1)
    median[observed < min_players] = np.nan

    valid = ~np.isnan(median)
    values = np.where(valid, median, 0.0)
    csum = np.concatenate(([0.0], np.cumsum(values)))
    ccnt = np.concatenate(([0], np.cumsum(valid)))
    index = np.arange(len(median))
    low = np.maximum(0, index - window)
    high = np.minimum(len(median), index + window + 1)
    count = ccnt[high] - ccnt[low]
    with np.errstate(invalid="ignore", divide="ignore"):
        smooth = (csum[high] - csum[low]) / count
    smooth[count == 0] = np.nan
    return smooth


# ---------------------------------------------------------------- smart tagging

def load_tags(path, fps):
    out = []
    for instance in ET.parse(path).iter("instance"):
        try:
            start_s = float(instance.findtext("start"))
            end_s = float(instance.findtext("end"))
        except (TypeError, ValueError):
            continue
        out.append({
            "code": (instance.findtext("code") or "").strip().lower(),
            "start": int(round(start_s * fps)),
            "end": int(round(end_s * fps)),
            "tags": {(label.findtext("group") or "").strip().lower():
                     (label.findtext("text") or "").strip().lower()
                     for label in instance.findall("label")},
        })
    return out


def team_lookup(meta_path):
    """Map a Smart Tagging Team value to H/A by matching the club names in the ATD metadata."""
    home, away = [(name or "").strip().lower() for name in team_names(meta_path)]

    def side(value):
        value = (value or "").strip().lower()
        if not value or value == "n/a":
            return None
        if value == home:
            return H
        if value == away:
            return A
        words = set(value.split())
        if words & set(home.split()):
            return H
        if words & set(away.split()):
            return A
        return None
    return side


# ---------------------------------------------------------------- costs

def add_speed_costs(cost, ball_known, crowd, p):
    with np.errstate(invalid="ignore"):
        slow = crowd < p["slow_speed"]
        active = crowd > p["active_speed"]
    # At a stoppage someone usually stands on the ball (throw-in taker, free-kick taker), so the
    # proximity evidence keeps saying "alive". Walking players remove the base cost of D instead.
    cost[slow, D] = 0.0
    cost[slow, H] += p["slow_alive_cost"]
    cost[slow, A] += p["slow_alive_cost"]
    cost[active & ball_known, D] += p["active_dead_cost"]
    cost[active & ~ball_known, D] += p["unknown_active_dead"]
    return slow, active


def nearest_team_at(frame, ball, players, team):
    if np.isnan(ball[frame, 0]):
        return None
    delta = players[frame] - ball[frame]
    distance = np.sqrt((delta ** 2).sum(axis=-1))
    if np.all(np.isnan(distance)):
        return None
    return int(team[np.nanargmin(distance)])


def add_tag_costs(cost, tags, side, ball, players, team, blocks, p):
    n = len(cost)
    stats = dict(set_pieces=0, with_team=0, throw_in_geometry=0, kickoffs=0, phases=0)

    for tag in tags:
        code, s, e = tag["code"], tag["start"], tag["end"]
        tag_team = side(tag["tags"].get("team"))
        kind = tag["tags"].get("type", "")

        if code == "set pieces":
            stats["set_pieces"] += 1
            mid = (s + e) // 2
            lo = max(0, s - p["setpiece_pre"])
            cost[lo:mid, H] += p["setpiece_dead_cost"]
            cost[lo:mid, A] += p["setpiece_dead_cost"]

            restart = tag_team
            if restart is None and kind == "throw-in":
                restart = nearest_team_at(min(mid, n - 1), ball, players, team)
                if restart is not None:
                    stats["throw_in_geometry"] += 1
            if restart is not None:
                stats["with_team"] += 1
                other = A if restart == H else H
                cost[mid:min(n, e + p["setpiece_post"]), other] += p["setpiece_team_cost"]

            if kind == "kick off (start)":
                half_name = tag["tags"].get("half", "")
                for half, first, last in blocks:
                    wanted = "1st half" if half == 1 else "2nd half"
                    if half_name == wanted and first <= s <= last:
                        cost[first:s, H] += p["kickoff_force"]
                        cost[first:s, A] += p["kickoff_force"]
                        stats["kickoffs"] += 1
            continue

        if tag_team is None:
            continue
        if code in ATTACK_CODES:
            holder = tag_team
        elif code in DEFEND_CODES:
            holder = A if tag_team == H else H
        else:
            continue
        stats["phases"] += 1
        other = A if holder == H else H
        cost[s:e, other] += p["phase_cost"]
    return stats


# ---------------------------------------------------------------- pipeline

def build(data_dir=None, cache=None, params=None, use_tags=True):
    p1 = dict(v1.P, **(params or {}))
    p2 = dict(P2, **(params or {}))
    names = ["atd_meta", "atd_raw"] + (["smart_tagging"] if use_tags else [])
    paths = require(names, GAME, data_dir)
    meta_path, raw_path = paths[0], paths[1]
    n_frames = source_frames("tactical", GAME)
    pitch = GAMES[GAME]["pitch"]
    fps = GAMES[GAME]["fps"]
    team = team_of_each_track(meta_path)
    blocks = half_video_blocks(GAME)

    if cache and os.path.isfile(cache):
        print("loading cache {}".format(cache))
        stored = np.load(cache)
        ball, players, speed = stored["ball"], stored["players"], stored["speed"]
    else:
        print("reading {} (a minute or two) ...".format(os.path.basename(raw_path)))
        ball, players, speed = load_tracking_with_speed(raw_path, len(team), n_frames, pitch)
        if cache:
            np.savez_compressed(cache, ball=ball, players=players, speed=speed)
            print("saved cache {}  (derived data: keep it out of git)".format(cache))

    usable = np.sum(~np.isnan(speed) & (speed > 0))
    if usable < 0.2 * np.sum(~np.isnan(players[:, :, 0])):
        print("player speeds missing in file -> computing them from positions")
        speed = speed_from_positions(players, fps)
        source = "positions"
    else:
        source = "file"

    ball_raw_known = ~np.isnan(ball[:, 0])
    ball_i = v1.interpolate_short_gaps(ball, p1["max_gap"])
    ball_speed = v1.ball_speed(ball_i, fps)
    d_home, d_away = v1.team_distances(ball_i, players, team)
    cost = v1.build_costs(ball_i, d_home, d_away, ball_speed, pitch, p1)

    crowd = collective_speed(speed, p2["min_players"], p2["speed_window"])
    slow, active = add_speed_costs(cost, ~np.isnan(ball_i[:, 0]), crowd, p2)

    graded = np.zeros(n_frames, dtype=bool)
    for _, first, last in blocks:
        graded[first:last + 1] = True
    print("player speed source: {}; graded frames with walking players {:.1f}%, running {:.1f}%"
          .format(source, 100.0 * slow[graded].mean(), 100.0 * active[graded].mean()))
    print("ball seen on {:.1f}% of graded frames".format(100.0 * ball_raw_known[graded].mean()))

    if use_tags:
        tags = load_tags(paths[2], fps)
        side = team_lookup(meta_path)
        stats = add_tag_costs(cost, tags, side, ball_i, players, team, blocks, p2)
        print("smart tagging: {} tags | set pieces {} (team known {}, throw-in team from "
              "geometry {}) | kick-offs anchored {} | phase priors {}".format(
                  len(tags), stats["set_pieces"], stats["with_team"],
                  stats["throw_in_geometry"], stats["kickoffs"], stats["phases"]))

    trans = v1.transition_matrix(p1)
    labels = np.full(n_frames, "X", dtype="<U1")
    for half, first, last in blocks:
        print("decoding half {} ...".format(half))
        state = v1.viterbi(cost[first:last + 1], trans)
        state = v1.bridge_short_dead(state, p1["bridge_dead_frames"])
        labels[first:last + 1] = v1.LABELS[state]
    return labels


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--out", default=os.path.join(ROOT, "evaluation", "results",
                                                      "possession_v2.csv"))
    parser.add_argument("--cache", default=None, help="optional .npz with parsed tracking")
    parser.add_argument("--no-tags", action="store_true", help="ablation: ignore Smart Tagging")
    args = parser.parse_args()

    labels = build(args.data_dir, args.cache, use_tags=not args.no_tags)
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
