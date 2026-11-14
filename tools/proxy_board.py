"""Proxy leaderboard on PUBLIC Metrica sample games (not the competition match).

1. Possession truth derived from each sample game's event data (other games: allowed).
2. Tracking degraded to look like the competition ATD (ball missing ~41% in bursts, jitter,
   player dropouts, noisy speeds).
3. Our pipelines run on it and are scored with the official metric library.

    python proxy_board.py --repo /path/to/the-hackathon --samples /path/to/sample-data/data
"""

import argparse
import itertools
import os
import sys

import numpy as np
import pandas as pd

H, A, D, X = 0, 1, 2, 3
CONTROLLED = {"PASS", "RECOVERY", "SHOT", "SET PIECE", "CARRY"}
PITCH = (105.0, 68.0)
FPS = 25


# ------------------------------------------------------------------ data

def load_tracking(folder, game):
    if game == 3:
        return _load_tracking_epts(folder)

    def read(side):
        path = os.path.join(folder, "Sample_Game_{0}".format(game),
                            "Sample_Game_{0}_RawTrackingData_{1}_Team.csv".format(game, side))
        raw = pd.read_csv(path, skiprows=2, low_memory=False)
        values = raw.iloc[:, 3:].to_numpy(dtype=np.float64)
        pairs = values.reshape(len(raw), -1, 2)
        return raw["Period"].to_numpy(), pairs
    period, home = read("Home")
    _, away = read("Away")
    ball = home[:, -1, :]
    players = np.concatenate([home[:, :-1, :], away[:, :-1, :]], axis=1)
    team = np.array([0] * (home.shape[1] - 1) + [1] * (away.shape[1] - 1))
    scale = np.array(PITCH)
    return period, ball * scale, players * scale, team


def _load_tracking_epts(folder):
    """Sample game 3 is in the FIFA EPTS format (same family as the competition ATD)."""
    from kloppy import metrica
    base = os.path.join(folder, "Sample_Game_3", "Sample_Game_3_")
    ds = metrica.load_tracking_epts(meta_data=base + "metadata.xml", raw_data=base + "tracking.txt")
    df = ds.to_df()
    n = int(df["frame_id"].max())
    idx = df["frame_id"].to_numpy() - 1
    period = np.zeros(n, dtype=int)
    period[idx] = df["period_id"].to_numpy()
    ball = np.full((n, 2), np.nan)
    ball[idx] = df[["ball_x", "ball_y"]].to_numpy(dtype=float)
    ids, team = [], []
    for k, t in enumerate(ds.metadata.teams):  # first team is home
        for p in t.players:
            if p.player_id + "_x" in df.columns:
                ids.append(p.player_id)
                team.append(k)
    players = np.full((n, len(ids), 2), np.nan)
    for j, pid in enumerate(ids):
        players[idx, j, 0] = df[pid + "_x"].to_numpy(dtype=float)
        players[idx, j, 1] = df[pid + "_y"].to_numpy(dtype=float)
    scale = np.array(PITCH)
    return period, ball * scale, players * scale, np.array(team)


def _load_events_json(folder):
    import json
    path = os.path.join(folder, "Sample_Game_3", "Sample_Game_3_events.json")
    rows = []
    for e in json.load(open(path))["data"]:
        sub = e.get("subtypes")
        if isinstance(sub, dict):
            sub = sub.get("name", "")
        elif isinstance(sub, list):
            sub = "-".join(s.get("name", "") for s in sub)
        rows.append({"Team": "Home" if e["team"]["name"] == "Team A" else "Away",
                     "Type": e["type"]["name"], "Subtype": sub or "", "Period": e["period"],
                     "Start Frame": e["start"]["frame"], "End Frame": e["end"]["frame"] or 0})
    return pd.DataFrame(rows)


def load_events(folder, game):
    if game == 3:
        return _load_events_json(folder)
    path = os.path.join(folder, "Sample_Game_{0}".format(game),
                        "Sample_Game_{0}_RawEventsData.csv".format(game))
    ev = pd.read_csv(path)
    ev["Subtype"] = ev["Subtype"].fillna("").astype(str)
    return ev


def truth_from_events(ev, n):
    """H/A/D/X per frame. Possession = last team with a controlled action; dead from ball-out,
    foul or goal until the next restart."""
    labels = np.full(n, X, dtype=np.int8)
    for period in sorted(ev["Period"].unique()):
        e = ev[ev["Period"] == period].sort_values("Start Frame")
        first = int(e["Start Frame"].min()) - 1
        last = int(max(e["Start Frame"].max(), e["End Frame"].max())) - 1
        state, since = D, first
        for _, row in e.iterrows():
            kind, sub = row["Type"], row["Subtype"]
            side = H if row["Team"] == "Home" else A
            s = int(row["Start Frame"]) - 1
            end = int(row["End Frame"]) - 1 if row["End Frame"] > 0 else s
            if state == D:
                # only a restart ends a stoppage (goal kicks are logged as passes)
                if kind in ("SET PIECE", "PASS"):
                    labels[since:s] = D
                    state, since = side, s
                continue
            if kind in CONTROLLED and side != state:
                labels[since:s] = state
                state, since = side, s
            if kind == "BALL OUT" or (kind == "SHOT" and ("OUT" in sub or "GOAL" in sub)):
                stop = max(end, s)
                labels[since:stop] = state
                state, since = D, stop
            elif kind == "FAULT RECEIVED":
                labels[since:s] = state
                state, since = D, s
        labels[since:last + 1] = state
    return labels


def blocks_of(labels):
    """Contiguous graded blocks (one per period)."""
    graded = labels != X
    edges = np.flatnonzero(np.diff(np.concatenate(([0], graded.astype(int), [0]))))
    return [(i + 1, edges[2 * i], edges[2 * i + 1] - 1) for i in range(len(edges) // 2)]


# ------------------------------------------------------------------ degradation

def bursts(n, rng, share, mean_len):
    mask = np.zeros(n, dtype=bool)
    mean_gap = mean_len * (1 - share) / share
    pos = int(rng.exponential(mean_gap))
    while pos < n:
        length = max(3, int(rng.exponential(mean_len)))
        mask[pos:pos + length] = True
        pos += length + max(1, int(rng.exponential(mean_gap)))
    return mask


def degrade(ball, players, rng, graded, target=0.41):
    n, slots = players.shape[:2]
    already = np.mean(np.isnan(ball[graded, 0]))
    extra = max(0.0, (target - already) / max(1e-6, 1 - already))
    ball = ball + rng.normal(0, 0.3, ball.shape)
    if extra > 0:
        ball[bursts(n, rng, extra, 40)] = np.nan
    players = players + rng.normal(0, 0.2, players.shape)
    for k in range(slots):
        players[bursts(n, rng, 0.07, 50), k] = np.nan
    c = np.arange(5, n - 5)
    speed = np.full((n, slots), np.nan)
    delta = players[c + 5] - players[c - 5]
    speed[c] = np.sqrt((delta ** 2).sum(-1)) * FPS / 10.0
    speed = np.clip(speed + rng.normal(0, 0.3, speed.shape), 0, None)
    return ball.astype(np.float32), players.astype(np.float32), speed.astype(np.float32)


def fill_dead_ball(ball, truth, hold=50):
    """Give the ball a position where the sample data has none.

    The sample games blank the ball during most stoppages (81-98% of dead frames, against 6-12%
    of alive frames), so on the plain degradation above "ball missing" almost means "dead ball".
    The competition ATD is not like that (we measured ~71% missing in the 5 s before tagged set
    pieces and ~19% inside attacking phases). Here the ball moves in a straight line from where
    it was last seen to where it is next seen, and sits still on the restart spot for the last
    `hold` frames of each stoppage.
    """
    out = ball.astype(np.float64).copy()
    known = np.flatnonzero(~np.isnan(out[:, 0]))
    pos = np.arange(len(out))
    for k in (0, 1):
        out[:, k] = np.interp(pos, known, out[known, k])
    dead = truth == D
    edges = np.flatnonzero(np.diff(np.concatenate(([0], dead.astype(int), [0]))))
    for s, e in zip(edges[0::2], edges[1::2]):
        h = min(hold, e - s)
        out[e - h:e] = out[min(e, len(out) - 1)]
    return out


def degrade_realistic(ball, players, rng, truth, dead_miss=0.70, alive_miss=0.25):
    """Like degrade(), but the ball goes missing at competition-like rates by state:
    ~70% of dead frames and ~25% of alive frames, in long bursts (~16 s dead, ~6 s alive)."""
    n, slots = players.shape[:2]
    graded = truth != X
    full = fill_dead_ball(ball, truth) + rng.normal(0, 0.3, (n, 2))
    missing = bursts(n, rng, dead_miss, 400) & graded & (truth == D)
    alive = truth < D
    already = np.isnan(ball[:, 0]) & alive
    extra = max(0.0, (alive_miss - already[alive].mean()) / (1 - already[alive].mean()))
    missing |= (bursts(n, rng, extra, 150) & graded & alive) | already
    full[missing] = np.nan
    players = players + rng.normal(0, 0.2, players.shape)
    for k in range(slots):
        players[bursts(n, rng, 0.07, 50), k] = np.nan
    return full.astype(np.float32), players.astype(np.float32)


# ------------------------------------------------------------------ pipelines

def run_pipeline(v1, v2, ball, players, speed, team, blocks, p1, p2, use_speed):
    ball_i = v1.interpolate_short_gaps(ball, p1["max_gap"])
    bspeed = v1.ball_speed(ball_i, FPS)
    d_home, d_away = v1.team_distances(ball_i, players, team)
    cost = v1.build_costs(ball_i, d_home, d_away, bspeed, PITCH, p1)
    if use_speed:
        crowd = v2.collective_speed(speed, p2["min_players"], p2["speed_window"])
        v2.add_speed_costs(cost, ~np.isnan(ball_i[:, 0]), crowd, p2)
    trans = v1.transition_matrix(p1)
    out = np.full(len(ball), X, dtype=np.int8)
    for _, first, last in blocks:
        state = v1.viterbi(cost[first:last + 1], trans)
        out[first:last + 1] = v1.bridge_short_dead(state, p1["bridge_dead_frames"])
    return out


def baseline50(ball, players, team, blocks):
    delta = players - ball[:, None, :]
    dist = np.sqrt((delta ** 2).sum(-1))
    dist = np.where(np.isnan(dist), np.inf, dist)
    seen = ~np.isnan(ball[:, 0]) & np.isfinite(dist.min(1))
    codes = np.where(seen, team[dist.argmin(1)], -1)
    hc = np.concatenate(([0], np.cumsum(codes == 0)))
    ac = np.concatenate(([0], np.cumsum(codes == 1)))
    i = np.arange(len(codes))
    lo, hi = np.maximum(0, i - 50), np.minimum(len(codes), i + 51)
    h, a = hc[hi] - hc[lo], ac[hi] - ac[lo]
    out = np.where(h + a > 0, np.where(h >= a, H, A), D).astype(np.int8)
    mask = np.full(len(codes), X, dtype=np.int8)
    for _, f, l in blocks:
        mask[f:l + 1] = out[f:l + 1]
    return mask


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--samples", required=True)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--sweep", action="store_true")
    args = ap.parse_args()

    sys.path.insert(0, args.repo)
    sys.path.insert(0, os.path.join(args.repo, "starter"))
    sys.path.insert(0, os.path.join(args.repo, "evaluation"))
    import possession_v1 as v1
    import possession_v2 as v2
    import possession_metrics as pm

    cases = []
    for game in (1, 2):
        period, ball, players, team = load_tracking(args.samples, game)
        ev = load_events(args.samples, game)
        truth = truth_from_events(ev, len(ball))
        blocks = blocks_of(truth)
        g = truth[truth != X]
        n_poss = pm.possession_stats(truth, pm.valid_runs(truth))
        print("game {}: {} graded frames, dead {:.1f}%, truth stats {}".format(
            game, len(g), 100 * np.mean(g == D), n_poss))
        for seed in range(args.seeds):
            rng = np.random.default_rng(100 * game + seed)
            b, p, s = degrade(ball, players, rng, truth != X)
            miss = np.mean(np.isnan(b[truth != X, 0]))
            print("  seed {}: ball missing on {:.1f}% of graded frames".format(seed, 100 * miss))
            cases.append((game, seed, truth, blocks, b, p, s, team))

    def score(name, fn):
        sp, mf, poss = [], [], []
        for game, seed, truth, blocks, b, p, s, team in cases:
            pred = fn(b, p, s, team, blocks)
            r = pm.evaluate(truth, pred)
            sp.append(r["spell_f1_mean"]); mf.append(r["macro_f1"]); poss.append(r["n_pred_spells"])
        gt = [pm.evaluate(c[2], c[2])["n_gt_spells"] for c in cases[:1]]
        print("{:<46} spellF1 {:.4f} (min {:.4f})  macroF1 {:.4f}  poss/game {:.0f}".format(
            name, np.mean(sp), np.min(sp), np.mean(mf), np.mean(poss)))
        return np.mean(sp)

    print()
    score("baseline_smooth50", lambda b, p, s, t, bl: baseline50(b, p, t, bl))
    score("v1 (defaults)", lambda b, p, s, t, bl: run_pipeline(v1, v2, b, p, s, t, bl, v1.P, v2.P2, False))
    score("v2 speed layer (defaults)", lambda b, p, s, t, bl: run_pipeline(v1, v2, b, p, s, t, bl, v1.P, v2.P2, True))

    if args.sweep:
        print("\nsweep (v2 speed layer):")
        grid = {"switch_team": [10, 20, 40, 80], "enter_dead": [4, 8, 16],
                "slow_speed": [0.8, 1.0, 1.3], "active_speed": [1.8, 2.2, 2.8]}
        results = []
        for values in itertools.product(*grid.values()):
            over = dict(zip(grid.keys(), values))
            p1 = dict(v1.P, **{k: v for k, v in over.items() if k in v1.P})
            p2 = dict(v2.P2, **{k: v for k, v in over.items() if k in v2.P2})
            sp = score(str(over), lambda b, p, s, t, bl: run_pipeline(v1, v2, b, p, s, t, bl, p1, p2, True))
            results.append((sp, over))
        results.sort(key=lambda r: -r[0])
        print("\ntop 5:")
        for sp, over in results[:5]:
            print("  {:.4f}  {}".format(sp, over))


if __name__ == "__main__":
    main()
