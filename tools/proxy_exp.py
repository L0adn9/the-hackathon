"""Proxy experiments over 3 public games, leave-one-game-out shape weights, random search."""
import json, os, sys, time
import numpy as np

REPO, SAMPLES, MODE = sys.argv[1], sys.argv[2], sys.argv[3]
for p in (REPO, os.path.join(REPO, "starter"), os.path.join(REPO, "evaluation"), os.path.join(REPO, "tools")):
    sys.path.insert(0, p)
import possession_v1 as v1, possession_v2 as v2, possession_metrics as pm, team_shape as ts, proxy_board as pb

GAMES = (1, 2, 3)
SEEDS = [int(x) for x in os.environ.get("SEEDS", "0,1").split(",")]
cases = {}
for g in GAMES:
    _, ball, players, team = pb.load_tracking(SAMPLES, g)
    truth = pb.truth_from_events(pb.load_events(SAMPLES, g), len(ball))
    blocks = pb.blocks_of(truth)
    lst = []
    for seed in SEEDS:
        b, p, s = pb.degrade(ball, players, np.random.default_rng(100 * g + seed), truth != pb.X)
        lst.append(dict(b=b, p=p, s=s, f=ts.shape_features(p, team, blocks)))
    cases[g] = dict(truth=truth, blocks=blocks, team=team, cases=lst)

W = {}
for test in GAMES:
    Z, Y = [], []
    for g in GAMES:
        if g == test:
            continue
        t = cases[g]["truth"]; alive = (t == 0) | (t == 1)
        for c in cases[g]["cases"]:
            Z.append(c["f"][alive]); Y.append((t[alive] == 0).astype(float))
    z, m, sd = ts.standardise(np.vstack(Z))
    W[test] = (ts.fit_logistic(z, np.concatenate(Y)), m, sd)
    for c in cases[test]["cases"]:
        c["ph"] = np.clip(ts.predict_home(ts.standardise(c["f"], m, sd)[0], W[test][0]), 1e-3, 1 - 1e-3)


def control_signal(d_own, d_other, radius, margin, window):
    near = ((d_own < radius) & (d_own + margin < d_other)).astype(float)
    cs = np.concatenate(([0.0], np.cumsum(near)))
    i = np.arange(len(near)); lo = np.maximum(0, i - window); hi = np.minimum(len(near), i + window + 1)
    return (cs[hi] - cs[lo]) / (hi - lo)


def run(c, team, blocks, cfg):
    p1 = dict(v1.P, **{k: v for k, v in cfg.items() if k in v1.P})
    bi = v1.interpolate_short_gaps(c["b"], int(p1["max_gap"]))
    dh, da = v1.team_distances(bi, c["p"], team)
    cost = v1.build_costs(bi, dh, da, v1.ball_speed(bi, 25), pb.PITCH, p1)
    w = cfg.get("shape", 0.15)
    if w > 0:
        cost[:, 0] += -w * np.log(c["ph"]); cost[:, 1] += -w * np.log(1 - c["ph"])
    k = cfg.get("control", 0.0)
    if k > 0:
        ch = control_signal(dh, da, cfg.get("ctrl_radius", 1.5), 1.0, 6)
        ca = control_signal(da, dh, cfg.get("ctrl_radius", 1.5), 1.0, 6)
        cost[:, 1] += k * ch; cost[:, 0] += k * ca
    out = np.full(len(bi), 3, dtype=np.int8); trans = v1.transition_matrix(p1)
    for _, f, l in blocks:
        out[f:l + 1] = v1.bridge_short_dead(v1.viterbi(cost[f:l + 1], trans), int(p1["bridge_dead_frames"]))
    return out


def evaluate(cfg):
    per_game = {}
    for g in GAMES:
        d = cases[g]; sp = []
        for c in d["cases"]:
            sp.append(pm.evaluate(d["truth"], run(c, d["team"], d["blocks"], cfg))["spell_f1_mean"])
        per_game[g] = float(np.mean(sp))
    return float(np.mean(list(per_game.values()))), per_game


if MODE == "base":
    for name, cfg in [("v1", {"shape": 0.0}), ("v3 (shape .15)", {}),
                      ("v3 + control .2", {"control": 0.2}), ("v3 + control .5", {"control": 0.5}),
                      ("v3 + control 1.0", {"control": 1.0})]:
        m, pg = evaluate(cfg)
        print("{:<22} mean {:.4f}  per game {}".format(name, m, {k: round(v, 4) for k, v in pg.items()}), flush=True)

if MODE == "search":
    rng = np.random.default_rng(7)
    space = dict(softness=[1.0, 1.5, 2.5], control_radius=[2.0, 3.0, 4.5], min_weight=[0.05, 0.1, 0.2],
                 max_gap=[25, 50, 100], fast_ball=[9.0, 12.0, 16.0], switch_team=[16.0, 20.0, 30.0],
                 enter_dead=[6.0, 8.0, 11.0], exit_dead=[2.0, 4.0, 6.0], alive_cost_dead=[1.0, 1.5, 2.5],
                 unknown_alive_cost=[0.02, 0.04, 0.08], bridge_dead_frames=[50, 100, 150],
                 shape=[0.1, 0.15, 0.25], control=[0.0, 0.2, 0.5])
    base_m, base_pg = evaluate({})
    print("BASE v3 {:.4f} {}".format(base_m, base_pg), flush=True)
    results = []
    t0 = time.time()
    for it in range(int(sys.argv[4]) if len(sys.argv) > 4 else 60):
        cfg = {k: v[rng.integers(len(v))] for k, v in space.items()}
        m, pg = evaluate(cfg)
        worst_gain = min(pg[g] - base_pg[g] for g in GAMES)
        results.append((m, worst_gain, cfg, pg))
        print("{:3d} {:.4f} worst_gain {:+.4f} {} {}".format(it, m, worst_gain, json.dumps({k: float(v) for k, v in cfg.items()}), {k: round(v, 4) for k, v in pg.items()}), flush=True)
    results.sort(key=lambda r: -r[0])
    print("\nTOP (by mean):")
    for m, wg, cfg, pg in results[:8]:
        print("{:.4f} worst_gain {:+.4f} {}".format(m, wg, json.dumps({k: float(v) for k, v in cfg.items()})))
    print("elapsed {:.0f}s".format(time.time() - t0))

if MODE == "confirm":
    cands = {
      "v3 defaults": {},
      "A: dead 1.0 + switch 30": dict(alive_cost_dead=1.0, switch_team=30.0),
      "B: A + min_w .05 + gap 100": dict(alive_cost_dead=1.0, switch_team=30.0, min_weight=0.05, max_gap=100),
      "C: B + exit 2 + soft 2.5 + fast 9 + bridge 50": dict(alive_cost_dead=1.0, switch_team=30.0, min_weight=0.05, max_gap=100, exit_dead=2.0, softness=2.5, fast_ball=9.0, bridge_dead_frames=50, shape=0.1),
      "D: search #37": dict(softness=1.5, control_radius=2.0, min_weight=0.05, max_gap=100, fast_ball=12.0, switch_team=30.0, enter_dead=8.0, exit_dead=4.0, alive_cost_dead=1.0, unknown_alive_cost=0.04, bridge_dead_frames=100, shape=0.25),
      "E: B + shape .2": dict(alive_cost_dead=1.0, switch_team=30.0, min_weight=0.05, max_gap=100, shape=0.2),
    }
    for name, cfg in cands.items():
        m, pg = evaluate(cfg)
        print("{:<48} mean {:.4f}  {}".format(name, m, {k: round(v, 4) for k, v in pg.items()}), flush=True)
