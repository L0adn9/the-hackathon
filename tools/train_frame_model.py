"""Train the frame classifier on PUBLIC Metrica sample games 1, 2 and 3 (not the competition match).

    git clone https://github.com/metrica-sports/sample-data /tmp/sample-data
    python tools/train_frame_model.py --samples /tmp/sample-data/data

Writes starter/frame_model.pkl. Labels come from the sample games' own event data (Metrica,
other games -- allowed by RULES.md). Their tracking is degraded to look like the competition ATD
(same degradation as the proxy board) before features are computed.

--check runs leave-one-game-out instead: train on two games, score the third with the official
metric, for each game. That is how the model and the decoder settings in possession_v4.py were
chosen.
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (HERE, ROOT, os.path.join(ROOT, "starter"), os.path.join(ROOT, "evaluation")):
    sys.path.insert(0, p)
import proxy_board as pb  # noqa: E402
import frame_model as fm  # noqa: E402

GAMES = (1, 2, 3)
SEEDS = (0, 1)


def load_game(samples, game):
    _, ball, players, team = pb.load_tracking(samples, game)
    truth = pb.truth_from_events(pb.load_events(samples, game), len(ball))
    blocks = pb.blocks_of(truth)
    runs = []
    for seed in SEEDS:
        b, p, _ = pb.degrade(ball, players, np.random.default_rng(100 * game + seed), truth != pb.X)
        runs.append((b, p))
    return dict(truth=truth, blocks=blocks, team=team, runs=runs)


def rows(data, games):
    xs, ys = [], []
    for g in games:
        d = data[g]
        for b, p in d["runs"]:
            x, y = fm.training_rows(b, p, d["team"], d["blocks"], d["truth"], pb.PITCH)
            xs.append(x)
            ys.append(y)
    return np.vstack(xs), np.concatenate(ys)


def check(data):
    import possession_metrics as pm
    import possession_v4 as v4
    scores = []
    for test in GAMES:
        model = fm.fit(*rows(data, [g for g in GAMES if g != test]))
        d = data[test]
        for b, p in d["runs"]:
            prob = fm.probabilities(model, b, p, d["team"], d["blocks"], pb.PITCH)
            pred = v4.decode(prob, d["blocks"], len(b))
            r = pm.evaluate(d["truth"], pred, bootstrap_n=0)
            scores.append(r["spell_f1_mean"])
            print("test game {}: spellF1 {:.4f}  macroF1 {:.4f}  possessions {} (truth {})".format(
                test, r["spell_f1_mean"], r["macro_f1"], r["n_pred_spells"], r["n_gt_spells"]))
    print("mean spellF1 {:.4f}".format(np.mean(scores)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True, help="the data/ folder of metrica-sports/sample-data")
    ap.add_argument("--check", action="store_true", help="leave-one-game-out scores, no model saved")
    args = ap.parse_args()

    data = {g: load_game(args.samples, g) for g in GAMES}
    if args.check:
        check(data)
    else:
        x, y = rows(data, GAMES)
        model = fm.fit(x, y)
        fm.save(model)
        acc = np.mean(model.predict(x) == y)
        print("saved {}  ({} rows, training frame accuracy {:.3f})".format(fm.MODEL, len(y), acc))
