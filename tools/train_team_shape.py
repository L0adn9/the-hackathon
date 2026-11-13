"""Train the team-shape sensor on PUBLIC Metrica sample games 1, 2 and 3 (not the competition match).

    git clone https://github.com/metrica-sports/sample-data /tmp/sample-data
    python tools/train_team_shape.py --samples /tmp/sample-data/data

Writes starter/team_shape_weights.json. Deterministic: fixed seeds, plain gradient descent.

Labels come from the sample games' own event data (human-annotated by Metrica, OTHER games: allowed
by RULES.md). Their clean tracking is degraded to resemble the competition ATD before features are
computed, so the sensor is trained on the kind of input it will see.
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "starter"))
import proxy_board as pb  # noqa: E402
import team_shape as ts  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True, help="the data/ folder of metrica-sports/sample-data")
    args = ap.parse_args()

    Z, Y = [], []
    for game in (1, 2, 3):
        _, ball, players, team = pb.load_tracking(args.samples, game)
        truth = pb.truth_from_events(pb.load_events(args.samples, game), len(ball))
        blocks = pb.blocks_of(truth)
        alive = (truth == pb.H) | (truth == pb.A)
        for seed in range(2):
            rng = np.random.default_rng(100 * game + seed)
            _, p, _ = pb.degrade(ball, players, rng, truth != pb.X)
            Z.append(ts.shape_features(p, team, blocks)[alive])
            Y.append((truth[alive] == pb.H).astype(float))
    z, mean, std = ts.standardise(np.vstack(Z))
    y = np.concatenate(Y)
    w = ts.fit_logistic(z, y)
    acc = np.mean((ts.predict_home(z, w) > 0.5) == (y == 1))
    ts.save(w, mean, std, note="trained on metrica-sports/sample-data games 1-3 (degraded), "
            "training frame accuracy {:.3f}".format(acc))
    print("saved {}  (training frame accuracy {:.3f})".format(ts.WEIGHTS, acc))
