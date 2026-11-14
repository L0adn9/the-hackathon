"""Train the frame classifier on PUBLIC Metrica sample games 1, 2 and 3 (not the competition match).

    git clone https://github.com/metrica-sports/sample-data /tmp/sample-data
    python tools/train_frame_model.py --samples /tmp/sample-data/data              # v4 model
    python tools/train_frame_model.py --samples /tmp/sample-data/data --version v5 # v5 model

Labels come from the sample games' own event data (Metrica, other games -- allowed by RULES.md).
Their tracking is degraded to look like the competition ATD before features are computed.

v4: plain degradation (proxy_board.degrade), no dead-ball cues -> starter/frame_model.pkl
v5: plain + realistic degradation (proxy_board.degrade_realistic), with dead-ball cues
    -> starter/frame_model_v5.pkl

--check runs leave-one-game-out instead: train on two games, score the third with the official
metric, on both the plain and the realistic board. That is how the models and the decoder
settings were chosen.
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


def load_game(samples, game, realistic):
    _, ball, players, team = pb.load_tracking(samples, game)
    truth = pb.truth_from_events(pb.load_events(samples, game), len(ball))
    blocks = pb.blocks_of(truth)
    plain, real = [], []
    for seed in SEEDS:
        b, p, _ = pb.degrade(ball, players, np.random.default_rng(100 * game + seed), truth != pb.X)
        plain.append((b, p))
        if realistic:
            rng = np.random.default_rng(500 + 10 * game + seed)
            real.append(pb.degrade_realistic(ball, players, rng, truth))
    return dict(truth=truth, blocks=blocks, team=team, plain=plain, real=real)


def rows(data, games, boards, dead_cues):
    xs, ys = [], []
    for g in games:
        d = data[g]
        for board in boards:
            for b, p in d[board]:
                x, y = fm.training_rows(b, p, d["team"], d["blocks"], d["truth"], pb.PITCH,
                                        dead_cues=dead_cues)
                xs.append(x)
                ys.append(y)
    return np.vstack(xs), np.concatenate(ys)


def check(data, boards, dead_cues, decode):
    import possession_metrics as pm
    scores = {board: [] for board in ("plain", "real") if data[1][board]}
    for test in GAMES:
        model = fm.fit(*rows(data, [g for g in GAMES if g != test], boards, dead_cues))
        model.dead_cues_ = dead_cues
        d = data[test]
        for board in scores:
            for b, p in d[board]:
                prob = fm.probabilities(model, b, p, d["team"], d["blocks"], pb.PITCH)
                pred = decode(prob, d["blocks"], len(b))
                r = pm.evaluate(d["truth"], pred, bootstrap_n=0)
                scores[board].append(r["spell_f1_mean"])
                print("{:<5} board, test game {}: spellF1 {:.4f}  macroF1 {:.4f}  alive/dead F1 {:.3f}"
                      "  possessions {} (truth {})".format(
                          board, test, r["spell_f1_mean"], r["macro_f1"], r["alive_dead_f1"],
                          r["n_pred_spells"], r["n_gt_spells"]))
    for board, s in scores.items():
        print("{} board: mean spellF1 {:.4f}".format(board, np.mean(s)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True, help="the data/ folder of metrica-sports/sample-data")
    ap.add_argument("--version", choices=("v4", "v5"), default="v4")
    ap.add_argument("--check", action="store_true", help="leave-one-game-out scores, no model saved")
    args = ap.parse_args()

    v5 = args.version == "v5"
    boards = ("plain", "real") if v5 else ("plain",)
    data = {g: load_game(args.samples, g, realistic=v5 or args.check) for g in GAMES}
    if args.check:
        if v5:
            import possession_v5 as pv
        else:
            import possession_v4 as pv
        check(data, boards, v5, pv.decode)
    else:
        model = fm.fit(*rows(data, GAMES, boards, v5))
        path = fm.MODEL_V5 if v5 else fm.MODEL
        fm.save(model, path, dead_cues=v5)
        print("saved {}".format(path))
