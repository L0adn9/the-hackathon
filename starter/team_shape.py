"""Team-shape possession sensor: who has the ball, judged from the 20 players, not the ball.

The team in possession pushes forward and stretches; the team without it drops and compacts.
These features exist on every frame, including the ~41% where the ball is not tracked.

The weights of the small logistic model are learned ONLY on public Metrica sample games
(https://github.com/metrica-sports/sample-data), never on the competition match. See
tools/train_team_shape.py, which writes starter/team_shape_weights.json.
"""

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WEIGHTS = os.path.join(HERE, "team_shape_weights.json")


def _nan_stat(fn, values, axis):
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return fn(values, axis=axis)


def _smooth(x, window):
    """Centred NaN-aware moving average over +/- window frames."""
    valid = ~np.isnan(x)
    v = np.where(valid, x, 0.0)
    cs = np.concatenate(([0.0], np.cumsum(v)))
    cc = np.concatenate(([0], np.cumsum(valid)))
    i = np.arange(len(x))
    lo, hi = np.maximum(0, i - window), np.minimum(len(x), i + window + 1)
    n = cc[hi] - cc[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        out = (cs[hi] - cs[lo]) / n
    out[n == 0] = np.nan
    return out


def attack_directions(players, team, first, last):
    """+1 if the team attacks towards larger x in this block, -1 otherwise.

    A team's players spend the half, on average, nearer their own goal than the opponent does.
    """
    x = players[first:last + 1, :, 0]
    mean_home = np.nanmean(x[:, team == 0])
    mean_away = np.nanmean(x[:, team == 1])
    home_dir = 1.0 if mean_home < mean_away else -1.0
    return home_dir, -home_dir


def shape_features(players, team, blocks, fps=25, window=25):
    """(n, k) feature matrix, home-minus-away, oriented to each team's attacking direction."""
    n = len(players)
    feats = np.full((n, 5), np.nan)
    for _, first, last in blocks:
        sl = slice(first, last + 1)
        dirs = attack_directions(players, team, first, last)
        per_team = []
        for k in (0, 1):
            pts = players[sl][:, team == k, :]
            cx = _nan_stat(np.nanmean, pts[:, :, 0], 1) * dirs[k]
            sx = _nan_stat(np.nanstd, pts[:, :, 0], 1)
            sy = _nan_stat(np.nanstd, pts[:, :, 1], 1)
            vx = np.full(len(cx), np.nan)
            h = 12
            vx[h:-h] = (cx[2 * h:] - cx[:-2 * h]) * fps / (2.0 * h)
            per_team.append((cx, sx, sy, vx))
        (cx0, sx0, sy0, vx0), (cx1, sx1, sy1, vx1) = per_team
        block = np.stack([
            vx0 - vx1,                       # who is moving forward
            sx0 - sx1,                       # who is long
            sy0 - sy1,                       # who is wide
            (sx0 * sy0 - sx1 * sy1) / 50.0,  # who occupies more area
            (cx0 - cx1) / 10.0,              # territory: who is further up the pitch
        ], axis=1)
        for j in range(block.shape[1]):
            feats[sl, j] = _smooth(block[:, j], window)
    return feats


def standardise(feats, mean=None, std=None):
    # No centring and (below) no intercept: features are home-minus-away, so swapping the teams
    # must flip the prediction. A learned offset would only encode which side the TRAINING games'
    # home team happened to have the ball more.
    if mean is None:
        mean = np.zeros(feats.shape[1])
        std = np.sqrt(np.nanmean(feats ** 2, axis=0)) + 1e-6
    z = (feats - mean) / std
    return np.nan_to_num(z, nan=0.0), mean, std


def fit_logistic(z, y, l2=1e-2, steps=400, lr=0.5):
    """Plain gradient-descent logistic regression, numpy only, deterministic."""
    Xb = np.hstack([z, np.ones((len(z), 1))])
    w = np.zeros(Xb.shape[1])
    for _ in range(steps):
        p = 1.0 / (1.0 + np.exp(-Xb @ w))
        grad = Xb.T @ (p - y) / len(y) + l2 * np.r_[w[:-1], 0.0]
        grad[-1] = 0.0  # intercept fixed at 0 (symmetry, see standardise)
        w -= lr * grad
    return w


def predict_home(z, w):
    Xb = np.hstack([z, np.ones((len(z), 1))])
    return 1.0 / (1.0 + np.exp(-np.clip(Xb @ w, -30, 30)))


def save(w, mean, std, path=WEIGHTS, note=""):
    with open(path, "w") as f:
        json.dump({"w": list(map(float, w)), "mean": list(map(float, mean)),
                   "std": list(map(float, std)), "note": note}, f, indent=1)


def load(path=WEIGHTS):
    with open(path) as f:
        d = json.load(f)
    return np.array(d["w"]), np.array(d["mean"]), np.array(d["std"])


def p_home(players, team, blocks, path=WEIGHTS):
    w, mean, std = load(path)
    z, _, _ = standardise(shape_features(players, team, blocks), mean, std)
    return predict_home(z, w)
