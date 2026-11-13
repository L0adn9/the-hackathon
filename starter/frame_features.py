"""Per-frame features for the possession classifier (see frame_model.py).

Everything is computed from team-level tracking only (ball + player slots + team of each slot),
so the same code runs on the competition ATD and on the public sample games used for training.

Features are written from the home team's point of view. Calling features() again with the
teams swapped (team -> 1 - team) gives the away team's point of view, which is how the model is
trained symmetrically and how predictions are made symmetric at run time.
"""

import warnings

import numpy as np

import possession_v1 as v1
import team_shape as ts

WINDOWS = (12, 25, 75)  # frames: ~0.5 s, 1 s, 3 s
SHORT_GAP = 25          # ball gaps up to 1 s are interpolated for the distance features
CAP = 250               # frames: cap for "time since / until the ball is seen"


def _nanmedian(x, axis):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(x, axis=axis)


def _window_mean(x, before, after):
    """NaN-aware mean of x over [t - before, t + after]."""
    valid = ~np.isnan(x)
    v = np.where(valid, x, 0.0)
    cs = np.concatenate(([0.0], np.cumsum(v)))
    cc = np.concatenate(([0], np.cumsum(valid)))
    i = np.arange(len(x))
    lo = np.maximum(0, i - before)
    hi = np.minimum(len(x), i + after + 1)
    n = cc[hi] - cc[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        out = (cs[hi] - cs[lo]) / n
    out[n == 0] = np.nan
    return out


def _frames_since(seen):
    """Frames since the last True (0 on a True frame), capped."""
    n = len(seen)
    idx = np.where(seen, np.arange(n), -1)
    last = np.maximum.accumulate(idx)
    out = np.where(last >= 0, np.arange(n) - last, CAP)
    return np.minimum(out, CAP).astype(float)


def _sorted_team_distances(ball, players, team, k):
    """Distances from the ball to the k nearest players of one team, NaN-padded."""
    delta = players[:, team == k, :] - ball[:, None, :]
    dist = np.sqrt((delta ** 2).sum(axis=-1))
    dist = np.where(np.isnan(dist), np.inf, dist)
    dist.sort(axis=1)
    first = dist[:, 0] if dist.shape[1] > 0 else np.full(len(ball), np.inf)
    second = dist[:, 1] if dist.shape[1] > 1 else np.full(len(ball), np.inf)
    within5 = (dist < 5.0).sum(axis=1).astype(float)
    within10 = (dist < 10.0).sum(axis=1).astype(float)
    return first, second, within5, within10


def _player_speeds(players, fps, half=12):
    n = len(players)
    speed = np.full(players.shape[:2], np.nan)
    c = np.arange(half, n - half)
    delta = players[c + half] - players[c - half]
    speed[c] = np.sqrt((delta ** 2).sum(-1)) * fps / (2.0 * half)
    return np.where(speed > 12.0, np.nan, speed)  # identity switches make impossible jumps


def features(ball, players, team, blocks, pitch, fps=25):
    """(n, k) float matrix of features, NaN where undefined. Names in feature_names()."""
    length, width = pitch
    n = len(ball)
    ball = ball.astype(np.float64)
    players = players.astype(np.float64)
    seen = ~np.isnan(ball[:, 0])
    bi = v1.interpolate_short_gaps(ball, SHORT_GAP)
    known = ~np.isnan(bi[:, 0])

    h1, h2, h5, h10 = _sorted_team_distances(bi, players, team, 0)
    a1, a2, a5, a10 = _sorted_team_distances(bi, players, team, 1)
    for arr in (h1, h2, a1, a2):
        arr[~np.isfinite(arr)] = np.nan
        np.minimum(arr, 60.0, out=arr)
    for arr in (h5, h10, a5, a10):
        arr[~known] = np.nan

    adv = np.clip(a1 - h1, -15.0, 15.0)            # > 0: home is closer
    dmin = np.fmin(h1, a1)
    with np.errstate(invalid="ignore"):
        control = np.where(dmin < 2.0, np.sign(adv), 0.0)
    control[~known] = np.nan

    speed = v1.ball_speed(bi, fps, half_window=6).astype(np.float64)
    speed = np.minimum(speed, 40.0)

    # ball position in the home team's attacking direction, per half
    bx = np.full(n, np.nan)
    vx = np.full(n, np.nan)
    shape = ts.shape_features(players, team, blocks, fps)
    for _, first, last in blocks:
        home_dir, _ = ts.attack_directions(players, team, first, last)
        sl = slice(first, last + 1)
        x = bi[sl, 0]
        bx[sl] = x if home_dir > 0 else length - x
        v = np.full(last - first + 1, np.nan)
        v[6:-6] = (x[12:] - x[:-12]) * fps / 12.0
        vx[sl] = v * home_dir
    out_x = np.maximum(np.maximum(-bi[:, 0], bi[:, 0] - length), 0.0)
    out_y = np.maximum(np.maximum(-bi[:, 1], bi[:, 1] - width), 0.0)
    outside = np.where(known, out_x + out_y, np.nan)
    to_side = np.where(known, np.minimum(bi[:, 1], width - bi[:, 1]), np.nan)
    to_end = np.where(known, np.minimum(bi[:, 0], length - bi[:, 0]), np.nan)

    spd = _player_speeds(players, fps)
    home_speed = _nanmedian(spd[:, team == 0], 1)
    away_speed = _nanmedian(spd[:, team == 1], 1)
    crowd = _nanmedian(spd, 1)
    tracked = (~np.isnan(players[:, :, 0])).sum(axis=1).astype(float)

    cols = [h1, h2, a1, a2, h5, a5, h10, a10, adv, control, speed, bx, vx,
            outside, to_side, to_end, known.astype(float), seen.astype(float),
            _frames_since(seen), _frames_since(seen[::-1])[::-1],
            home_speed - away_speed, crowd, _window_mean(crowd, 75, 75), tracked]
    for w in WINDOWS:
        cols += [_window_mean(adv, w, 0), _window_mean(adv, 0, w),
                 _window_mean(control, w, 0), _window_mean(control, 0, w),
                 _window_mean(speed, w, w), _window_mean(outside, w, w),
                 _window_mean(seen.astype(float), w, w)]
    cols += [shape[:, j] for j in range(shape.shape[1])]
    return np.column_stack(cols).astype(np.float32)


def feature_names():
    names = ["home_d1", "home_d2", "away_d1", "away_d2", "home_n5", "away_n5", "home_n10",
             "away_n10", "adv", "control", "ball_speed", "ball_x_att", "ball_vx_att",
             "outside", "to_side", "to_end", "ball_known", "ball_seen", "since_seen",
             "until_seen", "speed_diff", "crowd_speed", "crowd_speed_6s", "tracked"]
    for w in WINDOWS:
        names += ["adv_past{}".format(w), "adv_next{}".format(w), "control_past{}".format(w),
                  "control_next{}".format(w), "ball_speed_{}".format(w),
                  "outside_{}".format(w), "seen_share_{}".format(w)]
    names += ["shape_vx", "shape_len", "shape_wid", "shape_area", "shape_terr"]
    return names
