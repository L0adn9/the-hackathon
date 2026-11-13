"""Per-frame H / A / D probabilities from a gradient-boosted classifier.

The model is trained ONLY on the public Metrica sample games (tools/train_frame_model.py), with
their tracking degraded to look like the competition ATD. Nothing from the competition match is
used to train or tune it.

Predictions are made team-symmetric: the model sees the match from the home side and from the
away side (teams swapped), and the two answers are averaged. A model that has learned "the home
team usually has the ball" in the training games therefore cannot carry that over.
"""

import os
import pickle

import numpy as np

import frame_features as ff

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(HERE, "frame_model.pkl")

PARAMS = dict(max_iter=200, learning_rate=0.1, max_leaf_nodes=31, l2_regularization=1.0,
              min_samples_leaf=200, random_state=0, early_stopping=False)
SWAP = np.array([1, 0, 2, 3])  # H <-> A, D and X unchanged


def training_rows(ball, players, team, blocks, truth, pitch, step=3):
    """Feature rows and labels from both teams' points of view, every `step`-th graded frame."""
    ok = np.flatnonzero(truth != 3)[::step]
    x_home = ff.features(ball, players, team, blocks, pitch)[ok]
    x_away = ff.features(ball, players, 1 - team, blocks, pitch)[ok]
    return np.vstack([x_home, x_away]), np.concatenate([truth[ok], SWAP[truth][ok]])


def fit(x, y):
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(**PARAMS).fit(x, y)


def save(model, path=MODEL):
    import sklearn
    with open(path, "wb") as f:
        pickle.dump({"model": model, "sklearn": sklearn.__version__,
                     "features": ff.feature_names()}, f)


def load(path=MODEL):
    import sklearn
    with open(path, "rb") as f:
        stored = pickle.load(f)
    if stored["sklearn"] != sklearn.__version__:
        raise RuntimeError("frame_model.pkl was saved with scikit-learn {} but {} is installed; "
                           "install the pinned version (requirements-v4.txt)".format(
                               stored["sklearn"], sklearn.__version__))
    if stored["features"] != ff.feature_names():
        raise RuntimeError("feature list changed since the model was trained; "
                           "rerun tools/train_frame_model.py")
    return stored["model"]


def probabilities(model, ball, players, team, blocks, pitch):
    """(n, 3) symmetric P(H), P(A), P(D) per frame."""
    p = model.predict_proba(ff.features(ball, players, team, blocks, pitch))
    q = model.predict_proba(ff.features(ball, players, 1 - team, blocks, pitch))
    return np.column_stack([(p[:, 0] + q[:, 1]) / 2.0,
                            (p[:, 1] + q[:, 0]) / 2.0,
                            (p[:, 2] + q[:, 2]) / 2.0])
