# Technical document — More than a Hack 2027, Phase 1: Ball Possession

Team: `<team name>` · Repository: https://github.com/L0adn9/the-hackathon

Submitted file: `evaluation/results/possession_v5.csv`, produced by `starter/possession_v5.py`.

## 1. Summary

We label every frame of the tactical video `H`, `A` or `D` in two steps:

1. **A frame classifier** gives, for every frame, the probability of `H`, `A` and `D`. It is a
   gradient-boosted tree model (scikit-learn `HistGradientBoostingClassifier`) reading 70
   features computed from the tracking: distances from the ball to both teams, ball speed and
   position, how long the ball has been unseen, collective player speed, team shape, ball-centred
   dead-ball cues (where the ball vanished and reappears, whether it was heading out, whether it
   sits still on a restart spot, whether a kick follows), and averages of these over the previous
   and next 0.5 s, 1 s and 3 s. The time windows are what let it judge a frame with no ball from
   what happened just before and just after.
2. **Viterbi decoding** turns those probabilities into whole possessions. Each state costs
   `-log P(state)` per frame and every change of state has a price, so a team change needs
   sustained evidence and the output does not flicker. Smart Tagging set pieces are added as extra
   costs before decoding (restarts and the restarting team).

The classifier is trained **only on public Metrica sample games** (other matches, with their
tracking degraded to look like the competition ATD). All settings were chosen on a proxy
leaderboard built from those games (section 4), never on the competition match or its public five
minutes.

**One finding shaped the final version.** In the sample games the ball position is blanked during
most stoppages, so on a naive proxy "ball missing" almost means "dead ball". The competition ATD is
not like that. We rebuilt the proxy to remove this shortcut (section 4.2); on the corrected board,
with each game scored by a model that never saw it, the final version scores spellF1 0.681 against
0.566 for the same model without the dead-ball work and 0.473 for our rule-based version.

## 2. Assets used and dropped

| asset | used | how |
|---|---|---|
| Tactical ATD — ball and player positions | yes | the input to every feature |
| Tactical ATD — player speeds column | no | we compute speeds from positions instead, so the same code runs on the sample games |
| Smart Tagging — set pieces and kick-offs | yes | restart anchoring and restarting team (v2 layer, unchanged) |
| Smart Tagging — set-piece timing | yes, for analysis only | to measure how often the ATD loses the ball before restarts (section 4.2); not a model input beyond the restart layer |
| Smart Tagging — phase codes (Build Up, Defending…) | tried, dropped | phases are not possessions; lowered the proxy score in v2 |
| Tactical homography | no | tracking is already in pitch coordinates |
| Tactical / broadcast video | no | time; noted as future work |
| Broadcast ATD | no | not frame-aligned to the tactical timeline |

## 3. Method

### 3.1 Pre-processing
- Raw ATD parsed with numpy (`possession_v2.load_tracking_with_speed`); normalised coordinates
  scaled to metres (105.3 × 68.0).
- Ball gaps of up to 1 s are linearly interpolated for the distance features. Longer gaps stay
  unknown, and the model is told how long the ball has been missing and how long until it is
  seen again.
- Each team's attacking direction is found per half from where its players stand on average,
  so ball position and velocity are expressed relative to the goal a team attacks.

### 3.2 Features (`starter/frame_features.py`)
All features use team membership only — no player identities, which the ATD does not have.

| group | features |
|---|---|
| ball vs players | nearest and second-nearest home / away player; home and away players within 5 m and 10 m; distance advantage (away minus home); "control" (+1 / −1 when the nearest player is within 2 m) |
| ball | speed; position along the home attacking direction; velocity towards goal; distance outside the pitch; distance to side and end lines |
| ball visibility | ball seen / known this frame; frames since last seen; frames until next seen |
| players | median speed of all tracked players (smoothed over 6 s, and over the past and next 3 s); home minus away median speed; number of tracked objects |
| team shape | the five v3 team-shape features: forward movement, length, width, area, territory (home minus away) |
| dead-ball cues | for the last-seen and next-seen ball position: distance to the nearest line, to the nearest corner, to the centre spot and to the goal area; whether the ball was heading out when last seen; ball speed when last and next seen; frames since the ball last moved; frames until the next kick (> 8 m/s); share of still-ball frames over ±2 s; players 8–11 m from the ball (a wall, or players kept away at a restart) and players within 3 m, with short averages |
| time context | advantage and control averaged over the past and the next 12 / 25 / 75 frames; ball speed, ball-outside and ball-seen share over ±12 / 25 / 75 frames |

The dead-ball cues follow the laws of the game rather than any particular player: a stoppage
usually starts with the ball leaving over a line or the play stopping around it, and ends with a
still ball on a restart spot (touchline, corner, goal area, centre spot, free-kick spot with a
wall) followed by a kick.

### 3.3 Classifier (`starter/frame_model.py`)
- `HistGradientBoostingClassifier`, 200 iterations, learning rate 0.1, 31 leaves,
  `min_samples_leaf=200`, L2 1.0, fixed `random_state`, no early stopping. Three classes:
  `H`, `A`, `D`.
- Trained on every third graded frame of the three sample games, under two degradations (plain
  and realistic, section 4) with two seeds each: 1,133,032 rows.
- **Team symmetry.** Every training frame is used twice: once as it is, and once with the teams
  swapped (features recomputed with home and away exchanged, label `H` ↔ `A`). At prediction time
  the model is run from both points of view and the two answers are averaged. This stops the model
  from learning "the home team usually has the ball" from the training games, which would not
  transfer to Barcelona v Athletic.

### 3.4 Smart Tagging restarts
Unchanged from v2/v3: before each tagged set piece H and A pay an extra cost (dead ball before the
restart); after it the non-restarting team pays a cost. Each half is dead until its kick-off.
Throw-ins are tagged with no team, so the restarting team is taken from geometry: the team of the
player standing on the ball.

### 3.5 Decoding
Viterbi per half over {H, A, D}. Per-frame cost `-log P(state)` (probabilities clipped at 1e-4)
plus the tagging costs. Transition prices: H↔A 10, alive→D 15, D→alive 10. Dead-ball runs of up
to 2 s with the same team on both sides are bridged (an unseen ball is not a stoppage). Frames
outside the two graded halves are `X`; no `X` is written inside a half.

### 3.6 Possession rules implemented
- Possession changes only on sustained evidence for the other team: the switch price means a
  touch, a deflection or a lost aerial duel is not enough on its own, and loose-ball frames keep
  the last team.
- A restart starts a new possession for the restarting team.
- Stoppages: the ball leaving over a line, play stopping around a still ball, players slowing
  down collectively, a still ball on a restart spot followed by a kick, plus the run-up to each
  tagged set piece and kick-off.

## 4. Validation without a leaderboard: the proxy board

The public five minutes contain 17 possessions and are not a tuning set (RULES.md). We built our own
board from **public Metrica sample games 1, 2 and 3** (https://github.com/metrica-sports/sample-data).

### 4.1 Plain board
1. **Truth from events** (other games, human-annotated by Metrica): possession = last team with a
   controlled action (pass, carry, recovery, shot, set piece); dead from ball out / foul / goal until
   the next restart (`tools/proxy_board.py`).
2. **Degradation** of their clean tracking to resemble the competition ATD: ball missing on ~41% of
   graded frames in bursts, positional jitter (ball 0.3 m, players 0.2 m), player dropouts (~7%).
3. Scoring with the official `possession_metrics.py`.
4. **Leave one game out.** The classifier is trained on two games and scored on the third, for each
   game in turn (`python tools/train_frame_model.py --samples ... --version v5 --check`).

### 4.2 The shortcut in the plain board, and the realistic board
In the clean sample games the ball is already missing on **81–98% of dead-ball frames** but only
6–12% of alive frames: the ball is blanked during stoppages. After the plain degradation, "ball
missing" is therefore close to an answer key for `D`, and every version that reads ball absence
learns it.

We measured the same thing on the competition ATD using only Smart Tagging timing (no labels): the
ball is missing on **71%** of frames in the 5 s before tagged set pieces and on **19%** of frames
inside tagged attacking phases (40.7% of all graded frames). The link exists, but it is much weaker.

The **realistic board** (`proxy_board.degrade_realistic`) removes the shortcut:
- during each stoppage the ball is given a position — a straight line from where it was last seen
  to where it is next seen, then still on the restart spot for the last 2 s;
- the ball is then hidden on ~70% of dead frames and ~25% of alive frames, in long bursts
  (~16 s and ~6 s), which also brings gap lengths closer to the competition ATD;
- jitter and player dropouts as on the plain board.

All final decisions were made on the realistic board, with the plain board as a second check.

### 4.3 Results
Mean spellF1 over two degradation seeds per game, leave one game out:

| pipeline | realistic g1 | g2 | g3 | **realistic mean** | plain mean |
|---|---|---|---|---|---|
| v3 (rule-based + team shape) | 0.442 | 0.502 | 0.474 | 0.473 | 0.595 |
| v4 (classifier, no dead-ball cues, plain training) | 0.531 | 0.609 | 0.557 | 0.566 | 0.699 |
| **v5 (final)** | **0.692** | **0.689** | **0.662** | **0.681** | **0.717** |

On the realistic board v5 is better than v4 on every game, and its alive/dead F1 goes from about
0.85 to 0.95. On the plain board it is better on games 1 and 3 and slightly worse on game 2.

**Decoder settings.** A 72-point grid over the four transition settings, scored on both boards,
spans 0.662 to 0.679 on the realistic board; both boards prefer a cheaper team switch than v4 used.

The proxy is easier than the real data (the official baseline scores 0.36 on the plain board vs.
0.24 on the real match), so absolute numbers do not transfer; it is used for **relative** decisions
only.

The public five minutes were used only to check that the file is correct (frames aligned, labels
meaning what we think). No setting was chosen on them.

## 5. Training data

| data | source | labelled by | amount | used for |
|---|---|---|---|---|
| Metrica sample games 1–3: tracking + event data | public GitHub repo above | Metrica (event data) | 3 matches | frame classifier, team-shape weights, proxy boards |

**No labels of any kind from the competition match** were used for training, tuning or validation.

## 6. Limitations and future work
- The proxy truth is derived from events, which approximates but is not identical to the corrected-
  tracking definition used by the organisers.
- The ball position during stoppages on the realistic board is reconstructed, not observed; how
  the ball really moves while dead (carried, thrown back, placed) is simplified.
- Three training games is a small sample of teams and styles; more public tracking with events
  would help most.
- On the proxy, v5 still reports fewer possessions than the truth (short possessions get merged
  into their neighbours).
- Smart Tagging restarts could not be validated on the proxy (sample games have no tagging).
- Not used: video (ball detection to fill gaps), referee whistle from audio, broadcast feed.

## 7. Reproduction

```bash
git clone https://github.com/L0adn9/the-hackathon && cd the-hackathon
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-v4.txt
export MTAH_DATA_DIR=/path/to/data      # tactical/ and broadcast/ folders from the organisers
python starter/possession_v5.py --cache tracking_v2.npz
python evaluation/validate_submission.py evaluation/results/possession_v5.csv
```

The pipeline is deterministic: running it twice, with or without the cache, and after retraining
the model from scratch, gives a byte-identical file (and a byte-identical model file). The trained
model is committed as `starter/frame_model_v5.pkl` and is tied to scikit-learn 1.9.1 (pinned);
`frame_model.load()` refuses to run with another version. To retrain it:

```bash
git clone https://github.com/metrica-sports/sample-data /tmp/sample-data
python tools/train_frame_model.py --samples /tmp/sample-data/data --version v5           # writes starter/frame_model_v5.pkl
python tools/train_frame_model.py --samples /tmp/sample-data/data --version v5 --check   # leave-one-game-out scores
```

## 8. Earlier versions

Kept in the repository; each produces a valid file.

- `possession_v1.py` — hand-set per-frame costs from ball geometry, Viterbi decoding.
- `possession_v2.py` — adds collective player speed and the Smart Tagging layer.
- `possession_v3.py` — v1 + a small logistic team-shape model + Smart Tagging set pieces; settings
  from a random search on the plain board.
- `possession_v4.py` — the frame classifier without dead-ball cues, trained on the plain board only
  (`starter/frame_model.pkl`). Strong on the plain board (0.699), but much of its dead-ball skill
  came from the shortcut in 4.2, which is why v5 exists.
