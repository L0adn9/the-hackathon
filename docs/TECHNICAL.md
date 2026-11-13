# Technical document — More than a Hack 2027, Phase 1: Ball Possession

Team: `<team name>` · Repository: https://github.com/L0adn9/the-hackathon

## 1. Summary

We label every frame of the tactical video `H`, `A` or `D` with a **three-state cost model decoded by
Viterbi**, so that possessions are produced as whole intervals rather than as flickering per-frame
guesses. The per-frame evidence comes from three sources:

1. **Ball geometry** — which team is nearest to the ball, how close, how fast the ball moves, and
   whether it is outside the pitch.
2. **Team shape** — a small, team-symmetric logistic model that reads possession from the shape of
   the 20 outfield players (who pushes forward, who stretches, who occupies more area). It needs no
   ball, so it carries the ~41% of graded frames where the ball is not tracked. Its weights are
   learned **only on public Metrica sample games**, never on the competition match.
3. **Smart Tagging restarts** — every tagged set piece is a restart: the moments before it are dead
   ball and the moments after it belong to the restarting team; each half is dead until its tagged
   kick-off. Throw-ins are tagged without a team, so the restarting team is taken from geometry
   (team of the player standing on the ball).

All settings were chosen on a **proxy leaderboard** we built from public games of other matches
(section 4), never on the competition match or its public five minutes.

## 2. Assets used and dropped

| asset | used | how |
|---|---|---|
| Tactical ATD (tracking + ball) | yes | ball and player positions; the core input |
| Tactical ATD player speeds | tried, dropped | collective-speed stoppage cue; neutral on the proxy board |
| Smart Tagging — set pieces / kick-offs | yes | restart anchoring and restarting team |
| Smart Tagging — phase codes (Build Up, Defending…) | tried, dropped | phases are not possessions; lowered the score |
| Tactical homography | no | not needed: tracking is already in pitch coordinates |
| Tactical / broadcast video | no | time; noted as future work |
| Broadcast ATD | no | not frame-aligned to the tactical timeline |

## 3. Method

### 3.1 Pre-processing
- Raw ATD parsed with numpy; normalised coordinates scaled to metres (105.3 × 68.0).
- Ball gaps of up to 4 s are linearly interpolated; longer gaps stay unknown.

### 3.2 Per-frame costs (lower = more likely)
- **H / A**: `-w · log p(team)`, where `p` is a logistic function of the distance advantage of the
  nearest home vs. away player, and the weight `w` fades when nobody is close to the ball or the ball
  is in flight (loose balls do not change possession).
- **D**: cheap when the ball is outside the pitch lines or is still with nobody near; otherwise a
  fixed cost. On frames with no ball, H/A/D are nearly free, so the decoder keeps the current state
  (possession persists through occlusions).
- **Team shape**: `-0.25 · log p_shape(team)` added on every frame.
- **Set pieces**: H and A pay a cost before the restart (middle of the ~5 s tag window); the
  non-restarting team pays a cost after it.

### 3.3 Decoding
Viterbi per half over {H, A, D} with transition prices (H↔A: 30, alive→D: 8, D→alive: 4). A team
change therefore needs sustained evidence. Dead-ball runs shorter than 4 s with the same team on
both sides are bridged (an unseen ball is not a stoppage). Frames outside the two graded halves are
`X`. No `X` is written inside a half.

### 3.4 Possession rules implemented
- Possession changes only on sustained control by the other team; single touches, deflections and
  loose balls keep the last team in possession (low evidence weight when nobody is close).
- A restart starts a new possession for the restarting team.
- Stoppages: ball out of the pitch, still ball with nobody near, and the run-up to each tagged set
  piece and kick-off.

## 4. Validation without a leaderboard: the proxy board

The public five minutes contain 17 possessions and are not a tuning set (RULES.md). We built our own
board from **public Metrica sample games 1, 2 and 3** (https://github.com/metrica-sports/sample-data):

1. **Truth from events** (other games, human-annotated by Metrica): possession = last team with a
   controlled action (pass, carry, recovery, shot, set piece); dead from ball out / foul / goal until
   the next restart (`tools/proxy_board.py`).
2. **Degradation** of their clean tracking to resemble the competition ATD: ball missing on ~41% of
   graded frames in bursts, positional jitter (ball 0.3 m, players 0.2 m), player dropouts (~7%).
3. Scoring with the official `possession_metrics.py`. Team-shape weights are trained
   leave-one-game-out, so each game is scored by a model that never saw it.

| pipeline | game 1 | game 2 | game 3 | mean spellF1 |
|---|---|---|---|---|
| v1 (ball geometry + Viterbi) | 0.524 | 0.545 | 0.472 | 0.514 |
| v1 + team shape (weight 0.15) | 0.557 | 0.584 | 0.542 | 0.561 |
| **v3 final settings** (fresh degradation seeds) | **0.581** | **0.634** | **0.544** | **0.586** |

Settings were chosen by a 60-point random search (`tools/proxy_exp.py`) and then **re-checked on
new degradation seeds** the search never saw, to avoid picking a lucky configuration. The final
settings beat the v1 defaults on every game. Changes from v1: control radius 2 m, minimum evidence
weight 0.05, ball-gap interpolation up to 4 s, H↔A price 30, dead-ball base cost 1.0, team-shape
weight 0.25.

The proxy is easier than the real data (the official baseline scores 0.36 there vs. 0.24 on the real
match), so absolute numbers do not transfer; it is used for **relative** decisions only.

## 5. Training data

| data | source | labelled by | amount | used for |
|---|---|---|---|---|
| Metrica sample games 1–3: tracking + event data | public GitHub repo above | Metrica (event data) | 3 matches | team-shape weights; proxy board |

**No labels of any kind from the competition match** were used for training, tuning or validation.
The public five minutes were used only to check that the pipeline and file format are correct.

## 6. Limitations and future work
- The proxy truth is derived from events, which approximates but is not identical to the corrected-
  tracking definition used by the organisers.
- Smart Tagging restarts could not be validated on the proxy (sample games have no tagging).
- Restart instant is taken as the middle of the tag window; a ball-kick detector would be sharper.
- Not used: video (ball detection to fill gaps), referee whistle from audio, broadcast feed.

## 7. Reproduction

```bash
git clone https://github.com/L0adn9/the-hackathon && cd the-hackathon
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export MTAH_DATA_DIR=/path/to/data      # tactical/ and broadcast/ folders from the organisers
python starter/possession_v3.py --cache tracking_v2.npz
python evaluation/validate_submission.py evaluation/results/possession_v3.csv
```

The pipeline is deterministic (no randomness at inference); the team-shape weights are committed in
`starter/team_shape_weights.json` and can be regenerated with `tools/train_team_shape.py`
(requires `kloppy` for sample game 3).
