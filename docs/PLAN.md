# More than a Hack 2027 — Team plan and progress

Repo: https://github.com/L0adn9/the-hackathon
Official challenge: https://github.com/metrica-sports/more-than-a-hack-2027
**Our target: final submission done by 15 Nov 2026** (official deadline 17 Nov; 16–17 are buffer only).

---

## 1. The task in one paragraph

One full match (FC Barcelona = `H`, Athletic Club = `A`). For each of the 199,319 frames of the
tactical video, output `H`, `A`, `D` (dead ball) or `X` (no answer). Ranked on **spellF1**
(whole possessions matched by temporal IoU), ties broken by macroF1. Published floor to beat:
**spellF1 0.2433**. Graded frames: 11185–90651 and 110808–190197.

Deliverables (one Google Form): `<team>.csv`, a **public** repo that regenerates the exact file,
and a short technical document (`<team>.pdf` or `.md`).

### Rules we must not break
- No hand labelling of this match, in any amount, for any purpose.
- The public 5 minutes are for checking the pipeline only — never for training or tuning.
- No human-produced data about this match (event data, corrected tracking...).
- Do not upload the match data or extracts to any cloud LLM. Data stays local.
- Never commit `data/`, caches (`ball.npz`) or any frame-level export.
- The repo must reproduce the submitted CSV exactly (deterministic code only).
- Team of 3–5 members.

---

## 2. What is done (12 Nov)

- Read the challenge, data spec, metric and rules.
- Data downloaded and organised locally (`~/dev/mtah-data/{tactical,broadcast}`, outside the repo).
- Python environment set up (`.venv`, numpy + pandas).
- Baselines regenerated and scored locally.
- **v1 submission** — `starter/possession_v1.py`:
  - parses the raw tracking (numpy only),
  - interpolates short ball gaps (≤ 2 s),
  - per frame: distance to nearest home / away player + ball speed,
  - per-frame costs for H / A / D (control weighting, ball out of pitch, still ball),
  - Viterbi decoding per half with switching penalties (kills flicker),
  - bridges short dead-ball runs between the same team.
- v1 output passes `validate_submission.py`. Full match: 358 possessions, mean 11.0 s.
- Sanity check on the public window only (not a tuning signal): spellF1 0.50 vs 0.42 for the best
  baseline; 19 possessions vs 17 true.
- Known weakness: dead-ball detection (most stoppage frames on the public window are labelled as
  possession).

### How to run
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export MTAH_DATA_DIR=~/dev/mtah-data
python starter/possession_v1.py --cache ball.npz
python evaluation/validate_submission.py evaluation/results/possession_v1.csv
```

---

## 3. The approach we are building: "read the match as events"

1. **Touch detector (ball physics)** — detect touches from kinks in the ball trajectory
   (sudden change of direction or speed) and attribute them to the nearest player. Possession
   changes only on *controlled* touches by the other team (two consecutive touches, or touch +
   carry); deflections and single touches do not count — the challenge's own definition.
2. **Team-shape possession sensor (no ball needed)** — the team in possession stretches and moves
   forward; the defending team compacts and drops. Team width, length, centroid velocity and line
   spacing cover the 41% of frames where the ball is missing.
3. **Laws-of-the-game stoppages and restarts** — dead ball from: ball out of bounds, collective
   slowdown of players, set-piece formations (corner, throw-in, goal kick, kick-off), and Smart
   Tagging set pieces. Restart team from the laws: last touch before going out → the other team
   restarts.
4. **Self-training from Smart Tagging** — algorithmic pseudo-labels (allowed by the rules): attacking
   phases → that team has the ball; defending phases → the other team. Train a light ML model on the
   features above.
5. **Possession-level decoding** — duration-aware decoder (possessions and stoppages have typical
   lengths) instead of a flat switch cost. Use `X` only inside long ball occlusions within a
   possession; always commit at transitions.

### Our own leaderboard (proxy validation)
There is no live leaderboard and the public window cannot rank anything. We build one from
**public matches of other games** (Metrica sample data; game 3 is in the same EPTS format):
derive true possession from their events, degrade their tracking to match our ATD (≈41% ball
missing in bursts, identity switches, jitter), and score with the official scorer. All parameter
choices are made there. Allowed by the rules (external data about other games), and documented.

### Stretch only if time allows
- Referee whistle detection from the tactical video audio (check the file has audio first).
- Local ball detector on the video for frames where the ball is missing (heavy; needs disk space).

---

## 4. Schedule (finish by 15 Nov)

| Day | Code (Youssef + Claude) | Teammate |
|---|---|---|
| **13 Nov** | Layer 3 (stoppages + restarts) and layer 1 (touch detector) → v2 | Download data locally, run v1, confirm identical CSV |
| **14 Nov** | Layer 2 (team shape), layer 4 (Smart Tagging self-training), layer 5 (decoder); proxy leaderboard ready; choose settings on it → v3 | Draft the technical document from this file |
| **15 Nov** | Freeze code. Final CSV, validate, regenerate on a second machine, finish doc, make repo **public**, submit the form | Review doc, check repo has no data, submit |
| 16–17 Nov | Buffer only (fixes if something fails) | |

Every layer ships on its own and produces a valid file, so we always have a submittable version.

### Submission checklist
- [ ] `<team>.csv` passes `validate_submission.py` (199,319 rows)
- [ ] Repo regenerates the identical file from the data bundle
- [ ] No data, caches or frame-level exports in the repo
- [ ] Repo switched to Public
- [ ] Technical document: approach, assets used/dropped, training data (none from this match),
      possession rules, limitations, how to reproduce
- [ ] Files named after the team; form submitted with a Google account
