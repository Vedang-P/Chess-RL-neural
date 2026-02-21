# Chess RL Endgames: Neural KQ vs K

<p align="center">
  <img src="assets/chess-rl-live.gif" alt="Live neural training dashboard (KQ vs K)" width="920" />
</p>

Neural reinforcement learning in a solved chess endgame: **King + Queen vs King (KQK)**.

This project trains a Q-learning style neural agent to convert winning KQK positions into checkmate, while exposing training dynamics in a live GUI (board + metrics + outcomes).

## Why This Project Exists

Most chess RL projects jump directly to full-board self-play and huge models.
This one does the opposite:

- Uses a **small solved domain** (KQK) so behavior is measurable.
- Uses **structured features** instead of raw board pixels.
- Uses **explicit reward shaping** tied to endgame geometry.
- Compares behavior against optional **Syzygy tablebase** signals.

This makes the project easier to explain on a resume and easier to debug scientifically.

## What You Can Do

- Train a neural agent in a live dashboard with board animation.
- Run headless experiments and log CSV artifacts.
- Evaluate against heuristic, random, or Syzygy-based defenders.
- Track win/draw/queen-loss trends and policy quality over time.

## Project Layout

```text
chess_RL/
  scripts/
    kqk_neural.py          # main entry point (live/train/eval)
  src/chess_rl/
    env.py                 # KQK environment + reward shaping
    neural_agent.py        # neural Q-agent (replay + target network)
    train.py               # training loop
    evaluate.py            # evaluation loop + metrics
    live_viewer.py         # Tkinter + Matplotlib live UI
    features.py            # structured state/action encoding
    policies.py            # defender/attacker policies
    syzygy.py              # tablebase oracle wrapper
    syzygy_utils.py        # syzygy path discovery
    endgames/kqk.py        # state abstraction + curriculum sampler
  tests/
  artifacts/               # model/log outputs (generated)
```

## Tech Stack

- [Python](https://www.python.org/)
- [python-chess](https://python-chess.readthedocs.io/)
- [PyTorch](https://pytorch.org/)
- [Matplotlib](https://matplotlib.org/)
- [Syzygy tablebases](https://syzygy-tables.info/) (optional)

## Installation

From repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,analysis,rl]"
```

## Quick Start

Run live training dashboard:

```bash
PYTHONPATH=src python3 scripts/kqk_neural.py \
  --mode live \
  --episodes 20000 \
  --defender heuristic \
  --board-size 780
```

Run headless train + eval:

```bash
PYTHONPATH=src python3 scripts/kqk_neural.py \
  --mode train-eval \
  --episodes 20000 \
  --eval-episodes 2000 \
  --defender heuristic \
  --eval-defender heuristic
```

Evaluate an existing model only:

```bash
PYTHONPATH=src python3 scripts/kqk_neural.py \
  --mode eval \
  --eval-episodes 2000 \
  --eval-defender heuristic
```

## CLI Modes

- `live`: train with GUI.
- `train`: train headless, save model + log.
- `eval`: evaluate existing saved model.
- `train-eval`: train then evaluate.
- `live-eval`: live training then evaluation.

See all options:

```bash
PYTHONPATH=src python3 scripts/kqk_neural.py --help
```

## How Training Works (Beginner-Friendly)

Each episode:

1. A legal KQK position is sampled (curriculum can bias easier starts first).
2. White (agent) chooses a move from legal actions.
3. Black (defender policy) replies.
4. Environment returns:
   - next abstract state
   - shaped reward
   - done flag (`checkmate`, `draw`, `queen_lost`, etc.)
5. Agent stores transition in replay buffer and periodically updates Q-network.

### State and Action Encoding

The model does not see raw board images. It sees engineered features:

- State: king distance, queen distance, edge/corner pressure, opposition flags, mobility bucket.
- Action: move type (king/queen), from/to coordinates, move vector, check flag, capture flag.

This keeps learning interpretable and lightweight.

### Reward Design (Current)

The environment encourages fast, safe mating patterns:

- `+10` checkmate + speed bonus
- `-2` draw
- `-3` queen loss
- `-0.02` per step
- positive shaping for reducing black mobility and forcing king toward edge/corner
- stall penalty for no progress streaks

## Understanding the Dashboard

- **Learning Signal**:
  - reward rolling mean
  - loss rolling mean
- **Outcome Rates**:
  - checkmate
  - draw
  - queen_lost
  - max_length
  - loss
- **Efficiency / Safety**:
  - episode length
  - check frequency
  - queen safety
- **Policy Quality / Exploration**:
  - epsilon
  - average selected Q
  - optional TB-optimal move rate when Syzygy is available

## Why Neural Results Can Look Bad Early

If you see low checkmate rates and high draw/queen-loss (like your screenshot), this is usually expected in early-to-mid training and comes from a few concrete factors:

1. Exploration is still high.
At ~2000 episodes with default schedule, epsilon is often still around `0.20+`, so many random tactical blunders are still injected.

2. Heuristic defender is non-trivial.
The black defender actively seeks safer king geometry, making random/imperfect white play collapse into draws or queen losses.

3. Most transitions are negative before mating is discovered.
Per-step penalties + draw penalties dominate until agent reliably builds mating nets, so mean reward stays negative for a while.

4. Low loss does not guarantee good policy.
A small TD loss can mean the model is fitting a mediocre fixed point (predicting similarly bad values), not that it found high-quality play.

## Practical Hyperparameter Tips

If neural convergence is too slow, start here:

```bash
PYTHONPATH=src python3 scripts/kqk_neural.py \
  --mode train-eval \
  --episodes 40000 \
  --eval-episodes 4000 \
  --defender heuristic \
  --eval-defender heuristic \
  --epsilon 0.15 \
  --epsilon-decay 0.9995 \
  --gamma 0.92 \
  --batch-size 32 \
  --warmup-steps 256 \
  --train-interval 4 \
  --updates-per-step 2
```

Interpretation goal:
- queen losses should trend down first
- then draw rate should drop
- then checkmate rate should rise

## Syzygy Setup (Optional, Recommended)

Syzygy is optional. If missing, code falls back gracefully.

Download minimal KQK files:

```bash
mkdir -p ~/syzygy
cd ~/syzygy
curl -LO https://tablebase.lichess.ovh/tables/standard/3-4-5-wdl/KQvK.rtbw
curl -LO https://tablebase.lichess.ovh/tables/standard/3-4-5-dtz/KQvK.rtbz
```

Run with Syzygy defender:

```bash
PYTHONPATH=src python3 scripts/kqk_neural.py \
  --mode train-eval \
  --defender syzygy \
  --eval-defender syzygy \
  --syzygy-path ~/syzygy
```

Note:
- `--require-syzygy` currently emits a warning when Syzygy is not loaded; it does not hard-fail.

## Outputs

By default:

- Model: `artifacts/kqk_neural_q.pt`
- Training log: `artifacts/kqk_neural_training.csv`

These are intentionally excluded from git via `.gitignore`.

## Running Tests

```bash
PYTHONPATH=src pytest -q
```

## Add Your GIF at Top

This README expects your demo GIF at:

- `assets/chess-rl-live.gif`

Suggested workflow:

1. Record screen during live training.
2. Convert to GIF (or use `ffmpeg` + `gifski`).
3. Place file at `assets/chess-rl-live.gif`.
4. Commit and push.

## Resume-Ready Summary

You can describe this project as:

> Built a neural reinforcement-learning system for the solved chess endgame KQ vs K using structured state abstractions, shaped rewards, replay-based Q-learning, optional Syzygy tablebase integration, and a live training dashboard for real-time diagnostics and failure analysis.
