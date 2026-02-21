# Chess RL Endgames (Neural-Only KQ vs K)

This repo is neural-only. Tabular code has been removed.

Main script:
- `scripts/kqk_neural.py`

Modes:
- `--mode live` (GUI training)
- `--mode train` (headless training)
- `--mode eval` (evaluate saved model)
- `--mode train-eval` (train then evaluate)
- `--mode live-eval` (live train then evaluate)

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,analysis,rl]"
```

## Run Commands

Live GUI training:

```bash
PYTHONPATH=src python3 scripts/kqk_neural.py --mode live --episodes 20000 --defender heuristic --board-size 760
```

Headless train + eval:

```bash
PYTHONPATH=src python3 scripts/kqk_neural.py --mode train-eval --episodes 20000 --eval-episodes 2000 --defender heuristic --eval-defender heuristic
```

Evaluate existing model only:

```bash
PYTHONPATH=src python3 scripts/kqk_neural.py --mode eval --eval-episodes 2000 --eval-defender heuristic
```

## Syzygy (Optional)

If Syzygy files exist (`*.rtbw`, `*.rtbz`), they are used. If not found, script prints a warning and continues.

Download minimal KQK Syzygy files:

```bash
mkdir -p ~/syzygy
cd ~/syzygy
curl -LO https://tablebase.lichess.ovh/tables/standard/3-4-5-wdl/KQvK.rtbw
curl -LO https://tablebase.lichess.ovh/tables/standard/3-4-5-dtz/KQvK.rtbz
```

Run with Syzygy defender:

```bash
PYTHONPATH=src python3 scripts/kqk_neural.py --mode train-eval --defender syzygy --eval-defender syzygy --syzygy-path ~/syzygy
```

