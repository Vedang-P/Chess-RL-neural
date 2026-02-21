#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from tqdm import tqdm

from chess_rl.evaluate import evaluate_kqk
from chess_rl.live_viewer import run_live_viewer
from chess_rl.neural_agent import NeuralQAgent
from chess_rl.policies import HeuristicDefenderPolicy, RandomDefenderPolicy, SyzygyDefenderPolicy
from chess_rl.syzygy import SyzygyOracle
from chess_rl.syzygy_utils import discover_syzygy_paths
from chess_rl.train import train_kqk_neural, write_training_log


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified KQ vs K neural runner (live/train/eval).")
    parser.add_argument("--mode", choices=["live", "train", "eval", "train-eval", "live-eval"], default="live")
    parser.add_argument("--episodes", type=int, default=20_000)
    parser.add_argument("--eval-episodes", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=3e-4)
    parser.add_argument("--alpha-decay", type=float, default=0.9998)
    parser.add_argument("--alpha-min", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.90)
    parser.add_argument("--epsilon", type=float, default=0.30)
    parser.add_argument("--epsilon-decay", type=float, default=0.99985)
    parser.add_argument("--epsilon-min", type=float, default=0.02)
    parser.add_argument("--defender", choices=["random", "heuristic", "syzygy"], default="heuristic")
    parser.add_argument("--eval-defender", choices=["random", "heuristic", "syzygy"], default="heuristic")
    parser.add_argument("--syzygy-path", action="append", default=[])
    parser.add_argument("--disable-auto-syzygy", action="store_true")
    parser.add_argument("--require-syzygy", action="store_true")
    parser.add_argument("--enable-threefold-draw", action="store_true")
    parser.add_argument("--stability-window", type=int, default=1_000)
    parser.add_argument("--freeze-min-episode", type=int, default=5_000)
    parser.add_argument("--freeze-win-rate", type=float, default=0.75)
    parser.add_argument("--freeze-draw-rate-max", type=float, default=0.15)
    parser.add_argument("--freeze-queen-loss-rate-max", type=float, default=0.10)
    parser.add_argument("--early-stop-after-freeze", type=int, default=2_000)
    parser.add_argument("--no-curriculum", action="store_true")
    parser.add_argument("--curriculum-easy-fraction", type=float, default=0.35)
    parser.add_argument("--curriculum-medium-fraction", type=float, default=0.40)
    parser.add_argument("--replay-size", type=int, default=50_000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--warmup-steps", type=int, default=64)
    parser.add_argument("--updates-per-step", type=int, default=1)
    parser.add_argument("--train-interval", type=int, default=8)
    parser.add_argument("--model-out", default="artifacts/kqk_neural_q.pt")
    parser.add_argument("--log-out", default="artifacts/kqk_neural_training.csv")
    parser.add_argument("--no-progress", action="store_true")

    parser.add_argument("--board-size", type=int, default=720)
    parser.add_argument("--rolling-window", type=int, default=100)
    parser.add_argument("--render-every-step", type=int, default=2)
    parser.add_argument("--update-plots-every-episode", type=int, default=2)
    parser.add_argument("--max-fps", type=float, default=20.0)
    parser.add_argument("--max-queue-events", type=int, default=2_000)
    return parser.parse_args()


def build_defender(name: str, oracle: SyzygyOracle):
    if name == "random":
        return RandomDefenderPolicy()
    if name == "heuristic":
        return HeuristicDefenderPolicy()
    return SyzygyDefenderPolicy(oracle)


def syzygy_details(args: argparse.Namespace) -> tuple[SyzygyOracle, dict[str, Any]]:
    paths, file_count = discover_syzygy_paths(
        explicit_paths=args.syzygy_path,
        auto_discover=not args.disable_auto_syzygy,
        cwd=Path.cwd(),
    )
    oracle = SyzygyOracle(paths)
    info: dict[str, Any] = {
        "syzygy_loaded": oracle.available,
        "syzygy_files": file_count,
        "syzygy_dirs": paths,
    }
    return oracle, info


def enforce_required_syzygy(args: argparse.Namespace, info: dict[str, Any]) -> None:
    if not args.require_syzygy:
        return
    if bool(info.get("syzygy_loaded", False)):
        return
    print(
        "[warn] Syzygy was requested but no valid tablebases were loaded. "
        "Continuing with fallback behavior (heuristic where needed). "
        "Add --syzygy-path with .rtbw/.rtbz files to enable Syzygy."
    )


def summarize_rows(rows: list[dict[str, float | int | str]]) -> dict[str, Any]:
    outcomes = Counter(str(row["outcome"]) for row in rows)
    return {
        "checkmates": outcomes.get("checkmate", 0),
        "draws": outcomes.get("draw", 0),
        "queen_losses": outcomes.get("queen_lost", 0),
        "max_length": outcomes.get("max_length", 0),
        "losses": outcomes.get("loss", 0),
    }


def summarize_log(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = [row for row in reader]
    if not rows:
        return {}
    outcomes = Counter(row.get("outcome", "") for row in rows)
    epsilon = float(rows[-1].get("epsilon", "0") or 0.0)
    return {
        "episodes_logged": len(rows),
        "checkmates": outcomes.get("checkmate", 0),
        "draws": outcomes.get("draw", 0),
        "queen_losses": outcomes.get("queen_lost", 0),
        "max_length": outcomes.get("max_length", 0),
        "losses": outcomes.get("loss", 0),
        "final_epsilon": epsilon,
    }


def run_headless_train(args: argparse.Namespace, defender) -> tuple[NeuralQAgent, list[dict[str, float | int | str]]]:
    progress = None
    on_episode_end = None
    if not args.no_progress:
        progress = tqdm(total=args.episodes, desc="Train KQK Neural", unit="ep", dynamic_ncols=True)

        def _on_episode_end(row: dict[str, float | int | str]) -> None:
            assert progress is not None
            progress.update(1)
            progress.set_postfix(
                reward=f"{float(row['reward']):+.3f}",
                loss=f"{float(row['loss']):.4f}",
                eps=f"{float(row['epsilon']):.3f}",
                outcome=str(row["outcome"]),
            )

        on_episode_end = _on_episode_end

    try:
        agent, rows = train_kqk_neural(
            episodes=args.episodes,
            defender_policy=defender,
            seed=args.seed,
            alpha=args.alpha,
            alpha_decay=args.alpha_decay,
            alpha_min=args.alpha_min,
            gamma=args.gamma,
            epsilon=args.epsilon,
            epsilon_decay=args.epsilon_decay,
            epsilon_min=args.epsilon_min,
            claim_draw_by_repetition=args.enable_threefold_draw,
            stability_window=args.stability_window,
            freeze_min_episode=args.freeze_min_episode,
            freeze_win_rate=args.freeze_win_rate,
            freeze_draw_rate_max=args.freeze_draw_rate_max,
            freeze_queen_loss_rate_max=args.freeze_queen_loss_rate_max,
            early_stop_after_freeze=args.early_stop_after_freeze,
            curriculum=not args.no_curriculum,
            curriculum_easy_fraction=args.curriculum_easy_fraction,
            curriculum_medium_fraction=args.curriculum_medium_fraction,
            replay_size=args.replay_size,
            batch_size=args.batch_size,
            warmup_steps=args.warmup_steps,
            updates_per_step=args.updates_per_step,
            train_interval=args.train_interval,
            on_episode_end=on_episode_end,
        )
    finally:
        if progress is not None:
            progress.close()
    return agent, rows


def run_eval(args: argparse.Namespace, oracle: SyzygyOracle, model_path: Path) -> dict[str, Any]:
    agent = NeuralQAgent.load(model_path, seed=args.seed)
    metrics = evaluate_kqk(
        episodes=args.eval_episodes,
        attacker_kind="neural",
        defender_policy=build_defender(args.eval_defender, oracle),
        neural_agent=agent,
        oracle=oracle,
        seed=args.seed,
    )
    metrics["attacker"] = "neural"
    metrics["defender"] = args.eval_defender
    metrics["model_path"] = str(model_path)
    metrics["syzygy_loaded"] = oracle.available
    return metrics


def main() -> None:
    args = parse_args()
    model_path = Path(args.model_out)
    log_path = Path(args.log_out)
    oracle, syzygy_info = syzygy_details(args)
    enforce_required_syzygy(args, syzygy_info)

    output: dict[str, Any] = {
        "mode": args.mode,
        "agent": "neural",
        **syzygy_info,
        "model_path": str(model_path),
        "log_path": str(log_path),
    }

    if args.mode in {"live", "live-eval"}:
        run_live_viewer(args)
        output["train"] = summarize_log(log_path)
    elif args.mode in {"train", "train-eval"}:
        defender = build_defender(args.defender, oracle)
        agent, rows = run_headless_train(args, defender)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        agent.save(model_path)
        write_training_log(log_path, rows)
        output["train"] = {"episodes": args.episodes, "defender": args.defender, **summarize_rows(rows)}

    if args.mode in {"eval", "train-eval", "live-eval"}:
        output["eval"] = run_eval(args, oracle, model_path)

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
