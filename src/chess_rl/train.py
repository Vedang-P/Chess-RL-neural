from __future__ import annotations

import csv
from collections import Counter, deque
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from chess_rl.endgames.kqk import curriculum_phase, random_kqk_curriculum_position
from chess_rl.env import KQKEnv
from chess_rl.policies import DefenderPolicy

if TYPE_CHECKING:
    from chess_rl.neural_agent import NeuralQAgent


def write_training_log(path: str | Path, rows: list[dict[str, float | int | str]]) -> None:
    if not rows:
        return
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def train_kqk_neural(
    episodes: int,
    defender_policy: DefenderPolicy,
    seed: int = 0,
    alpha: float = 3e-4,
    alpha_decay: float = 0.9998,
    alpha_min: float = 1e-4,
    gamma: float = 0.90,
    epsilon: float = 0.30,
    epsilon_decay: float = 0.99985,
    epsilon_min: float = 0.02,
    claim_draw_by_repetition: bool = False,
    stability_window: int = 1_000,
    freeze_min_episode: int = 5_000,
    freeze_win_rate: float = 0.75,
    freeze_draw_rate_max: float = 0.15,
    freeze_queen_loss_rate_max: float = 0.10,
    early_stop_after_freeze: int = 2_000,
    curriculum: bool = True,
    curriculum_easy_fraction: float = 0.35,
    curriculum_medium_fraction: float = 0.40,
    replay_size: int = 50_000,
    batch_size: int = 16,
    warmup_steps: int = 64,
    updates_per_step: int = 1,
    train_interval: int = 8,
    on_episode_end: Callable[[dict[str, float | int | str]], None] | None = None,
) -> tuple[NeuralQAgent, list[dict[str, float | int | str]]]:
    from chess_rl.neural_agent import NeuralQAgent

    env = KQKEnv(defender_policy=defender_policy, seed=seed, claim_draw_by_repetition=claim_draw_by_repetition)
    agent = NeuralQAgent(
        alpha=alpha,
        alpha_decay=alpha_decay,
        alpha_min=alpha_min,
        gamma=gamma,
        epsilon=epsilon,
        epsilon_decay=epsilon_decay,
        epsilon_min=epsilon_min,
        seed=seed,
        replay_size=replay_size,
        batch_size=batch_size,
        warmup_steps=warmup_steps,
        updates_per_step=updates_per_step,
        train_interval=train_interval,
    )

    recent_outcomes: deque[str] = deque(maxlen=max(50, stability_window))
    frozen = False
    frozen_episode: int | None = None
    rows: list[dict[str, float | int | str]] = []
    for episode in range(1, episodes + 1):
        if curriculum:
            phase = curriculum_phase(
                episode=episode,
                total_episodes=episodes,
                easy_fraction=curriculum_easy_fraction,
                medium_fraction=curriculum_medium_fraction,
            )
            board = random_kqk_curriculum_position(env.rng, phase=phase, white_to_move=True)
            state = env.reset(board=board)
        else:
            state = env.reset()

        done = False
        total_reward = 0.0
        total_loss = 0.0
        update_steps = 0
        outcome = "ongoing"
        white_moves = 0

        while not done:
            legal_actions = env.legal_action_ucis()
            if not legal_actions:
                break

            board_before = env.board.copy(stack=False)
            action = agent.select_action(state, board_before, legal_actions, greedy_only=False)
            result = env.step_uci(action)
            board_after = env.board.copy(stack=False)
            next_legal = env.legal_action_ucis() if not result.done else []

            if not frozen:
                loss = agent.update(
                    state=state,
                    board_before=board_before,
                    action_uci=action,
                    reward=result.reward,
                    next_state=result.state,
                    board_after=board_after,
                    next_legal_actions=next_legal,
                    done=result.done,
                )
                total_loss += loss
                update_steps += 1

            state = result.state
            total_reward += result.reward
            done = result.done
            outcome = str(result.info["outcome"])
            white_moves = int(result.info["white_moves"])

        recent_outcomes.append(outcome)
        if not frozen:
            agent.decay_exploration()
            agent.decay_learning_rate()

        if (not frozen) and episode >= max(1, freeze_min_episode) and len(recent_outcomes) == recent_outcomes.maxlen:
            counts = Counter(recent_outcomes)
            window_size = len(recent_outcomes)
            win_rate = counts.get("checkmate", 0) / window_size
            draw_rate = counts.get("draw", 0) / window_size
            queen_loss_rate = counts.get("queen_lost", 0) / window_size
            if (
                win_rate >= freeze_win_rate
                and draw_rate <= freeze_draw_rate_max
                and queen_loss_rate <= freeze_queen_loss_rate_max
            ):
                frozen = True
                frozen_episode = episode
                agent.epsilon = agent.epsilon_min

        row = {
            "episode": episode,
            "reward": round(total_reward, 6),
            "loss": round(total_loss / update_steps, 6) if update_steps else 0.0,
            "white_moves": white_moves,
            "outcome": outcome,
            "epsilon": round(agent.epsilon, 6),
            "alpha": round(agent.alpha, 6),
            "frozen": int(frozen),
            "curriculum": int(curriculum),
        }
        rows.append(row)
        if on_episode_end is not None:
            on_episode_end(row)
        if frozen and frozen_episode is not None and episode - frozen_episode >= max(1, early_stop_after_freeze):
            break

    return agent, rows

