from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING, Literal

from chess_rl.env import KQKEnv
from chess_rl.policies import DefenderPolicy, GreedyAttackerPolicy, RandomAttackerPolicy
from chess_rl.syzygy import SyzygyOracle

if TYPE_CHECKING:
    from chess_rl.neural_agent import NeuralQAgent

AttackerKind = Literal["neural", "random", "greedy"]


def evaluate_kqk(
    episodes: int,
    attacker_kind: AttackerKind,
    defender_policy: DefenderPolicy,
    neural_agent: NeuralQAgent | None = None,
    oracle: SyzygyOracle | None = None,
    seed: int = 0,
) -> dict[str, float | int]:
    if attacker_kind == "neural" and neural_agent is None:
        raise ValueError("Neural evaluation requires a trained NeuralQAgent.")

    env = KQKEnv(defender_policy=defender_policy, seed=seed)
    random_attacker = RandomAttackerPolicy()
    greedy_attacker = GreedyAttackerPolicy()

    outcomes: Counter[str] = Counter()
    total_rewards = 0.0
    mate_lengths: list[int] = []
    optimal_checks = 0
    optimal_hits = 0

    for _ in range(episodes):
        state = env.reset()
        done = False

        while not done:
            legal = env.legal_action_ucis()
            if not legal:
                break

            board_for_eval = env.board.copy(stack=False)
            if oracle and oracle.available:
                best_moves = oracle.optimal_moves(board_for_eval)
                if best_moves is not None:
                    optimal_checks += 1
            else:
                best_moves = None

            if attacker_kind == "neural":
                assert neural_agent is not None
                action_uci = neural_agent.select_action(state, env.board, legal, greedy_only=True)
            elif attacker_kind == "random":
                action_uci = random_attacker.select_move(env.board, env.rng).uci()
            else:
                action_uci = greedy_attacker.select_move(env.board, env.rng).uci()

            if best_moves is not None and any(move.uci() == action_uci for move in best_moves):
                optimal_hits += 1

            result = env.step_uci(action_uci)
            total_rewards += result.reward
            state = result.state
            done = result.done
            if done:
                outcome = str(result.info["outcome"])
                outcomes[outcome] += 1
                if outcome == "checkmate":
                    mate_lengths.append(int(result.info["white_moves"]))

    wins = outcomes.get("checkmate", 0)
    draws = outcomes.get("draw", 0)
    losses = outcomes.get("loss", 0)
    maxed = outcomes.get("max_length", 0)
    queen_losses = outcomes.get("queen_lost", 0)

    return {
        "episodes": episodes,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "queen_losses": queen_losses,
        "max_length": maxed,
        "win_rate": wins / episodes if episodes else 0.0,
        "avg_reward": total_rewards / episodes if episodes else 0.0,
        "avg_mate_length": (sum(mate_lengths) / len(mate_lengths)) if mate_lengths else 0.0,
        "optimal_move_rate": (optimal_hits / optimal_checks) if optimal_checks else 0.0,
    }

