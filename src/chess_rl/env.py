from __future__ import annotations

import random
from dataclasses import dataclass

import chess

from chess_rl.endgames.kqk import (
    KQKState,
    abstract_kqk_state,
    confinement_potential,
    corner_distance,
    edge_distance,
    defender_mobility,
    random_kqk_position,
    white_has_single_queen,
)
from chess_rl.policies import DefenderPolicy, RandomDefenderPolicy


@dataclass
class StepResult:
    state: KQKState
    reward: float
    done: bool
    info: dict[str, object]


@dataclass(frozen=True)
class RewardConfig:
    mate_reward: float = 10.0
    mate_speed_bonus: float = 2.0
    draw_penalty: float = -2.0
    queen_loss_penalty: float = -3.0
    loss_penalty: float = -3.0
    max_length_penalty: float = -2.5
    step_penalty: float = -0.02
    mobility_weight: float = 0.08
    edge_weight: float = 0.05
    corner_weight: float = 0.03
    check_bonus: float = 0.04
    stall_penalty: float = -0.05
    progress_tolerance: float = 1e-6


class KQKEnv:
    def __init__(
        self,
        defender_policy: DefenderPolicy | None = None,
        seed: int = 0,
        max_white_moves: int = 75,
        step_penalty: float = -0.02,
        shaping_weight: float = 0.20,
        claim_draw_by_repetition: bool = False,
        reward_config: RewardConfig | None = None,
    ):
        self.rng = random.Random(seed)
        self.defender_policy = defender_policy or RandomDefenderPolicy()
        self.max_white_moves = max_white_moves
        # Kept for backward compatibility with existing scripts.
        self.step_penalty = step_penalty
        self.shaping_weight = shaping_weight
        self.reward_config = reward_config or RewardConfig(step_penalty=step_penalty)
        self.claim_draw_by_repetition = claim_draw_by_repetition
        self.board = random_kqk_position(self.rng, white_to_move=True)
        self.white_moves_played = 0
        self.no_progress_streak = 0

    def reset(self, board: chess.Board | None = None) -> KQKState:
        if board is None:
            self.board = random_kqk_position(self.rng, white_to_move=True)
        else:
            self.board = board.copy(stack=False)
        if self.board.turn != chess.WHITE:
            raise ValueError("Environment expects white to move on reset.")
        self.white_moves_played = 0
        self.no_progress_streak = 0
        return abstract_kqk_state(self.board)

    def legal_action_ucis(self) -> list[str]:
        if self.board.turn != chess.WHITE:
            return []
        return [move.uci() for move in self.board.legal_moves]

    def _is_draw(self) -> bool:
        board = self.board
        if board.is_stalemate() or board.is_insufficient_material() or board.can_claim_fifty_moves():
            return True
        # Threefold claim probing is expensive in python-chess because it simulates candidate move lines.
        if self.claim_draw_by_repetition and board.can_claim_threefold_repetition():
            return True
        return False

    def step_uci(self, move_uci: str) -> StepResult:
        return self.step(chess.Move.from_uci(move_uci))

    def _speed_bonus(self) -> float:
        fraction_remaining = max(0.0, (self.max_white_moves - self.white_moves_played) / max(1, self.max_white_moves))
        return self.reward_config.mate_speed_bonus * fraction_remaining

    def _progress_features(self) -> tuple[int, int, int]:
        black_king = self.board.king(chess.BLACK)
        if black_king is None:
            return (0, 0, 0)
        return (defender_mobility(self.board), edge_distance(black_king), corner_distance(black_king))

    def step(self, move: chess.Move) -> StepResult:
        if self.board.turn != chess.WHITE:
            raise RuntimeError("Agent can only act on white turns.")
        if move not in self.board.legal_moves:
            raise ValueError(f"Illegal move: {move.uci()}")

        pre_potential = confinement_potential(self.board)
        pre_mobility, pre_edge, pre_corner = self._progress_features()
        gave_check = self.board.gives_check(move)
        self.board.push(move)
        self.white_moves_played += 1

        if not white_has_single_queen(self.board):
            return StepResult(
                state=(0, 0, 0, 0, 0, 0, 0, 1),
                reward=self.reward_config.queen_loss_penalty,
                done=True,
                info={"outcome": "queen_lost", "white_moves": self.white_moves_played, "fen": self.board.fen()},
            )

        if self.board.is_checkmate():
            return StepResult(
                state=abstract_kqk_state(self.board),
                reward=self.reward_config.mate_reward + self._speed_bonus(),
                done=True,
                info={"outcome": "checkmate", "white_moves": self.white_moves_played, "fen": self.board.fen()},
            )

        if self._is_draw():
            return StepResult(
                state=abstract_kqk_state(self.board),
                reward=self.reward_config.draw_penalty,
                done=True,
                info={"outcome": "draw", "white_moves": self.white_moves_played, "fen": self.board.fen()},
            )

        defender_move = self.defender_policy.select_move(self.board, self.rng)
        self.board.push(defender_move)

        if not white_has_single_queen(self.board):
            return StepResult(
                state=(0, 0, 0, 0, 0, 0, 0, 1),
                reward=self.reward_config.queen_loss_penalty,
                done=True,
                info={"outcome": "queen_lost", "white_moves": self.white_moves_played, "fen": self.board.fen()},
            )

        if self.board.is_checkmate():
            return StepResult(
                state=abstract_kqk_state(self.board),
                reward=self.reward_config.loss_penalty,
                done=True,
                info={"outcome": "loss", "white_moves": self.white_moves_played, "fen": self.board.fen()},
            )

        if self._is_draw():
            return StepResult(
                state=abstract_kqk_state(self.board),
                reward=self.reward_config.draw_penalty,
                done=True,
                info={"outcome": "draw", "white_moves": self.white_moves_played, "fen": self.board.fen()},
            )

        post_mobility, post_edge, post_corner = self._progress_features()
        post_potential = confinement_potential(self.board)
        progress = (
            self.reward_config.mobility_weight * (pre_mobility - post_mobility)
            + self.reward_config.edge_weight * (pre_edge - post_edge)
            + self.reward_config.corner_weight * (pre_corner - post_corner)
            + self.shaping_weight * (post_potential - pre_potential)
        )
        if progress <= self.reward_config.progress_tolerance:
            self.no_progress_streak += 1
            progress += self.reward_config.stall_penalty * min(3, self.no_progress_streak)
        else:
            self.no_progress_streak = 0

        reward = self.reward_config.step_penalty + progress
        if gave_check:
            reward += self.reward_config.check_bonus

        done = False
        outcome = "ongoing"
        if self.white_moves_played >= self.max_white_moves:
            done = True
            outcome = "max_length"
            reward += self.reward_config.max_length_penalty

        return StepResult(
            state=abstract_kqk_state(self.board),
            reward=reward,
            done=done,
            info={
                "outcome": outcome,
                "white_moves": self.white_moves_played,
                "fen": self.board.fen(),
                "progress": progress,
            },
        )
