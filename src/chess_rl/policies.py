from __future__ import annotations

import random
from typing import Protocol

import chess

from chess_rl.endgames.kqk import confinement_potential, corner_distance, edge_distance
from chess_rl.syzygy import SyzygyOracle


class DefenderPolicy(Protocol):
    def select_move(self, board: chess.Board, rng: random.Random) -> chess.Move:
        ...


class RandomDefenderPolicy:
    def select_move(self, board: chess.Board, rng: random.Random) -> chess.Move:
        moves = list(board.legal_moves)
        return rng.choice(moves)


class HeuristicDefenderPolicy:
    def _score(self, board: chess.Board) -> float:
        black_king = board.king(chess.BLACK)
        if black_king is None:
            return -1_000.0
        edge = edge_distance(black_king)
        corner = corner_distance(black_king)
        copy_board = board.copy(stack=False)
        copy_board.turn = chess.BLACK
        black_mobility = copy_board.legal_moves.count()
        return 0.45 * edge + 0.45 * corner + 0.10 * black_mobility

    def select_move(self, board: chess.Board, rng: random.Random) -> chess.Move:
        best_score = float("-inf")
        best_moves: list[chess.Move] = []
        for move in board.legal_moves:
            board.push(move)
            score = self._score(board)
            board.pop()
            if score > best_score:
                best_score = score
                best_moves = [move]
            elif score == best_score:
                best_moves.append(move)
        return rng.choice(best_moves)


class SyzygyDefenderPolicy:
    def __init__(self, oracle: SyzygyOracle, fallback: DefenderPolicy | None = None):
        self.oracle = oracle
        self.fallback = fallback or HeuristicDefenderPolicy()

    def select_move(self, board: chess.Board, rng: random.Random) -> chess.Move:
        move = self.oracle.choose_optimal_move(board, rng)
        if move is not None:
            return move
        return self.fallback.select_move(board, rng)


class RandomAttackerPolicy:
    def select_move(self, board: chess.Board, rng: random.Random) -> chess.Move:
        return rng.choice(list(board.legal_moves))


class GreedyAttackerPolicy:
    def select_move(self, board: chess.Board, rng: random.Random) -> chess.Move:
        current = confinement_potential(board)
        best_score = float("-inf")
        best_moves: list[chess.Move] = []
        for move in board.legal_moves:
            board.push(move)
            if board.is_checkmate():
                score = 10.0
            elif board.is_stalemate():
                score = -10.0
            else:
                score = confinement_potential(board) - current
            board.pop()
            if score > best_score:
                best_score = score
                best_moves = [move]
            elif score == best_score:
                best_moves.append(move)
        return rng.choice(best_moves)
