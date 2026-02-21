from __future__ import annotations

from typing import Iterable

import chess

from chess_rl.endgames.kqk import KQKState

STATE_DIM = 8
ACTION_DIM = 11
STATE_ACTION_DIM = STATE_DIM + ACTION_DIM


def encode_state(state: KQKState) -> list[float]:
    king_dist, queen_dist, edge, corner, opposition, queen_protected, mobility_bucket, side_to_move = state
    return [
        king_dist / 7.0,
        queen_dist / 7.0,
        edge / 3.0,
        corner / 7.0,
        float(opposition),
        float(queen_protected),
        mobility_bucket / 3.0,
        float(side_to_move),
    ]


def encode_action(board: chess.Board, move: chess.Move) -> list[float]:
    piece = board.piece_at(move.from_square)
    is_king = 1.0 if piece and piece.piece_type == chess.KING else 0.0
    is_queen = 1.0 if piece and piece.piece_type == chess.QUEEN else 0.0
    from_file = chess.square_file(move.from_square) / 7.0
    from_rank = chess.square_rank(move.from_square) / 7.0
    to_file = chess.square_file(move.to_square) / 7.0
    to_rank = chess.square_rank(move.to_square) / 7.0
    delta_file = (chess.square_file(move.to_square) - chess.square_file(move.from_square)) / 7.0
    delta_rank = (chess.square_rank(move.to_square) - chess.square_rank(move.from_square)) / 7.0
    move_len = max(abs(delta_file), abs(delta_rank))
    gives_check = 1.0 if board.gives_check(move) else 0.0
    is_capture = 1.0 if board.is_capture(move) else 0.0
    return [
        is_king,
        is_queen,
        from_file,
        from_rank,
        to_file,
        to_rank,
        delta_file,
        delta_rank,
        move_len,
        gives_check,
        is_capture,
    ]


def encode_state_action(state: KQKState, board: chess.Board, move: chess.Move) -> list[float]:
    return encode_state(state) + encode_action(board, move)


def encode_state_action_batch(state: KQKState, board: chess.Board, moves: Iterable[chess.Move]) -> list[list[float]]:
    return [encode_state_action(state, board, move) for move in moves]
