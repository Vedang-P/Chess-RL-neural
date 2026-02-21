from __future__ import annotations

import random

import chess

from chess_rl.endgames.kqk import abstract_kqk_state, random_kqk_position
from chess_rl.features import ACTION_DIM, STATE_ACTION_DIM, STATE_DIM, encode_action, encode_state, encode_state_action


def test_feature_dimensions() -> None:
    board = random_kqk_position(rng=random.Random(1))
    state = abstract_kqk_state(board)
    move = next(iter(board.legal_moves))

    s = encode_state(state)
    a = encode_action(board, move)
    sa = encode_state_action(state, board, move)

    assert len(s) == STATE_DIM
    assert len(a) == ACTION_DIM
    assert len(sa) == STATE_ACTION_DIM


def test_action_piece_flags() -> None:
    board = chess.Board(None)
    board.set_piece_at(chess.E1, chess.Piece(chess.KING, chess.WHITE))
    board.set_piece_at(chess.D1, chess.Piece(chess.QUEEN, chess.WHITE))
    board.set_piece_at(chess.E8, chess.Piece(chess.KING, chess.BLACK))
    board.turn = chess.WHITE

    king_move = chess.Move.from_uci("e1e2")
    queen_move = chess.Move.from_uci("d1d5")

    king_features = encode_action(board, king_move)
    queen_features = encode_action(board, queen_move)

    assert king_features[0] == 1.0
    assert king_features[1] == 0.0
    assert queen_features[0] == 0.0
    assert queen_features[1] == 1.0
