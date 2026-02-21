from __future__ import annotations

import chess

from chess_rl.env import KQKEnv
from chess_rl.policies import RandomDefenderPolicy


class CaptureQueenPolicy:
    def select_move(self, board: chess.Board, rng):  # noqa: ANN001
        for move in board.legal_moves:
            if board.is_capture(move):
                captured = board.piece_at(move.to_square)
                if captured and captured.piece_type == chess.QUEEN and captured.color == chess.WHITE:
                    return move
        return next(iter(board.legal_moves))


def test_reset_and_step_are_valid() -> None:
    env = KQKEnv(defender_policy=RandomDefenderPolicy(), seed=7)
    state = env.reset()
    assert len(state) == 8

    legal = env.legal_action_ucis()
    assert legal

    result = env.step_uci(legal[0])
    assert len(result.state) == 8
    assert isinstance(result.reward, float)
    assert isinstance(result.done, bool)
    assert "outcome" in result.info


def test_queen_loss_is_terminal() -> None:
    board = chess.Board(None)
    board.set_piece_at(chess.A1, chess.Piece(chess.KING, chess.WHITE))
    board.set_piece_at(chess.C2, chess.Piece(chess.QUEEN, chess.WHITE))
    board.set_piece_at(chess.D3, chess.Piece(chess.KING, chess.BLACK))
    board.turn = chess.WHITE
    board.castling_rights = chess.BB_EMPTY

    env = KQKEnv(defender_policy=CaptureQueenPolicy(), seed=1)
    env.reset(board)
    result = env.step_uci("a1a2")
    assert result.done is True
    assert result.info["outcome"] == "queen_lost"

