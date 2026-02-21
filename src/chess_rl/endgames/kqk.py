from __future__ import annotations

import random
from typing import Literal, Tuple

import chess

KQKState = Tuple[int, int, int, int, int, int, int, int]


def chebyshev_distance(a: chess.Square, b: chess.Square) -> int:
    return max(abs(chess.square_file(a) - chess.square_file(b)), abs(chess.square_rank(a) - chess.square_rank(b)))


def manhattan_distance(a: chess.Square, b: chess.Square) -> int:
    return abs(chess.square_file(a) - chess.square_file(b)) + abs(chess.square_rank(a) - chess.square_rank(b))


def edge_distance(square: chess.Square) -> int:
    file_idx = chess.square_file(square)
    rank_idx = chess.square_rank(square)
    return min(file_idx, 7 - file_idx, rank_idx, 7 - rank_idx)


def corner_distance(square: chess.Square) -> int:
    corners = [chess.A1, chess.A8, chess.H1, chess.H8]
    return min(manhattan_distance(square, corner) for corner in corners)


def is_opposition(white_king: chess.Square, black_king: chess.Square) -> bool:
    same_file = chess.square_file(white_king) == chess.square_file(black_king)
    same_rank = chess.square_rank(white_king) == chess.square_rank(black_king)
    return (same_file or same_rank) and chebyshev_distance(white_king, black_king) == 2


def _white_queen_square(board: chess.Board) -> chess.Square:
    squares = list(board.pieces(chess.QUEEN, chess.WHITE))
    if len(squares) != 1:
        raise ValueError("KQK abstraction expects exactly one white queen.")
    return squares[0]


def white_has_single_queen(board: chess.Board) -> bool:
    return len(list(board.pieces(chess.QUEEN, chess.WHITE))) == 1


def defender_mobility(board: chess.Board) -> int:
    copy_board = board.copy(stack=False)
    copy_board.turn = chess.BLACK
    return copy_board.legal_moves.count()


def abstract_kqk_state(board: chess.Board) -> KQKState:
    white_king = board.king(chess.WHITE)
    black_king = board.king(chess.BLACK)
    queen = _white_queen_square(board)
    if white_king is None or black_king is None:
        raise ValueError("Both kings must be present.")

    king_dist = chebyshev_distance(white_king, black_king)
    queen_dist = chebyshev_distance(queen, black_king)
    black_edge = edge_distance(black_king)
    black_corner = corner_distance(black_king)
    opposition = int(is_opposition(white_king, black_king))
    queen_protected = int(chebyshev_distance(white_king, queen) <= 1)
    black_moves = defender_mobility(board)
    black_mobility_bucket = min(3, black_moves // 3)
    side_to_move = int(board.turn == chess.WHITE)

    return (
        king_dist,
        queen_dist,
        black_edge,
        black_corner,
        opposition,
        queen_protected,
        black_mobility_bucket,
        side_to_move,
    )


def confinement_potential(board: chess.Board) -> float:
    black_king = board.king(chess.BLACK)
    white_king = board.king(chess.WHITE)
    queen = _white_queen_square(board)
    if black_king is None or white_king is None:
        return 0.0

    edge_score = (3 - edge_distance(black_king)) / 3.0
    corner_score = (7 - corner_distance(black_king)) / 7.0
    queen_pressure = (7 - chebyshev_distance(queen, black_king)) / 7.0
    king_support = (7 - chebyshev_distance(white_king, black_king)) / 7.0
    mobility_score = 1.0 - min(defender_mobility(board), 8) / 8.0

    return 0.30 * edge_score + 0.25 * corner_score + 0.20 * queen_pressure + 0.15 * mobility_score + 0.10 * king_support


def random_kqk_position(rng: random.Random, white_to_move: bool = True, max_tries: int = 50_000) -> chess.Board:
    for _ in range(max_tries):
        white_king, white_queen, black_king = rng.sample(range(64), 3)
        board = chess.Board(None)
        board.set_piece_at(white_king, chess.Piece(chess.KING, chess.WHITE))
        board.set_piece_at(white_queen, chess.Piece(chess.QUEEN, chess.WHITE))
        board.set_piece_at(black_king, chess.Piece(chess.KING, chess.BLACK))
        board.turn = chess.WHITE if white_to_move else chess.BLACK
        board.castling_rights = chess.BB_EMPTY
        board.ep_square = None
        board.halfmove_clock = 0
        board.fullmove_number = 1

        if not board.is_valid():
            continue
        if board.is_checkmate() or board.is_stalemate() or board.is_insufficient_material():
            continue
        return board
    raise RuntimeError("Failed to sample a non-terminal legal KQK position.")


CurriculumPhase = Literal["easy", "medium", "full"]


def curriculum_phase(episode: int, total_episodes: int, easy_fraction: float = 0.35, medium_fraction: float = 0.40) -> CurriculumPhase:
    if total_episodes <= 0:
        return "full"
    frac = episode / total_episodes
    if frac <= easy_fraction:
        return "easy"
    if frac <= easy_fraction + medium_fraction:
        return "medium"
    return "full"


def random_kqk_curriculum_position(
    rng: random.Random,
    phase: CurriculumPhase,
    white_to_move: bool = True,
    max_tries: int = 5_000,
) -> chess.Board:
    for _ in range(max_tries):
        board = random_kqk_position(rng=rng, white_to_move=white_to_move, max_tries=200)
        wk = board.king(chess.WHITE)
        bk = board.king(chess.BLACK)
        if wk is None or bk is None:
            continue
        queen_squares = list(board.pieces(chess.QUEEN, chess.WHITE))
        if len(queen_squares) != 1:
            continue
        q = queen_squares[0]

        king_dist = chebyshev_distance(wk, bk)
        queen_dist = chebyshev_distance(q, bk)
        edge = edge_distance(bk)
        queen_attacked = board.is_attacked_by(chess.BLACK, q)

        if phase == "easy":
            if edge > 1:
                continue
            if king_dist > 3:
                continue
            if queen_dist > 4:
                continue
            if queen_attacked:
                continue
        elif phase == "medium":
            if edge > 2:
                continue
            if king_dist > 4:
                continue
            if queen_dist > 5:
                continue
        return board
    return random_kqk_position(rng=rng, white_to_move=white_to_move)
