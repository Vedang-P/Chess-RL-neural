from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable

import chess
import chess.syzygy


class SyzygyOracle:
    def __init__(self, paths: Iterable[str] | None):
        self._tablebase: chess.syzygy.Tablebase | None = None
        if not paths:
            return
        tablebase = chess.syzygy.Tablebase()
        valid_dirs = 0
        for raw_path in paths:
            path = Path(raw_path).expanduser()
            if path.is_dir():
                tablebase.add_directory(path)
                valid_dirs += 1
        if valid_dirs > 0:
            self._tablebase = tablebase

    @property
    def available(self) -> bool:
        return self._tablebase is not None

    def probe_wdl(self, board: chess.Board) -> int | None:
        if not self._tablebase:
            return None
        try:
            return self._tablebase.probe_wdl(board)
        except (KeyError, chess.syzygy.MissingTableError, ValueError):
            return None

    def optimal_moves(self, board: chess.Board) -> list[chess.Move] | None:
        if not self._tablebase:
            return None

        best: list[chess.Move] = []
        best_wdl: int | None = None

        for move in board.legal_moves:
            board.push(move)
            child_wdl = self.probe_wdl(board)
            board.pop()
            if child_wdl is None:
                continue

            current_side_wdl = -child_wdl
            if best_wdl is None or current_side_wdl > best_wdl:
                best_wdl = current_side_wdl
                best = [move]
            elif current_side_wdl == best_wdl:
                best.append(move)

        return best if best else None

    def choose_optimal_move(self, board: chess.Board, rng: random.Random) -> chess.Move | None:
        moves = self.optimal_moves(board)
        if not moves:
            return None
        return rng.choice(moves)
