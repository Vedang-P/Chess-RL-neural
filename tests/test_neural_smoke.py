from __future__ import annotations

import importlib.util

import pytest

from chess_rl.policies import RandomDefenderPolicy
from chess_rl.train import train_kqk_neural


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="PyTorch not installed")
def test_short_neural_training_run() -> None:
    agent, rows = train_kqk_neural(
        episodes=3,
        defender_policy=RandomDefenderPolicy(),
        seed=2,
    )
    assert rows
    assert hasattr(agent, "model")
