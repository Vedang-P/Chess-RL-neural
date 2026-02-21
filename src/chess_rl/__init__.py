"""Chess endgame RL package."""

from chess_rl.env import KQKEnv

__all__ = ["KQKEnv"]

try:
    from chess_rl.neural_agent import NeuralQAgent

    __all__.append("NeuralQAgent")
except ImportError:
    pass
