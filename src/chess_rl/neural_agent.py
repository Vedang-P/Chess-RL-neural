from __future__ import annotations

import random
from collections import deque
from pathlib import Path
from typing import Iterable

import chess

from chess_rl.endgames.kqk import KQKState
from chess_rl.features import STATE_ACTION_DIM, encode_state_action, encode_state_action_batch

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover - exercised only when torch missing.
    raise ImportError(
        "NeuralQAgent requires PyTorch. Install it with `pip install -e \".[rl]\"`."
    ) from exc


class QRegressor(nn.Module):
    def __init__(self, hidden_sizes: tuple[int, int] = (128, 64)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(STATE_ACTION_DIM, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class NeuralQAgent:
    def __init__(
        self,
        alpha: float = 1e-3,
        alpha_decay: float = 0.9998,
        alpha_min: float = 1e-4,
        gamma: float = 0.99,
        epsilon: float = 0.20,
        epsilon_decay: float = 0.9995,
        epsilon_min: float = 0.02,
        seed: int = 0,
        hidden_sizes: tuple[int, int] = (128, 64),
        target_update_interval: int = 200,
        q_clip: float = 12.0,
        replay_size: int = 50_000,
        batch_size: int = 16,
        warmup_steps: int = 64,
        updates_per_step: int = 1,
        train_interval: int = 8,
    ):
        self.alpha = alpha
        self.alpha_decay = alpha_decay
        self.alpha_min = alpha_min
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.target_update_interval = target_update_interval
        self.q_clip = q_clip
        self.batch_size = batch_size
        self.warmup_steps = warmup_steps
        self.updates_per_step = updates_per_step
        self.train_interval = train_interval
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(seed)
        torch.manual_seed(seed)
        self.rng = random.Random(seed)
        self.replay: deque[tuple[list[float], float, bool, list[list[float]]]] = deque(maxlen=replay_size)
        self.transitions_seen = 0

        self.model = QRegressor(hidden_sizes=hidden_sizes).to(self.device)
        self.target_model = QRegressor(hidden_sizes=hidden_sizes).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=alpha)
        self.loss_fn = nn.SmoothL1Loss()
        self.update_steps = 0

    def _batch_q_values(self, state: KQKState, board: chess.Board, moves: list[chess.Move]) -> torch.Tensor:
        encoded = encode_state_action_batch(state, board, moves)
        x = torch.tensor(encoded, dtype=torch.float32, device=self.device)
        return self.model(x)

    def q_value(self, state: KQKState, board: chess.Board, action_uci: str) -> float:
        move = chess.Move.from_uci(action_uci)
        x = torch.tensor([encode_state_action(state, board, move)], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            return float(self.model(x)[0].item())

    def select_action(self, state: KQKState, board: chess.Board, legal_actions: Iterable[str], greedy_only: bool = False) -> str:
        actions = list(legal_actions)
        if not actions:
            raise ValueError("No legal actions available.")

        if (not greedy_only) and torch.rand(1, generator=self.generator).item() < self.epsilon:
            index = int(torch.randint(0, len(actions), (1,), generator=self.generator).item())
            return actions[index]

        moves = [chess.Move.from_uci(uci) for uci in actions]
        with torch.no_grad():
            q_values = self._batch_q_values(state, board, moves)
            best_index = int(torch.argmax(q_values).item())
        return actions[best_index]

    def update(
        self,
        state: KQKState,
        board_before: chess.Board,
        action_uci: str,
        reward: float,
        next_state: KQKState,
        board_after: chess.Board,
        next_legal_actions: Iterable[str],
        done: bool,
    ) -> float:
        self.replay.append(
            self._encode_transition(
                state=state,
                board_before=board_before,
                action_uci=action_uci,
                reward=reward,
                next_state=next_state,
                board_after=board_after,
                next_legal_actions=next_legal_actions,
                done=done,
            )
        )
        self.transitions_seen += 1
        if self.transitions_seen % max(1, self.train_interval) != 0:
            return 0.0
        if len(self.replay) < max(self.batch_size, self.warmup_steps):
            return 0.0

        self.model.train()
        loss_value = 0.0
        for _ in range(max(1, self.updates_per_step)):
            batch = self.rng.sample(self.replay, self.batch_size)
            current_encoded: list[list[float]] = []
            targets: list[float] = []

            for current_vec, b_reward, b_done, next_encoded in batch:
                current_encoded.append(current_vec)

                if b_done:
                    target_value = b_reward
                else:
                    if next_encoded:
                        with torch.no_grad():
                            x_next = torch.tensor(next_encoded, dtype=torch.float32, device=self.device)
                            next_q_values = self.target_model(x_next)
                            max_next_q = float(next_q_values.max().item())
                        target_value = b_reward + self.gamma * max_next_q
                    else:
                        target_value = b_reward
                targets.append(max(-self.q_clip, min(self.q_clip, target_value)))

            x_curr = torch.tensor(current_encoded, dtype=torch.float32, device=self.device)
            y = torch.tensor(targets, dtype=torch.float32, device=self.device)
            pred = self.model(x_curr)
            loss = self.loss_fn(pred, y)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            loss_value = float(loss.item())
            self.update_steps += 1
            if self.update_steps % max(1, self.target_update_interval) == 0:
                self.target_model.load_state_dict(self.model.state_dict())

        return loss_value

    def _encode_transition(
        self,
        state: KQKState,
        board_before: chess.Board,
        action_uci: str,
        reward: float,
        next_state: KQKState,
        board_after: chess.Board,
        next_legal_actions: Iterable[str],
        done: bool,
    ) -> tuple[list[float], float, bool, list[list[float]]]:
        move = chess.Move.from_uci(action_uci)
        current_vec = encode_state_action(state, board_before, move)
        next_actions = list(next_legal_actions)
        if done or not next_actions:
            next_encoded: list[list[float]] = []
        else:
            next_moves = [chess.Move.from_uci(uci) for uci in next_actions]
            next_encoded = encode_state_action_batch(next_state, board_after, next_moves)
        return (current_vec, reward, done, next_encoded)

    def decay_exploration(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def decay_learning_rate(self) -> None:
        lr = self.optimizer.param_groups[0]["lr"]
        next_lr = max(self.alpha_min, lr * self.alpha_decay)
        for group in self.optimizer.param_groups:
            group["lr"] = next_lr
        self.alpha = next_lr

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "alpha": self.alpha,
            "alpha_decay": self.alpha_decay,
            "alpha_min": self.alpha_min,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "epsilon_decay": self.epsilon_decay,
            "epsilon_min": self.epsilon_min,
            "target_update_interval": self.target_update_interval,
            "q_clip": self.q_clip,
            "batch_size": self.batch_size,
            "warmup_steps": self.warmup_steps,
            "updates_per_step": self.updates_per_step,
            "train_interval": self.train_interval,
            "state_dict": self.model.state_dict(),
            "target_state_dict": self.target_model.state_dict(),
        }
        torch.save(payload, target)

    @classmethod
    def load(cls, path: str | Path, seed: int = 0) -> "NeuralQAgent":
        payload = torch.load(Path(path), map_location="cpu")
        agent = cls(
            alpha=float(payload["alpha"]),
            alpha_decay=float(payload.get("alpha_decay", 0.9998)),
            alpha_min=float(payload.get("alpha_min", 1e-4)),
            gamma=float(payload["gamma"]),
            epsilon=float(payload["epsilon"]),
            epsilon_decay=float(payload["epsilon_decay"]),
            epsilon_min=float(payload["epsilon_min"]),
            seed=seed,
            target_update_interval=int(payload.get("target_update_interval", 200)),
            q_clip=float(payload.get("q_clip", 12.0)),
            batch_size=int(payload.get("batch_size", 16)),
            warmup_steps=int(payload.get("warmup_steps", 64)),
            updates_per_step=int(payload.get("updates_per_step", 1)),
            train_interval=int(payload.get("train_interval", 8)),
        )
        agent.model.load_state_dict(payload["state_dict"])
        target_state = payload.get("target_state_dict")
        if target_state:
            agent.target_model.load_state_dict(target_state)
        else:
            agent.target_model.load_state_dict(payload["state_dict"])
        agent.model.eval()
        agent.target_model.eval()
        return agent
