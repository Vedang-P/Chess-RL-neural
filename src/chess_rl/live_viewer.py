from __future__ import annotations

import argparse
import math
import queue
import threading
import time
import traceback
from collections import Counter, deque
from pathlib import Path
from tkinter import BOTH, LEFT, RIGHT, TOP, Canvas, StringVar, Tk, ttk
from typing import Any

import chess
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from chess_rl.endgames.kqk import curriculum_phase, random_kqk_curriculum_position
from chess_rl.env import KQKEnv
from chess_rl.neural_agent import NeuralQAgent
from chess_rl.policies import HeuristicDefenderPolicy, RandomDefenderPolicy, SyzygyDefenderPolicy
from chess_rl.syzygy import SyzygyOracle
from chess_rl.syzygy_utils import discover_syzygy_paths
from chess_rl.train import write_training_log


PIECE_UNICODE = {
    "P": "♙",
    "N": "♘",
    "B": "♗",
    "R": "♖",
    "Q": "♕",
    "K": "♔",
    "p": "♟",
    "n": "♞",
    "b": "♝",
    "r": "♜",
    "q": "♛",
    "k": "♚",
}


def add_live_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--episodes", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--defender", choices=["random", "heuristic", "syzygy"], default="heuristic")
    parser.add_argument("--syzygy-path", action="append", default=[])
    parser.add_argument("--disable-auto-syzygy", action="store_true")
    parser.add_argument("--alpha", type=float, default=3e-4)
    parser.add_argument("--alpha-decay", type=float, default=0.9998)
    parser.add_argument("--alpha-min", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.90)
    parser.add_argument("--epsilon", type=float, default=0.30)
    parser.add_argument("--epsilon-decay", type=float, default=0.99985)
    parser.add_argument("--epsilon-min", type=float, default=0.02)
    parser.add_argument("--board-size", type=int, default=720)
    parser.add_argument("--rolling-window", type=int, default=100)
    parser.add_argument("--render-every-step", type=int, default=2)
    parser.add_argument("--update-plots-every-episode", type=int, default=2)
    parser.add_argument("--max-fps", type=float, default=20.0)
    parser.add_argument("--enable-threefold-draw", action="store_true")
    parser.add_argument("--stability-window", type=int, default=1_000)
    parser.add_argument("--freeze-min-episode", type=int, default=5_000)
    parser.add_argument("--freeze-win-rate", type=float, default=0.75)
    parser.add_argument("--freeze-draw-rate-max", type=float, default=0.15)
    parser.add_argument("--freeze-queen-loss-rate-max", type=float, default=0.10)
    parser.add_argument("--early-stop-after-freeze", type=int, default=2_000)
    parser.add_argument("--no-curriculum", action="store_true")
    parser.add_argument("--curriculum-easy-fraction", type=float, default=0.35)
    parser.add_argument("--curriculum-medium-fraction", type=float, default=0.40)
    parser.add_argument("--replay-size", type=int, default=50_000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--warmup-steps", type=int, default=64)
    parser.add_argument("--updates-per-step", type=int, default=1)
    parser.add_argument("--train-interval", type=int, default=8)
    parser.add_argument("--max-queue-events", type=int, default=2_000)
    parser.add_argument("--model-out", default="")
    parser.add_argument("--log-out", default="")
    return parser


def parse_live_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live neural training dashboard for KQ vs K.")
    add_live_args(parser)
    return parser.parse_args()


def _build_defender(name: str, oracle: SyzygyOracle):
    if name == "random":
        return RandomDefenderPolicy()
    if name == "heuristic":
        return HeuristicDefenderPolicy()
    return SyzygyDefenderPolicy(oracle)


class LiveTrainingViewer:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.root = Tk()
        self.root.title("Chess RL Live Trainer (NEURAL | KQ vs K)")
        self.root.minsize(1450, 900)
        self.queue: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=max(128, args.max_queue_events))
        self.stop_event = threading.Event()
        self.training_thread: threading.Thread | None = None
        self.board = chess.Board()
        self.last_move: chess.Move | None = None
        self.start_time = time.time()
        self.last_board_draw_ts = 0.0

        self.outcome_counts = Counter()
        self.outcome_history: list[str] = []
        self.episode_rewards: list[float] = []
        self.episode_losses: list[float] = []
        self.episode_lengths: list[int] = []
        self.episode_epsilons: list[float] = []
        self.episode_check_rates: list[float] = []
        self.episode_optimal_rates: list[float] = []
        self.episode_queen_safety: list[float] = []
        self.episode_avg_q_values: list[float] = []
        self.total_steps = 0
        self.unique_states: set[str] = set()

        self.status_var = StringVar(value="Preparing live dashboard...")
        self.syzygy_var = StringVar(value="Syzygy: loading...")
        self.agent_var = StringVar(value="Agent: neural")
        self.episode_var = StringVar(value=f"Episode: 0/{args.episodes}")
        self.speed_var = StringVar(value="Speed: 0.0 ep/s | 0.0 steps/s")
        self.step_var = StringVar(value="White moves: 0")
        self.action_var = StringVar(value="Last action: -")
        self.reward_var = StringVar(value="Step reward: 0.000")
        self.loss_var = StringVar(value="Loss: -")
        self.epsilon_var = StringVar(value=f"Epsilon: {args.epsilon:.4f}")
        self.summary_var = StringVar(value="W:0 D:0 Q-loss:0 Max:0")
        self.rate_var = StringVar(value="Win rate: 0.0% | Draw rate: 0.0%")
        self.tb_var = StringVar(value="TB-optimal move rate: n/a")

        self.requested_board_size = max(320, int(args.board_size))
        self.board_size = self.requested_board_size
        self.square_size = max(24, self.board_size // 8)
        self.canvas_size = self.requested_board_size + 56
        self.board_origin_x = 28
        self.board_origin_y = 28

        self._build_ui()
        self._draw_board()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(60, self._poll_queue)
        self.root.after(120, self._start_training)

    def _build_ui(self) -> None:
        root_frame = ttk.Frame(self.root, padding=8)
        root_frame.pack(fill=BOTH, expand=True)
        paned = ttk.Panedwindow(root_frame, orient="horizontal")
        paned.pack(fill=BOTH, expand=True)

        left = ttk.Frame(paned, padding=(6, 6, 12, 6))
        right = ttk.Frame(paned, padding=(6, 6, 6, 6))
        paned.add(left, weight=1)
        paned.add(right, weight=1)

        self.canvas = Canvas(
            left,
            width=self.canvas_size,
            height=self.canvas_size,
            bg="#222222",
            highlightthickness=0,
            bd=0,
        )
        self.canvas.pack(side=TOP, fill=BOTH, expand=True)
        self.canvas.bind("<Configure>", self._on_canvas_resize)

        stat_grid = ttk.Frame(right)
        stat_grid.pack(side=TOP, fill=BOTH, expand=False)
        status_font = ("Helvetica", 13, "bold")
        body_font = ("Helvetica", 11)
        mono_font = ("Courier", 11)

        ttk.Label(stat_grid, textvariable=self.status_var, font=status_font, wraplength=760, justify="left").grid(
            row=0, column=0, columnspan=4, sticky="w", padx=(0, 10), pady=(0, 4)
        )
        ttk.Label(stat_grid, textvariable=self.syzygy_var, font=body_font, wraplength=760, justify="left").grid(
            row=1, column=0, columnspan=4, sticky="w", padx=(0, 10), pady=(0, 8)
        )

        labels = [
            ("Agent", self.agent_var),
            ("Episode", self.episode_var),
            ("Speed", self.speed_var),
            ("White Moves", self.step_var),
            ("Action", self.action_var),
            ("Reward", self.reward_var),
            ("Loss", self.loss_var),
            ("Epsilon", self.epsilon_var),
            ("Outcomes", self.summary_var),
            ("Rates", self.rate_var),
            ("TB Metric", self.tb_var),
        ]
        for idx, (label, var) in enumerate(labels, start=2):
            ttk.Label(stat_grid, text=label + ":", font=body_font).grid(
                row=idx, column=0, sticky="w", padx=(0, 6), pady=1
            )
            ttk.Label(stat_grid, textvariable=var, font=mono_font).grid(row=idx, column=1, columnspan=3, sticky="w", pady=1)

        fig = Figure(figsize=(9.5, 7.6), dpi=100, constrained_layout=True)
        axs = fig.subplots(2, 2)
        self.ax_reward = axs[0][0]
        self.ax_outcomes = axs[0][1]
        self.ax_efficiency = axs[1][0]
        self.ax_quality = axs[1][1]
        self.figure_canvas = FigureCanvasTkAgg(fig, master=right)
        self.figure_canvas.draw()
        self.figure_canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=True, pady=(12, 0))

    def _on_close(self) -> None:
        self.stop_event.set()
        self.status_var.set("Stopping training thread...")
        self.root.after(160, self.root.destroy)

    def _enqueue(self, payload: dict[str, Any], allow_drop: bool = False) -> None:
        try:
            self.queue.put_nowait(payload)
        except queue.Full:
            if not allow_drop:
                self.queue.put(payload)

    def _on_canvas_resize(self, event: Any) -> None:
        width = max(1, int(getattr(event, "width", 1)))
        height = max(1, int(getattr(event, "height", 1)))
        min_side = min(width, height)
        if min_side < 120:
            return

        margin = max(8, int(min_side * 0.03))
        usable = max(8 * 12, min_side - 2 * margin)
        square_size = max(12, usable // 8)
        board_size = square_size * 8
        origin_x = (width - board_size) // 2
        origin_y = (height - board_size) // 2

        if (
            square_size == self.square_size
            and board_size == self.board_size
            and origin_x == self.board_origin_x
            and origin_y == self.board_origin_y
        ):
            return

        self.square_size = square_size
        self.board_size = board_size
        self.board_origin_x = origin_x
        self.board_origin_y = origin_y
        self._draw_board()

    def _draw_board(self) -> None:
        self.canvas.delete("all")
        light = "#f0d9b5"
        dark = "#b58863"
        border = "#1f1f1f"
        file_color_light = "#74543a"
        file_color_dark = "#f1e3c4"
        last_from = self.last_move.from_square if self.last_move else None
        last_to = self.last_move.to_square if self.last_move else None
        self.canvas.create_rectangle(
            self.board_origin_x - 6,
            self.board_origin_y - 6,
            self.board_origin_x + self.board_size + 6,
            self.board_origin_y + self.board_size + 6,
            fill=border,
            outline=border,
        )

        piece_font = ("Arial Unicode MS", max(28, int(self.square_size * 0.82)), "bold")
        for rank in range(8):
            for file_idx in range(8):
                square = chess.square(file_idx, 7 - rank)
                x0 = self.board_origin_x + file_idx * self.square_size
                y0 = self.board_origin_y + rank * self.square_size
                x1 = x0 + self.square_size
                y1 = y0 + self.square_size
                square_light = (rank + file_idx) % 2 == 0
                base_color = light if square_light else dark
                if square == last_from:
                    base_color = "#f7ec6e"
                elif square == last_to:
                    base_color = "#9ad66f"
                self.canvas.create_rectangle(x0, y0, x1, y1, fill=base_color, outline="")
                if rank == 7:
                    file_label = chr(ord("a") + file_idx)
                    label_color = file_color_light if square_light else file_color_dark
                    self.canvas.create_text(
                        x1 - 11,
                        y1 - 11,
                        text=file_label,
                        font=("Helvetica", max(9, int(self.square_size * 0.16)), "bold"),
                        fill=label_color,
                    )
                if file_idx == 0:
                    rank_label = str(8 - rank)
                    label_color = file_color_light if square_light else file_color_dark
                    self.canvas.create_text(
                        x0 + 10,
                        y0 + 10,
                        text=rank_label,
                        font=("Helvetica", max(9, int(self.square_size * 0.16)), "bold"),
                        fill=label_color,
                    )
                piece = self.board.piece_at(square)
                if piece is None:
                    continue
                symbol = PIECE_UNICODE.get(piece.symbol(), "?")
                cx = (x0 + x1) / 2
                cy = (y0 + y1) / 2
                fg = "#151515" if piece.color == chess.BLACK else "#f8f8f8"
                shadow = "#f2f2f2" if piece.color == chess.BLACK else "#202020"
                self.canvas.create_text(cx + 1.0, cy + 1.0, text=symbol, font=piece_font, fill=shadow)
                self.canvas.create_text(cx, cy, text=symbol, font=piece_font, fill=fg)

    @staticmethod
    def _rolling_mean(values: list[float], window: int) -> list[float]:
        if not values:
            return []
        out: list[float] = []
        running = 0.0
        for idx, val in enumerate(values):
            running += val
            if idx >= window:
                running -= values[idx - window]
            out.append(running / min(idx + 1, window))
        return out

    def _rolling_outcome_rate(self, label: str, window: int) -> list[float]:
        out: list[float] = []
        values = [1.0 if outcome == label else 0.0 for outcome in self.outcome_history]
        running = 0.0
        for idx, val in enumerate(values):
            running += val
            if idx >= window:
                running -= values[idx - window]
            out.append(running / min(idx + 1, window))
        return out

    def _refresh_plots(self) -> None:
        if not self.episode_rewards:
            return
        window = max(5, self.args.rolling_window)
        x = list(range(1, len(self.episode_rewards) + 1))

        reward_roll = self._rolling_mean(self.episode_rewards, window)
        loss_source = [v for v in self.episode_losses if not math.isnan(v)]
        loss_roll = self._rolling_mean(loss_source, window) if loss_source else []
        length_norm = [m / 75.0 for m in self.episode_lengths]
        length_roll = self._rolling_mean(length_norm, window)
        check_roll = self._rolling_mean(self.episode_check_rates, window)
        safety_roll = self._rolling_mean(self.episode_queen_safety, window)
        epsilon_roll = self._rolling_mean(self.episode_epsilons, window)
        optimal_known = [v for v in self.episode_optimal_rates if not math.isnan(v)]
        optimal_roll = self._rolling_mean(optimal_known, window) if optimal_known else []

        self.ax_reward.clear()
        self.ax_reward.plot(x, reward_roll, color="#2563eb", linewidth=2.2, label=f"reward ({window}-ep mean)")
        if loss_roll:
            loss_x = list(range(1, len(loss_roll) + 1))
            self.ax_reward.plot(loss_x, loss_roll, color="#dc2626", linewidth=1.6, label=f"loss ({window}-ep mean)")
        self.ax_reward.set_title("Learning Signal")
        self.ax_reward.set_xlabel("Episode")
        self.ax_reward.set_ylabel("Value")
        self.ax_reward.grid(alpha=0.2)
        self.ax_reward.legend(loc="best", fontsize=8)

        self.ax_outcomes.clear()
        outcome_specs = [
            ("checkmate", "#22c55e"),
            ("draw", "#f59e0b"),
            ("queen_lost", "#ef4444"),
            ("max_length", "#8b5cf6"),
            ("loss", "#0f172a"),
        ]
        for label, color in outcome_specs:
            series = self._rolling_outcome_rate(label, window)
            self.ax_outcomes.plot(x, series, linewidth=1.8, label=label, color=color)
        self.ax_outcomes.set_title("Outcome Rates (rolling)")
        self.ax_outcomes.set_xlabel("Episode")
        self.ax_outcomes.set_ylabel("Rate")
        self.ax_outcomes.set_ylim(0.0, 1.0)
        self.ax_outcomes.grid(alpha=0.2)
        self.ax_outcomes.legend(loc="upper right", fontsize=8)

        self.ax_efficiency.clear()
        self.ax_efficiency.plot(x, length_roll, color="#0ea5e9", linewidth=2.0, label="length / 75")
        self.ax_efficiency.plot(x, check_roll, color="#14b8a6", linewidth=1.8, label="check rate")
        self.ax_efficiency.plot(x, safety_roll, color="#f97316", linewidth=1.8, label="queen safety")
        self.ax_efficiency.set_title("Efficiency / Safety")
        self.ax_efficiency.set_xlabel("Episode")
        self.ax_efficiency.set_ylabel("Normalized rate")
        self.ax_efficiency.set_ylim(0.0, 1.0)
        self.ax_efficiency.grid(alpha=0.2)
        self.ax_efficiency.legend(loc="best", fontsize=8)

        self.ax_quality.clear()
        self.ax_quality.plot(x, epsilon_roll, color="#7c3aed", linewidth=2.0, label="epsilon")
        if optimal_roll:
            ox = list(range(1, len(optimal_roll) + 1))
            self.ax_quality.plot(ox, optimal_roll, color="#16a34a", linewidth=1.8, label="TB-optimal move rate")
        avg_q_roll = self._rolling_mean(self.episode_avg_q_values, window)
        self.ax_quality.plot(x, avg_q_roll, color="#334155", linewidth=1.4, label="avg selected Q")
        self.ax_quality.set_title("Policy Quality / Exploration")
        self.ax_quality.set_xlabel("Episode")
        self.ax_quality.set_ylabel("Score")
        self.ax_quality.grid(alpha=0.2)
        self.ax_quality.legend(loc="best", fontsize=8)
        self.figure_canvas.draw_idle()

    def _update_speed(self, episode: int) -> None:
        elapsed = max(1e-6, time.time() - self.start_time)
        ep_per_sec = episode / elapsed
        step_per_sec = self.total_steps / elapsed
        self.speed_var.set(f"Speed: {ep_per_sec:.2f} ep/s | {step_per_sec:.1f} steps/s")

    def _poll_queue(self) -> None:
        if self.stop_event.is_set():
            return
        processed = 0
        latest_step: dict[str, Any] | None = None
        while processed < 320:
            try:
                event = self.queue.get_nowait()
            except queue.Empty:
                break
            processed += 1
            if event.get("type") == "step":
                latest_step = event
            else:
                self._handle_event(event)
        if latest_step is not None:
            self._handle_event(latest_step)
        delay = 16 if not self.queue.empty() else 48
        self.root.after(delay, self._poll_queue)

    def _handle_event(self, event: dict[str, Any]) -> None:
        etype = event.get("type")
        if etype == "step":
            now = time.time()
            min_interval = 1.0 / max(1.0, float(self.args.max_fps))
            should_draw = bool(event.get("force_draw")) or (now - self.last_board_draw_ts >= min_interval)
            if should_draw:
                self.board = chess.Board(event["fen"])
                self.last_move = chess.Move.from_uci(event["action_uci"]) if event["action_uci"] else None
                self._draw_board()
                self.last_board_draw_ts = now
            self.step_var.set(f"White moves: {event['white_moves']}")
            self.action_var.set(f"Last action: {event['action_uci']}")
            self.reward_var.set(f"Step reward: {float(event['reward']):+.4f}")
            self.epsilon_var.set(f"Epsilon: {float(event['epsilon']):.4f}")
            if event["loss"] is not None:
                self.loss_var.set(f"Loss: {float(event['loss']):.6f}")
            self.total_steps = int(event["total_steps"])
            return

        if etype == "episode_end":
            episode = int(event["episode"])
            self.episode_rewards.append(float(event["episode_reward"]))
            self.episode_lengths.append(int(event["white_moves"]))
            self.episode_epsilons.append(float(event["epsilon"]))
            self.episode_check_rates.append(float(event["check_rate"]))
            self.episode_queen_safety.append(float(event["queen_safety_rate"]))
            self.episode_avg_q_values.append(float(event["avg_q"]))
            optimal = event.get("optimal_move_rate")
            self.episode_optimal_rates.append(float(optimal) if optimal is not None else float("nan"))
            loss = event.get("episode_loss")
            self.episode_losses.append(float(loss) if loss is not None else float("nan"))
            if loss is not None:
                self.loss_var.set(f"Loss: {float(loss):.6f}")
            outcome = str(event["outcome"])
            self.outcome_history.append(outcome)
            self.outcome_counts.update({outcome: 1})
            wins = self.outcome_counts.get("checkmate", 0)
            draws = self.outcome_counts.get("draw", 0)
            queen_lost = self.outcome_counts.get("queen_lost", 0)
            maxed = self.outcome_counts.get("max_length", 0)
            total = len(self.outcome_history)
            self.summary_var.set(f"W:{wins} D:{draws} Q-loss:{queen_lost} Max:{maxed}")
            self.rate_var.set(
                "Win rate: {win:.1f}% | Draw rate: {draw:.1f}%".format(
                    win=(100.0 * wins / total) if total else 0.0,
                    draw=(100.0 * draws / total) if total else 0.0,
                )
            )
            self.tb_var.set(
                "TB-optimal move rate: {rate}".format(
                    rate=(f"{100.0 * float(optimal):.1f}% (ep)" if optimal is not None else "n/a")
                )
            )
            self.episode_var.set(f"Episode: {episode}/{self.args.episodes}")
            self._update_speed(episode)
            if episode % max(1, self.args.update_plots_every_episode) == 0:
                self._refresh_plots()
            return

        if etype == "status":
            self.status_var.set(str(event["message"]))
            return
        if etype == "syzygy":
            self.syzygy_var.set(str(event["message"]))
            return
        if etype == "done":
            self.status_var.set(str(event["message"]))
            self._refresh_plots()
            return
        if etype == "error":
            self.status_var.set("Training failed. See traceback in terminal.")
            print(event["traceback"])

    def _start_training(self) -> None:
        self.training_thread = threading.Thread(target=self._training_worker, daemon=True)
        self.training_thread.start()

    def _training_worker(self) -> None:
        args = self.args
        model_out = Path(args.model_out) if args.model_out else Path("artifacts/kqk_live_neural_q.pt")
        log_out = Path(args.log_out) if args.log_out else Path("artifacts/kqk_live_neural_training.csv")

        syzygy_paths, table_count = discover_syzygy_paths(
            explicit_paths=args.syzygy_path,
            auto_discover=not args.disable_auto_syzygy,
            cwd=Path.cwd(),
        )
        oracle = SyzygyOracle(syzygy_paths)
        defender = _build_defender(args.defender, oracle)
        if oracle.available:
            syzygy_message = f"Syzygy active: {table_count} table files loaded from {len(syzygy_paths)} directory(ies)."
        elif args.defender == "syzygy":
            syzygy_message = (
                "Syzygy requested but not found; fallback defender=heuristic. "
                "Pass --syzygy-path /path/to/tablebases (containing .rtbw/.rtbz files)."
            )
        else:
            syzygy_message = "Syzygy not loaded (no tablebase directory detected)."
        self._enqueue({"type": "syzygy", "message": syzygy_message})
        self._enqueue({"type": "status", "message": "Training started (neural)."})

        rows: list[dict[str, float | int | str]] = []
        try:
            env = KQKEnv(
                defender_policy=defender,
                seed=args.seed,
                claim_draw_by_repetition=args.enable_threefold_draw,
            )
            agent = NeuralQAgent(
                alpha=args.alpha,
                alpha_decay=args.alpha_decay,
                alpha_min=args.alpha_min,
                gamma=args.gamma,
                epsilon=args.epsilon,
                epsilon_decay=args.epsilon_decay,
                epsilon_min=args.epsilon_min,
                seed=args.seed,
                replay_size=args.replay_size,
                batch_size=args.batch_size,
                warmup_steps=args.warmup_steps,
                updates_per_step=args.updates_per_step,
                train_interval=args.train_interval,
            )

            recent_outcomes: deque[str] = deque(maxlen=max(50, args.stability_window))
            frozen = False
            frozen_episode: int | None = None

            for episode in range(1, args.episodes + 1):
                if self.stop_event.is_set():
                    break
                if args.no_curriculum:
                    state = env.reset()
                else:
                    phase = curriculum_phase(
                        episode=episode,
                        total_episodes=args.episodes,
                        easy_fraction=args.curriculum_easy_fraction,
                        medium_fraction=args.curriculum_medium_fraction,
                    )
                    board = random_kqk_curriculum_position(env.rng, phase=phase, white_to_move=True)
                    state = env.reset(board=board)

                self.unique_states.add(",".join(str(v) for v in state))
                done = False
                episode_reward = 0.0
                episode_loss_total = 0.0
                loss_steps = 0
                outcome = "ongoing"
                white_moves = 0
                step_index = 0
                checks_given = 0
                queen_attacked_steps = 0
                optimal_checks = 0
                optimal_hits = 0
                q_value_sum = 0.0
                q_value_steps = 0

                while not done and not self.stop_event.is_set():
                    legal_actions = env.legal_action_ucis()
                    if not legal_actions:
                        break
                    board_before = env.board.copy(stack=False)
                    action_uci = agent.select_action(state, board_before, legal_actions, greedy_only=False)
                    q_before = agent.q_value(state, board_before, action_uci)
                    q_value_sum += q_before
                    q_value_steps += 1

                    if oracle.available:
                        best_moves = oracle.optimal_moves(board_before)
                        if best_moves is not None:
                            optimal_checks += 1
                            if any(move.uci() == action_uci for move in best_moves):
                                optimal_hits += 1
                    if board_before.gives_check(chess.Move.from_uci(action_uci)):
                        checks_given += 1

                    result = env.step_uci(action_uci)
                    self.unique_states.add(",".join(str(v) for v in result.state))
                    next_legal = env.legal_action_ucis() if not result.done else []
                    board_after = env.board.copy(stack=False)
                    if not frozen:
                        loss = agent.update(
                            state=state,
                            board_before=board_before,
                            action_uci=action_uci,
                            reward=result.reward,
                            next_state=result.state,
                            board_after=board_after,
                            next_legal_actions=next_legal,
                            done=result.done,
                        )
                        episode_loss_total += loss
                        loss_steps += 1
                    else:
                        loss = None

                    queen_bitboard = env.board.pieces(chess.QUEEN, chess.WHITE)
                    if queen_bitboard:
                        qsq = next(iter(queen_bitboard))
                        if env.board.is_attacked_by(chess.BLACK, qsq):
                            queen_attacked_steps += 1

                    step_index += 1
                    self.total_steps += 1
                    state = result.state
                    done = result.done
                    episode_reward += result.reward
                    outcome = str(result.info["outcome"])
                    white_moves = int(result.info["white_moves"])

                    if step_index % max(1, args.render_every_step) == 0 or done:
                        self._enqueue(
                            {
                                "type": "step",
                                "episode": episode,
                                "white_moves": white_moves,
                                "action_uci": action_uci,
                                "reward": float(result.reward),
                                "loss": loss,
                                "epsilon": float(agent.epsilon),
                                "fen": str(result.info["fen"]),
                                "total_steps": self.total_steps,
                                "force_draw": bool(done),
                            },
                            allow_drop=True,
                        )

                recent_outcomes.append(outcome)
                if not frozen:
                    agent.decay_exploration()
                    agent.decay_learning_rate()

                if (
                    (not frozen)
                    and episode >= max(1, args.freeze_min_episode)
                    and len(recent_outcomes) == recent_outcomes.maxlen
                ):
                    counts = Counter(recent_outcomes)
                    window_size = len(recent_outcomes)
                    win_rate = counts.get("checkmate", 0) / window_size
                    draw_rate = counts.get("draw", 0) / window_size
                    queen_loss_rate = counts.get("queen_lost", 0) / window_size
                    if (
                        win_rate >= args.freeze_win_rate
                        and draw_rate <= args.freeze_draw_rate_max
                        and queen_loss_rate <= args.freeze_queen_loss_rate_max
                    ):
                        frozen = True
                        frozen_episode = episode
                        agent.epsilon = agent.epsilon_min
                        self._enqueue(
                            {
                                "type": "status",
                                "message": (
                                    f"Policy frozen at episode {episode} "
                                    f"(win={win_rate:.2f}, draw={draw_rate:.2f}, qloss={queen_loss_rate:.2f})."
                                ),
                            }
                        )

                check_rate = (checks_given / white_moves) if white_moves else 0.0
                queen_safety_rate = 1.0 - ((queen_attacked_steps / white_moves) if white_moves else 0.0)
                optimal_rate = (optimal_hits / optimal_checks) if optimal_checks else None
                avg_q = (q_value_sum / q_value_steps) if q_value_steps else 0.0
                episode_loss = (episode_loss_total / loss_steps) if loss_steps else 0.0

                row: dict[str, float | int | str] = {
                    "episode": episode,
                    "reward": round(episode_reward, 6),
                    "white_moves": white_moves,
                    "outcome": outcome,
                    "epsilon": round(agent.epsilon, 6),
                    "alpha": round(agent.alpha, 6),
                    "frozen": int(frozen),
                    "check_rate": round(check_rate, 6),
                    "queen_safety_rate": round(queen_safety_rate, 6),
                    "avg_q": round(avg_q, 6),
                    "loss": round(episode_loss, 6),
                }
                if optimal_rate is not None:
                    row["tb_optimal_rate"] = round(optimal_rate, 6)
                rows.append(row)

                self._enqueue(
                    {
                        "type": "episode_end",
                        "episode": episode,
                        "episode_reward": episode_reward,
                        "episode_loss": episode_loss,
                        "outcome": outcome,
                        "epsilon": agent.epsilon,
                        "white_moves": white_moves,
                        "check_rate": check_rate,
                        "queen_safety_rate": queen_safety_rate,
                        "optimal_move_rate": optimal_rate,
                        "avg_q": avg_q,
                    }
                )

                if (
                    frozen
                    and frozen_episode is not None
                    and episode - frozen_episode >= max(1, args.early_stop_after_freeze)
                ):
                    break

            model_out.parent.mkdir(parents=True, exist_ok=True)
            log_out.parent.mkdir(parents=True, exist_ok=True)
            write_training_log(log_out, rows)
            agent.save(model_out)

            status = "interrupted" if self.stop_event.is_set() else "completed"
            self._enqueue(
                {
                    "type": "done",
                    "message": (
                        f"Training {status}. model={model_out} log={log_out} "
                        f"states_seen={len(self.unique_states)} total_steps={self.total_steps}"
                    ),
                }
            )
        except Exception:
            self._enqueue({"type": "error", "traceback": traceback.format_exc()})

    def run(self) -> None:
        self.root.mainloop()


def run_live_viewer(args: argparse.Namespace) -> None:
    viewer = LiveTrainingViewer(args)
    viewer.run()
