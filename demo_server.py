"""Offline, inference-only Tetris demo server. No training or checkpoint writes."""
from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import numpy as np

from numpy_agents import ChallengeCNNPolicy, SurvivalMLPPolicy
from rl_experiments.environment import make_experiment_env
from rl_experiments.rewards import RewardKind
from tetris_gymnasium.envs import Tetris


ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
WIDTH, HEIGHT = 10, 20
AGENT_BASE_DELAY = 0.48
LEVELS = (
    {"number": 1, "name": "First Well", "layout": "straight_well", "rows": 4, "label": "Straight well · 4 hàng"},
    {"number": 2, "name": "Deeper Well", "layout": "straight_well", "rows": 6, "label": "Straight well · 6 hàng"},
    {"number": 3, "name": "Find the Well", "layout": "random_well", "rows": 5, "label": "Random straight well · 5 hàng"},
    {"number": 4, "name": "Shift Once", "layout": "single_shift_well", "rows": 6, "label": "Single-shift well · 6 hàng"},
    {"number": 5, "name": "Long Shift", "layout": "single_shift_well", "rows": 8, "label": "Single-shift well · 8 hàng"},
    {"number": 6, "name": "Two Segments", "layout": "segmented_well", "rows": 6, "label": "Segmented well · 6 hàng"},
    {"number": 7, "name": "Three Segments", "layout": "segmented_well", "rows": 8, "label": "Segmented well · 8 hàng"},
    {"number": 8, "name": "Final Descent", "layout": "segmented_well", "rows": 10, "label": "Segmented well · 10 hàng"},
)
COLOR_BY_ID = {
    2: "#50d6e7", 3: "#ffc857", 4: "#a98cff", 5: "#64dfa5",
    6: "#ff7181", 7: "#6c9cff", 8: "#ff9b64",
}


def generate_level(seed: int, level: dict) -> np.ndarray:
    """Generate a board from the same four well geometries as the trained model."""
    rng = np.random.default_rng(seed)
    rows = int(level["rows"])
    layout = level["layout"]
    if layout == "straight_well":
        holes = [[WIDTH // 2] for _ in range(rows)]
    elif layout == "random_well":
        column = int(rng.integers(0, WIDTH))
        holes = [[column] for _ in range(rows)]
    elif layout == "single_shift_well":
        boundary = int(rng.integers(2, rows - 1))
        first = int(rng.integers(1, WIDTH - 1))
        second = first + int(rng.choice((-1, 1)))
        holes = [[first] if row < boundary else [second] for row in range(rows)]
    elif layout == "segmented_well":
        count = int(rng.integers(2, min(4, rows // 2) + 1))
        lengths = [2] * count
        for _ in range(rows - 2 * count):
            lengths[int(rng.integers(0, count))] += 1
        column = int(rng.integers(1, WIDTH - 1))
        holes = []
        for segment, length in enumerate(lengths):
            if segment:
                step = int(rng.choice([-1, 1]))
                if not 0 <= column + step < WIDTH:
                    step *= -1
                column += step
            holes.extend([[column] for _ in range(length)])
    else:
        raise ValueError(f"Unknown challenge layout: {layout}")

    board = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    top = HEIGHT - rows
    for offset, gaps in enumerate(holes):
        board[top + offset] = rng.integers(2, 9, size=WIDTH, dtype=np.uint8)
        board[top + offset, gaps] = 0
    return board


def install_agent_level(env, board: np.ndarray):
    base = env.unwrapped
    base.board = base.create_board()
    base.board[:HEIGHT, base.padding:base.padding + WIDTH] = board
    if base.collision(base.active_tetromino, base.x, base.y):
        raise ValueError("Challenge board overlaps the starting tetromino")
    env.set_provenance(board != 0)
    grouped = env.env
    raw_candidates = grouped.observation(base._get_obs())
    info = {"action_mask": grouped.legal_actions_mask.copy(),
            "terminal_action_mask": grouped.terminal_actions_mask.copy()}
    env._before = env._current_features()
    return env.format_candidates(raw_candidates, info), info


class GameSession:
    def __init__(self):
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.thread = None
        self.base = None
        self.env = None
        self.policy = None
        self.observation = None
        self.mask = None
        self.mode = "survival"
        self.controller = "human"
        self.level_index = 0
        self.seed = 0
        self.agent_delay = 0.36
        self.agent_speed = 1
        self.state = "menu"
        self.message = "Chọn chế độ để bắt đầu"
        self.lines = 0
        self.placements = 0
        self.score = 0
        self.remaining = 0
        self.initial_cells = 0
        self.provenance = np.zeros((HEIGHT, WIDTH), dtype=bool)
        self.last_tick = 0.0
        self.started_at = 0.0
        self.error = None
        self.serial = 0

    def start(self, mode="survival", controller="human", level=0, seed=None, delay=0.36, agent_speed=1):
        agent_speed = self._validated_agent_speed(agent_speed)
        with self.lock:
            self._close_game()
            self.stop_event = threading.Event()
            self.mode = mode if mode in ("survival", "challenge") else "survival"
            self.controller = controller if controller in ("human", "agent") else "human"
            self.level_index = max(0, min(len(LEVELS) - 1, int(level)))
            self.seed = int(seed if seed is not None else (int(time.time()) % 2_000_000_000))
            self.agent_speed = agent_speed
            self.agent_delay = (AGENT_BASE_DELAY / agent_speed if self.controller == "agent"
                                else float(np.clip(delay, 0.08, 1.2)))
            self.lines = self.placements = self.score = 0
            self.error = None
            self.provenance = np.zeros((HEIGHT, WIDTH), dtype=bool)
            self.remaining = self.initial_cells = 0
            self.started_at = time.time()
            self.last_tick = time.monotonic() - (self.agent_delay if self.controller == "agent" else 0)

            if self.controller == "human":
                self.base = Tetris(gravity=False)
                self.base.reset(seed=self.seed)
                if self.mode == "challenge":
                    board = generate_level(self.seed, LEVELS[self.level_index])
                    self._install_human_level(board)
                self.state = "playing"
                self.message = "Bạn đang chơi"
            else:
                reward = RewardKind.DANGER if self.mode == "survival" else RewardKind.SPARSE
                self.env = make_experiment_env(reward)
                self.observation, info = self.env.reset(seed=self.seed)
                self.mask = np.asarray(info["action_mask"], dtype=np.float32)
                if self.mode == "survival":
                    self.policy = SurvivalMLPPolicy(MODEL_DIR / "survival_ddqn_danger.npz")
                else:
                    self.policy = ChallengeCNNPolicy(MODEL_DIR / "challenge_ddqn_sparse.npz")
                    board = generate_level(self.seed, LEVELS[self.level_index])
                    self.observation, info = install_agent_level(self.env, board)
                    self.mask = np.asarray(info["action_mask"], dtype=np.float32)
                    self.initial_cells = int(self.env.initial_garbage_cells)
                    self.remaining = self.initial_cells
                    self.provenance = board != 0
                self.state = "playing"
                self.message = "DDQN đang chơi · inference đóng băng"

            self.serial += 1
            self.thread = threading.Thread(target=self._loop, args=(self.stop_event,),
                                           name="tetris-demo-game", daemon=True)
            self.thread.start()
        return self.snapshot()

    def _install_human_level(self, board: np.ndarray):
        self.base.board = self.base.create_board()
        self.base.board[:HEIGHT, self.base.padding:self.base.padding + WIDTH] = board
        self.provenance = board != 0
        self.initial_cells = int(self.provenance.sum())
        self.remaining = self.initial_cells

    def _close_game(self):
        self.stop_event.set()
        if self.env is not None:
            try:
                self.env.close()
            except Exception:
                pass
        self.env = self.base = self.policy = None
        self.observation = self.mask = None

    def _loop(self, stop_event):
        while not stop_event.wait(0.01):
            with self.lock:
                if self.state != "playing" or self.stop_event is not stop_event:
                    continue
                if self.controller == "agent":
                    # Read the current delay on every tick, so changing x1 -> x5
                    # does not wait for an old, slower sleep to finish.
                    if time.monotonic() - self.last_tick < self.agent_delay:
                        continue
                    # Count inference inside the target interval. On a slow CPU,
                    # the next placement runs as soon as inference permits.
                    self.last_tick = time.monotonic()
                    try:
                        self._agent_step()
                    except Exception as exc:
                        self.error = str(exc)
                        self.state = "error"
                        self.message = "Không thể chạy agent"
                        stop_event.set()
                        break
                else:
                    delay = self._human_tick_interval()
                    if time.monotonic() - self.last_tick >= delay:
                        self._gravity_tick()
                        self.last_tick = time.monotonic()

    def _human_tick_interval(self):
        return float(np.clip(self.agent_delay, 0.08, 1.2))

    @staticmethod
    def _validated_agent_speed(value):
        speed = float(value)
        if not np.isfinite(speed) or not 1 <= speed <= 5 or speed != int(speed):
            raise ValueError("Agent speed must be an integer from 1 to 5")
        return int(speed)

    def set_speed(self, delay=None, agent_speed=None):
        with self.lock:
            if self.controller == "agent":
                self.agent_speed = self._validated_agent_speed(agent_speed)
                self.agent_delay = AGENT_BASE_DELAY / self.agent_speed
            elif delay is not None:
                self.agent_delay = float(np.clip(float(delay), 0.08, 1.2))
                self.last_tick = time.monotonic()
            return self.snapshot()

    def action(self, name: str):
        with self.lock:
            if self.state != "playing" or self.controller != "human" or self.base is None:
                return self.snapshot()
            mapping = {"left": self.base.actions.move_left, "right": self.base.actions.move_right,
                       "down": self.base.actions.move_down, "rotate": self.base.actions.rotate_clockwise,
                       "rotate_ccw": self.base.actions.rotate_counterclockwise,
                       "hold": self.base.actions.swap, "drop": self.base.actions.hard_drop}
            if name == "pause":
                self.state = "paused"
                self.message = "Đang tạm dừng"
                return self.snapshot()
            if name not in mapping:
                raise ValueError(f"Unknown game action: {name!r}")
            if name == "drop":
                self._human_lock_piece()
            else:
                _, _, terminated, _, _ = self.base.step(mapping[name])
                if terminated:
                    self._finish("game_over", "Ván chơi kết thúc")
            self.last_tick = time.monotonic()
            return self.snapshot()

    def toggle_pause(self):
        with self.lock:
            if self.state == "playing":
                self.state, self.message = "paused", "Đang tạm dừng"
            elif self.state == "paused":
                self.state, self.message = "playing", "Tiếp tục"
                self.last_tick = time.monotonic()
            return self.snapshot()

    def stop(self):
        with self.lock:
            if self.state in ("playing", "paused"):
                self._finish("stopped", "Đã dừng ván chơi")
            return self.snapshot()

    def _gravity_tick(self):
        if self.base.collision(self.base.active_tetromino, self.base.x, self.base.y + 1):
            self._human_lock_piece()
        else:
            self.base.step(self.base.actions.move_down)

    def _human_lock_piece(self):
        landing = self.base.landing_y(self.base.active_tetromino, self.base.x, self.base.y)
        projected = self.base.project_tetromino(self.base.active_tetromino, self.base.x, landing)
        inner = self.base.crop_padding(projected)[:HEIGHT]
        cleared_rows = np.all(inner != 0, axis=1)
        n_cleared = int(cleared_rows.sum())
        previous_lines = self.lines
        self.base.step(self.base.actions.hard_drop)
        self.placements += 1
        self.lines += n_cleared
        self.score += (n_cleared ** 2) * WIDTH
        if n_cleared:
            self.provenance = np.concatenate((np.zeros((n_cleared, WIDTH), dtype=bool),
                                              self.provenance[~cleared_rows]), axis=0)
            self.remaining = int(self.provenance.sum())
        if self.mode == "challenge" and self.remaining == 0:
            self._finish("level_clear", "Màn đã hoàn thành")
        elif self.base.game_over:
            self._finish("game_over", "Ván chơi kết thúc")
        self.last_tick = time.monotonic()

    def _agent_step(self):
        action = self.policy.act(self.observation, self.mask)
        self.observation, _, terminated, truncated, info = self.env.step(action)
        self.mask = np.asarray(info["action_mask"], dtype=np.float32)
        self.placements += 1
        self.lines += int(info.get("lines_cleared", 0))
        self.score += int(info.get("lines_cleared", 0)) ** 2 * WIDTH
        if self.mode == "challenge":
            self.remaining = int(info.get("remaining_initial_garbage_cells", self.remaining))
            self.provenance = self.env.provenance[:HEIGHT, self.env.unwrapped.padding:self.env.unwrapped.padding + WIDTH].copy()
            if info.get("solved_initial_garbage") or self.remaining == 0:
                self._finish("level_clear", "DDQN đã hoàn thành màn")
                return
        if terminated or truncated:
            self._finish("game_over", "Ván chơi kết thúc")

    def _finish(self, state: str, message: str):
        self.state, self.message = state, message

    @staticmethod
    def _piece_payload(piece):
        if piece is None:
            return None
        return {"shape": np.asarray(piece.matrix, dtype=np.uint8).tolist(),
                "id": int(piece.id), "color": "rgb(" + ",".join(map(str, piece.color_rgb)) + ")"}

    def _board_payload(self):
        if self.controller == "agent" and self.env is not None:
            base = self.env.unwrapped
            provenance = self.env.provenance[:HEIGHT, base.padding:base.padding + WIDTH]
        elif self.base is not None:
            base = self.base
            provenance = self.provenance
        else:
            return ([[0] * WIDTH for _ in range(HEIGHT)], [[False] * WIDTH for _ in range(HEIGHT)],
                    None, 0, None, [], None)

        board = base.crop_padding(base.board)[:HEIGHT, :WIDTH]
        active = self._piece_payload(base.active_tetromino)
        ghost_y = int(base.landing_y(base.active_tetromino, base.x, base.y)) if active and not base.game_over else 0
        active_x = int(base.x - base.padding)
        active_y = int(base.y)
        queue = [self._piece_payload(base.tetrominoes[index]) for index in base.queue.get_queue()]
        held_pieces = base.holder.get_tetrominoes()
        hold = self._piece_payload(held_pieces[0]) if held_pieces else None
        return board.tolist(), np.asarray(provenance, dtype=bool).tolist(), active, active_x, (active_y, ghost_y), queue, hold

    def snapshot(self):
        with self.lock:
            values = self._board_payload()
            board, provenance, active, active_x, y_values, queue, hold = values
            if y_values is None:
                active_y, ghost_y = 0, 0
            else:
                active_y, ghost_y = y_values
            is_challenge = self.mode == "challenge"
            level = LEVELS[self.level_index] if is_challenge else None
            speed_level = min(12, self.lines // 10) + 1
            return {
                "state": self.state, "message": self.message, "error": self.error,
                "mode": self.mode, "controller": self.controller,
                "agent": "DDQN + Sparse" if is_challenge else "DDQN + Danger",
                "level": level, "levelIndex": self.level_index, "levelCount": len(LEVELS),
                "board": board, "provenance": provenance, "active": active,
                "activeX": active_x, "activeY": active_y, "ghostY": ghost_y,
                "queue": queue, "hold": hold,
                "lines": self.lines, "placements": self.placements, "score": self.score,
                "initialCells": self.initial_cells, "remaining": self.remaining,
                "removedPct": (100.0 * (self.initial_cells - self.remaining) / self.initial_cells
                               if self.initial_cells else 0.0),
                "survivalLevel": speed_level, "seed": self.seed,
                "delay": self.agent_delay,
                "agentSpeed": self.agent_speed,
            }

    def next_level(self):
        with self.lock:
            next_index = (self.level_index + 1) % len(LEVELS)
            mode, controller, seed, delay = self.mode, self.controller, self.seed + 1, self.agent_delay
            agent_speed = self.agent_speed
        return self.start(mode, controller, next_index, seed, delay, agent_speed)


SESSION = GameSession()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        return

    def _json(self, value, status=200):
        body = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self):
        if self.headers.get("Transfer-Encoding"):
            raise ValueError("Send JSON with Content-Length; chunked requests are not supported")
        length = int(self.headers.get("Content-Length", "0"))
        if not 0 < length <= 4096:
            raise ValueError("A JSON object of 1 to 4096 bytes is required")
        body = json.loads(self.rfile.read(length).decode("utf-8"))
        if not isinstance(body, dict):
            raise ValueError("Request must be a JSON object")
        return body

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/api/state":
            return self._json(SESSION.snapshot())
        if path == "/api/levels":
            return self._json(list(LEVELS))
        self.send_error(404)

    def do_POST(self):
        path = urlparse(self.path).path
        try:
            body = self._body()
            if path == "/api/start":
                if body.get("mode") not in ("survival", "challenge") or body.get("controller") not in ("human", "agent"):
                    raise ValueError("Choose a valid mode and controller before starting")
                return self._json(SESSION.start(body.get("mode"), body.get("controller"),
                                                body.get("level", 0), body.get("seed"), body.get("delay", 0.36),
                                                body.get("agentSpeed", 1)))
            if path == "/api/action":
                return self._json(SESSION.action(str(body.get("action", ""))))
            if path == "/api/pause":
                return self._json(SESSION.toggle_pause())
            if path == "/api/stop":
                return self._json(SESSION.stop())
            if path == "/api/next":
                return self._json(SESSION.next_level())
            if path == "/api/speed":
                return self._json(SESSION.set_speed(body.get("delay"), body.get("agentSpeed")))
        except Exception as exc:
            return self._json({"error": str(exc)}, status=400)
        self.send_error(404)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8770)
    args = parser.parse_args()
    try:
        server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    except OSError:
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    url = f"http://127.0.0.1:{server.server_port}/"
    print(f"DEMO ready: {url}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        SESSION.stop()
        server.server_close()


if __name__ == "__main__":
    main()
