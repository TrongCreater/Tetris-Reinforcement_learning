"""Environment assembly and reward replacement for RL experiments."""
from typing import Any

import gymnasium as gym
import numpy as np

from tetris_gymnasium.envs import Tetris
from tetris_gymnasium.wrappers.grouped import GroupedActionsObservations
from tetris_gymnasium.wrappers.observation import FeatureVectorObservation

from .rewards import BoardFeatures, RewardKind, calculate_reward


GEOMETRY_SCALE = np.array([20.0] * 10 + [20.0, 200.0, 180.0], dtype=np.float32)
BOARD_CELLS = 20 * 10
PROVENANCE_CELLS = BOARD_CELLS
BASE_FEATURE_SIZE = len(GEOMETRY_SCALE) + BOARD_CELLS
LEGACY_FEATURE_SIZE = BASE_FEATURE_SIZE + PROVENANCE_CELLS + 1
QUEUE_SIZE = 4
PIECE_TYPES = 7
# Queue is the visible preview BEFORE taking the action; its first piece is
# the piece that will become active after this placement. Never sample ahead.
QUEUE_FEATURE_SIZE = QUEUE_SIZE * PIECE_TYPES
TRANSITION_FEATURE_SIZE = 5
ENGINEERED_FEATURE_SIZE = 30  # per-column garbage, covered garbage, and total holes
FEATURE_SIZE = LEGACY_FEATURE_SIZE + QUEUE_FEATURE_SIZE + TRANSITION_FEATURE_SIZE + ENGINEERED_FEATURE_SIZE
LEGACY_OBSERVATION_SCHEMA = "garbage_afterstate_v4"
OBSERVATION_SCHEMA = "garbage_afterstate_v5"
FEATURE_SCALE = np.concatenate(
    [GEOMETRY_SCALE, np.ones(PROVENANCE_CELLS + BOARD_CELLS + 1, dtype=np.float32)]
)


def observation_metadata(schema: str = OBSERVATION_SCHEMA) -> dict:
    if schema not in (LEGACY_OBSERVATION_SCHEMA, OBSERVATION_SCHEMA):
        raise ValueError(f"Unsupported observation schema: {schema}")
    return {
        "schema": schema, "feature_size": FEATURE_SIZE,
        "board_shape": [20, 10], "channels": ["occupied", "initial_garbage"],
        "queue_size": QUEUE_SIZE, "piece_types": PIECE_TYPES,
        "transition_features": ["lines_cleared/4", "garbage_cells_removed/40",
                                "delta_total_height/200", "delta_holes/200",
                                "delta_bumpiness/180"],
        "engineered_features": ([f"garbage_cells_column_{i}/20" for i in range(10)]
                                + [f"covered_garbage_column_{i}/20" for i in range(10)]
                                + [f"holes_column_{i}/20" for i in range(10)]),
        "covered_garbage_definition": (
            "initial-garbage provenance cell with any occupied cell above it in the same column"
            if schema == LEGACY_OBSERVATION_SCHEMA else
            "v5: initial-garbage provenance cell with any occupied cell strictly above it in the same column"
        ),
        "queue_timing": "visible preview before placement; first becomes next active piece",
    }


def advance_provenance(mask: np.ndarray, cleared_rows: np.ndarray, width: int, padding: int) -> np.ndarray:
    count = int(np.asarray(cleared_rows, dtype=bool).sum())
    if count == 0:
        return np.asarray(mask, dtype=bool).copy()
    retained = np.asarray(mask, dtype=bool)[~np.asarray(cleared_rows, dtype=bool)]
    empty_top = np.zeros((count, width + 2 * padding), dtype=bool)
    return np.concatenate((empty_top, retained), axis=0)


def normalize_candidates(
    observation: np.ndarray,
    terminal_mask=None,
    provenance_features: np.ndarray | None = None,
) -> np.ndarray:
    features = np.asarray(observation, dtype=np.float32)
    if features.shape[-1] == BASE_FEATURE_SIZE:
        if provenance_features is None:
            provenance = np.zeros(
                (*features.shape[:-1], PROVENANCE_CELLS), dtype=np.float32
            )
        else:
            provenance = np.asarray(provenance_features, dtype=np.float32).reshape(
                *features.shape[:-1], PROVENANCE_CELLS
            )
        features = np.concatenate(
            [
                features,
                provenance,
                np.zeros((*features.shape[:-1], 1), dtype=np.float32),
            ],
            axis=-1,
        )
    if features.shape[-1] != LEGACY_FEATURE_SIZE:
        raise ValueError(f"Expected {LEGACY_FEATURE_SIZE} base candidate features, got {features.shape}")
    if terminal_mask is not None:
        features = features.copy()
        features[..., -1] = np.asarray(terminal_mask, dtype=np.float32)
    return features / FEATURE_SCALE


def covered_garbage_columns(occupied: np.ndarray, garbage: np.ndarray,
                            schema: str = OBSERVATION_SCHEMA) -> np.ndarray:
    """Normalized per-column provenance cells with occupied cells above them."""
    occupied = np.asarray(occupied) > 0
    garbage = np.asarray(garbage) > 0
    if occupied.shape != garbage.shape or occupied.ndim != 3:
        raise ValueError("occupied and garbage must have matching (candidate, row, column) shapes")
    if schema not in (LEGACY_OBSERVATION_SCHEMA, OBSERVATION_SCHEMA):
        raise ValueError(f"Unsupported observation schema: {schema}")
    inclusive = np.maximum.accumulate(occupied, axis=1)
    if schema == LEGACY_OBSERVATION_SCHEMA:
        above = inclusive
    else:
        above = np.zeros_like(inclusive)
        above[:, 1:, :] = inclusive[:, :-1, :]
    return (garbage & above).sum(axis=1).astype(np.float32) / 20.0


class ExperimentRewardWrapper(gym.Wrapper):
    """Replace the game reward with one of the three documented strategies."""

    def __init__(self, env: gym.Env, reward_kind: RewardKind, observation_schema: str = OBSERVATION_SCHEMA):
        super().__init__(env)
        # v3 447-value prefix plus 30 per-column garbage/obstruction features.
        self.observation_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(env.action_space.n, FEATURE_SIZE),
            dtype=np.float32,
        )
        self.reward_kind = RewardKind(reward_kind)
        if observation_schema not in (LEGACY_OBSERVATION_SCHEMA, OBSERVATION_SCHEMA):
            raise ValueError(f"Unsupported observation schema: {observation_schema}")
        self.observation_schema = observation_schema
        self._feature_wrapper = FeatureVectorObservation(env.unwrapped)
        self._before = None
        self.provenance = np.zeros_like(env.unwrapped.board, dtype=bool)
        self.initial_garbage_cells = 0
        self._garbage_potential = 0.0

    def set_provenance(self, interior: np.ndarray) -> None:
        base = self.env.unwrapped
        interior = np.asarray(interior, dtype=bool)
        if interior.shape != (base.height, base.width):
            raise ValueError(
                f"Expected provenance shape {(base.height, base.width)}, got {interior.shape}"
            )
        self.provenance = np.zeros_like(base.board, dtype=bool)
        self.provenance[: base.height, base.padding : base.padding + base.width] = interior
        self.initial_garbage_cells = int(self.provenance.sum())
        self._garbage_potential = self.garbage_row_potential()

    def garbage_row_potential(self) -> float:
        """Potential for filling target gaps without covering them from above.

        It ignores ordinary rows made only from the agent's pieces, so it
        cannot directly reward farming clean lines. It rewards closing gaps in
        rows that still contain original garbage and penalizes target gaps that
        have become occluded by the agent's stack.
        """
        base = self.env.unwrapped
        board = base.crop_padding(base.board) != 0
        provenance = self.provenance[:base.height, base.padding:base.padding + base.width]
        target_rows = provenance.any(axis=1)
        if not np.any(target_rows):
            return 0.0
        target = board[target_rows]
        vacancies = (~target).sum(axis=1).astype(np.float32) / base.width
        gap_debt = float(np.square(vacancies).sum())
        # Rows use top-to-bottom indexing. A gap is covered if any occupied
        # cell exists above it in that column, making a vertical well harder
        # to reach with a placement. Crop engine padding before this test.
        above = np.maximum.accumulate(board, axis=0)
        target_full = np.zeros_like(board, dtype=bool)
        target_full[target_rows] = True
        covered_gaps = int(np.count_nonzero((~board) & above & target_full))
        scale = max(1, self.initial_garbage_cells)
        return -4.0 * gap_debt - 2.0 * covered_gaps / scale

    def target_gap_metrics(self) -> dict[str, int]:
        """Count open cells in provenance rows and those blocked from above."""
        base = self.env.unwrapped
        board = base.crop_padding(base.board) != 0
        provenance = self.provenance[:base.height, base.padding:base.padding + base.width]
        target_rows = provenance.any(axis=1)
        gaps = (~board) & target_rows[:, None]
        above_inclusive = np.maximum.accumulate(board, axis=0)
        above_exclusive = np.zeros_like(above_inclusive)
        above_exclusive[1:] = above_inclusive[:-1]
        return {"target_gap_count": int(gaps.sum()),
                "covered_target_gap_count": int((gaps & above_exclusive).sum())}

    def candidate_provenance_features(self) -> np.ndarray:
        base = self.env.unwrapped
        grouped = self.env
        candidates = []
        for rows in grouped.candidate_rows_cleared:
            projected = advance_provenance(
                self.provenance, rows, base.width, base.padding
            )
            cropped = projected[
                : base.height, base.padding : base.padding + base.width
            ]
            candidates.append(cropped.reshape(-1).astype(np.float32))
        return np.stack(candidates)

    def format_candidates(self, observation: np.ndarray, info: dict) -> np.ndarray:
        features = normalize_candidates(
            observation,
            info.get("terminal_action_mask"),
            self.candidate_provenance_features(),
        )
        base = self.env.unwrapped
        queue = base.queue.get_queue()
        if len(queue) != QUEUE_SIZE or len(base.tetrominoes) != PIECE_TYPES:
            raise ValueError("Observation v3 requires a 4-piece preview and 7 tetromino types")
        preview = np.eye(PIECE_TYPES, dtype=np.float32)[queue].reshape(-1)
        preview = np.broadcast_to(preview, (features.shape[0], QUEUE_FEATURE_SIZE))
        before = self._current_features()
        raw = np.asarray(observation, dtype=np.float32)
        remaining = features[:, BASE_FEATURE_SIZE:BASE_FEATURE_SIZE + PROVENANCE_CELLS].sum(axis=1)
        transitions = np.stack([
            self.env.candidate_rows_cleared.sum(axis=1) / 4.0,
            (self.provenance.sum() - remaining) / 40.0,
            (raw[:, :10].sum(axis=1) - before.total_height) / 200.0,
            (raw[:, 11] - before.holes) / 200.0,
            (raw[:, 12] - before.bumpiness) / 180.0,
        ], axis=-1).astype(np.float32)
        occupied = features[:, 13:BASE_FEATURE_SIZE].reshape(-1, 20, 10)
        garbage = features[:, BASE_FEATURE_SIZE:BASE_FEATURE_SIZE + PROVENANCE_CELLS].reshape(-1, 20, 10)
        garbage_by_column = garbage.sum(axis=1) / 20.0
        covered_by_column = covered_garbage_columns(occupied, garbage, self.observation_schema)
        holes_by_column = np.zeros_like(garbage_by_column)
        for column in range(10):
            occupied_bool = occupied[:, :, column] > 0
            above_inclusive = np.maximum.accumulate(occupied_bool, axis=1)
            empty_below_occupied = above_inclusive
            holes_by_column[:, column] = ((1.0 - occupied[:, :, column]) * empty_below_occupied).sum(axis=1) / 20.0
        engineered = np.concatenate([garbage_by_column, covered_by_column, holes_by_column], axis=1)
        result = np.concatenate([features, preview, transitions, engineered], axis=-1)
        # Invalid candidates do not represent reachable boards.
        result[np.asarray(info["action_mask"]) <= 0] = 0.0
        info["observation_schema"] = self.observation_schema
        return result.astype(np.float32)

    def _current_features(self) -> BoardFeatures:
        observation = self.env.unwrapped._get_obs()
        vector = self._feature_wrapper.observation(observation)
        return BoardFeatures.from_vector(vector, self.env.unwrapped.width)

    def reset(self, **kwargs) -> tuple[np.ndarray, dict[str, Any]]:
        observation, info = self.env.reset(**kwargs)
        self.provenance[:] = False
        self.initial_garbage_cells = 0
        self._garbage_potential = 0.0
        self._before = self._current_features()
        return self.format_candidates(observation, info), info

    def step(self, action: int):
        observation, original_reward, terminated, truncated, info = self.env.step(action)
        remaining_before = int(self.provenance.sum())
        self.provenance = advance_provenance(
            self.provenance,
            info.get("cleared_rows_mask", np.zeros(self.provenance.shape[0], dtype=bool)),
            self.env.unwrapped.width,
            self.env.unwrapped.padding,
        )
        remaining_after = int(self.provenance.sum())
        info["garbage_cells_removed"] = remaining_before - remaining_after
        info["remaining_initial_garbage_cells"] = remaining_after
        info.update(self.target_gap_metrics())
        info["solved_initial_garbage"] = bool(
            self.initial_garbage_cells > 0 and remaining_after == 0
        )
        after = self._current_features()
        potential_after = self.garbage_row_potential()
        reward, components = calculate_reward(
            self.reward_kind,
            int(info.get("lines_cleared", 0)),
            self._before,
            after,
            bool(terminated),
            garbage_removed=remaining_before - remaining_after,
            initial_garbage_cells=self.initial_garbage_cells,
            solved=bool(info["solved_initial_garbage"]),
            potential_before=self._garbage_potential,
            potential_after=potential_after,
        )
        self._before = after
        self._garbage_potential = potential_after
        info["garbage_potential"] = potential_after
        info["original_reward"] = float(original_reward)
        info["reward_components"] = components
        return self.format_candidates(observation, info), reward, terminated, truncated, info


def make_experiment_env(reward_kind: RewardKind, observation_schema: str = OBSERVATION_SCHEMA) -> ExperimentRewardWrapper:
    base = Tetris(gravity=False)
    features = FeatureVectorObservation(base)
    grouped = GroupedActionsObservations(
        base,
        observation_wrappers=[features],
        terminate_on_illegal_action=True,
    )
    return ExperimentRewardWrapper(grouped, reward_kind, observation_schema=observation_schema)

