"""Reward strategies used by the 3 x 3 experiment matrix."""
from dataclasses import dataclass
from enum import Enum

import numpy as np


class RewardKind(str, Enum):
    SPARSE = "sparse"
    SHAPED = "shaped"
    DANGER = "danger"


@dataclass(frozen=True)
class BoardFeatures:
    total_height: float
    max_height: float
    holes: float
    bumpiness: float

    @classmethod
    def from_vector(cls, vector: np.ndarray, width: int = 10) -> "BoardFeatures":
        vector = np.asarray(vector, dtype=np.float32)
        heights = vector[:width]
        return cls(
            total_height=float(heights.sum()),
            max_height=float(vector[width]),
            holes=float(vector[width + 1]),
            bumpiness=float(vector[width + 2]),
        )


def calculate_reward(
    kind: RewardKind,
    lines_cleared: int,
    before: BoardFeatures,
    after: BoardFeatures,
    game_over: bool,
    *,
    garbage_removed: int = 0,
    initial_garbage_cells: int = 0,
    solved: bool = False,
    potential_before: float = 0.0,
    potential_after: float = 0.0,
    gamma: float = 0.99,
) -> tuple[float, dict[str, float]]:
    """Calculate reward for the garbage-clearing objective or normal Tetris.

    The regular Tetris and retained Danger modes preserve their historical
    reward. Sparse/Shaped on a garbage level use only task progress, task
    completion/failure, and (for Shaped) potential feedback about target gaps.
    No survival, ordinary line, height, hole, or bumpiness reward is used for
    the garbage-clearing objective.
    """
    garbage_task = initial_garbage_cells > 0 and kind != RewardKind.DANGER
    if garbage_task:
        progress = 4.0 * float(garbage_removed) / float(initial_garbage_cells)
        success = 10.0 if solved else 0.0
        terminal = -5.0 if game_over and not solved else 0.0
        # Terminal states have zero potential. Otherwise use the standard
        # discounted potential difference with the same gamma as the agent.
        next_potential = 0.0 if game_over or solved else potential_after
        potential = (gamma * next_potential - potential_before
                     if kind == RewardKind.SHAPED else 0.0)
        components = {
            "garbage_progress": progress,
            "success": success,
            "terminal": terminal,
            "potential_shaping": potential,
            "line": 0.0,
            "holes": 0.0,
            "height": 0.0,
            "bumpiness": 0.0,
            "danger": 0.0,
        }
        return float(sum(components.values())), components

    # Preserve behavior for normal endless Tetris and the retained Danger mode.
    line = 2.0 * float(lines_cleared**2)
    terminal = -5.0 if game_over else 0.0
    holes = height = bumpiness = danger = 0.0

    if kind in (RewardKind.SHAPED, RewardKind.DANGER):
        holes = -1.0 * (after.holes - before.holes)
        height = -0.05 * (after.total_height - before.total_height)
        bumpiness = -0.05 * (after.bumpiness - before.bumpiness)

    if kind == RewardKind.DANGER:
        danger = -(
            0.25 * max(0.0, after.max_height - 12.0)
            + 0.75 * max(0.0, after.max_height - 16.0)
        )

    components = {
        "garbage_progress": 0.0,
        "success": 0.0,
        "potential_shaping": 0.0,
        "line": line,
        "holes": holes,
        "height": height,
        "bumpiness": bumpiness,
        "danger": danger,
        "terminal": terminal,
    }
    return float(sum(components.values())), components

