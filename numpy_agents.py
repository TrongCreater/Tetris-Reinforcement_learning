"""Small CPU-only inference adapters for the two frozen demo policies."""
from __future__ import annotations

from pathlib import Path

import numpy as np


class SurvivalMLPPolicy:
    """Legacy DDQN + Danger policy using the original 13 geometry features."""

    def __init__(self, path: Path):
        with np.load(path, allow_pickle=False) as data:
            self.weights = {key: data[key].copy() for key in data.files}
        if self.weights["network.0.weight"].shape != (64, 13):
            raise ValueError("Survival checkpoint is not the expected 13-feature DDQN + Danger model")

    def act(self, candidates: np.ndarray, mask: np.ndarray) -> int:
        x = np.asarray(candidates, dtype=np.float32)[:, :13]
        for layer in (0, 2, 4):
            x = x @ self.weights[f"network.{layer}.weight"].T + self.weights[f"network.{layer}.bias"]
            if layer != 4:
                np.maximum(x, 0, out=x)
        return _masked_argmax(x.reshape(-1), mask)


class ChallengeCNNPolicy:
    """NumPy forward pass for the exported 477-feature DDQN + Sparse CNN."""

    def __init__(self, path: Path):
        with np.load(path, allow_pickle=False) as data:
            self.weights = {key: data[key].copy() for key in data.files}
        expected = ("encoder.board.0.weight", "encoder.board.2.weight",
                    "encoder.board.5.weight", "encoder.context.0.weight",
                    "encoder.fusion.0.weight", "head.weight")
        if any(key not in self.weights for key in expected):
            raise ValueError("Challenge checkpoint is missing CNN layers")
        if self.weights["encoder.board.0.weight"].shape != (16, 2, 3, 3):
            raise ValueError("Challenge CNN input layer does not match the packaged model")

    @staticmethod
    def _conv2d_same(x: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
        padded = np.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)))
        windows = np.lib.stride_tricks.sliding_window_view(padded, (3, 3), axis=(2, 3))
        result = np.einsum("nchwij,ocij->nohw", windows, weight, optimize=True)
        result += bias[None, :, None, None]
        return result

    def values(self, candidates: np.ndarray) -> np.ndarray:
        x = np.asarray(candidates, dtype=np.float32)
        if x.ndim != 2 or x.shape[1] != 477:
            raise ValueError(f"Challenge policy expects (N, 477), got {x.shape}")

        board = np.stack((x[:, 13:213].reshape(-1, 20, 10),
                          x[:, 213:413].reshape(-1, 20, 10)), axis=1)
        board = np.maximum(self._conv2d_same(
            board, self.weights["encoder.board.0.weight"], self.weights["encoder.board.0.bias"]), 0)
        board = np.maximum(self._conv2d_same(
            board, self.weights["encoder.board.2.weight"], self.weights["encoder.board.2.bias"]), 0)
        board = board.reshape(board.shape[0], -1)
        board = np.maximum(board @ self.weights["encoder.board.5.weight"].T
                           + self.weights["encoder.board.5.bias"], 0)

        context_input = np.concatenate((x[:, :13], x[:, 413:]), axis=1)
        context = np.maximum(context_input @ self.weights["encoder.context.0.weight"].T
                             + self.weights["encoder.context.0.bias"], 0)
        fused = np.concatenate((board, context), axis=1)
        fused = np.maximum(fused @ self.weights["encoder.fusion.0.weight"].T
                           + self.weights["encoder.fusion.0.bias"], 0)
        return (fused @ self.weights["head.weight"].T + self.weights["head.bias"]).reshape(-1)

    def act(self, candidates: np.ndarray, mask: np.ndarray) -> int:
        return _masked_argmax(self.values(candidates), mask)


def _masked_argmax(values: np.ndarray, mask: np.ndarray) -> int:
    values = np.asarray(values, dtype=np.float32).copy()
    legal = np.asarray(mask) > 0
    if values.shape != legal.shape or not np.any(legal):
        raise ValueError("Policy received no legal placements or an invalid action mask")
    values[~legal] = -np.inf
    return int(np.argmax(values))
