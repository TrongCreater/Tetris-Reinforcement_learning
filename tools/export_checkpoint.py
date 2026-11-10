"""Export a DDQN + Sparse CNN checkpoint to inference-only NumPy arrays.

Run with a Python environment that has PyTorch installed:
    python tools/export_checkpoint.py model.pt model_inference.npz
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if payload.get("agent") != "ddqn" or payload.get("architecture") != "cnn":
        raise ValueError("Expected a DDQN CNN checkpoint")
    if payload.get("feature_size") != 477 or payload.get("observation", {}).get("schema") != "garbage_afterstate_v5":
        raise ValueError("Unsupported observation schema; this demo expects garbage_afterstate_v5")
    state = payload["model"]
    arrays = {name: tensor.detach().cpu().numpy().astype(np.float32, copy=False)
              for name, tensor in state.items()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    metadata = {
        "agent": "ddqn",
        "reward": "sparse",
        "architecture": "CandidateCNN",
        "feature_size": 477,
        "observation_schema": "garbage_afterstate_v5",
        "source_checkpoint": str(args.checkpoint.resolve()),
        "source_checkpoint_sha256": sha256(args.checkpoint),
        "inference_file": args.output.name,
        "inference_file_sha256": sha256(args.output),
        "training": "disabled",
    }
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
