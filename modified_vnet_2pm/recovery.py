# SPDX-License-Identifier: GPL-3.0-only
"""One committed epoch snapshot; history and best files are recoverable views."""

from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch


def atomic_replace(temporary: Path, path: Path) -> None:
    for attempt in range(5):
        try:
            temporary.replace(path)
            return
        except PermissionError as error:
            if os.name != "nt" or getattr(error, "winerror", None) not in (5, 32) or attempt == 4:
                raise
            time.sleep(0.05 * 2**attempt)


def atomic_torch(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    atomic_replace(temporary, path)


def project_epoch_views(output_root: Path, state: dict[str, Any]) -> None:
    # latest_model is the sole commit point; an interrupted view write is replayed.
    atomic_torch(output_root / "checkpoints/best_model.pt", state["best_checkpoint"])
    path = output_root / "training_history.jsonl"
    temporary = path.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in state["history"]:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    atomic_replace(temporary, path)


def validate_snapshot(state: dict[str, Any], identity: dict[str, Any], patch_count: int) -> None:
    if state.get("schema_version") != 2 or state.get("execution_identity") != identity:
        raise ValueError("resume sampling membership/seed/protocol/commit identity mismatch")
    if state.get("training_patch_count") != patch_count:
        raise ValueError("resume dynamic patch count mismatch")
    epoch = state.get("epoch")
    history = state.get("history", [])
    if type(epoch) is not int or not 1 <= epoch <= 199:
        raise ValueError("invalid committed epoch")
    if [row.get("epoch") for row in history] != list(range(1, epoch + 1)):
        raise ValueError("checkpoint history is not contiguous")
    best = state.get("best_checkpoint", {})
    if best.get("execution_identity") != identity or not 1 <= best.get("epoch", 0) <= epoch:
        raise ValueError("best snapshot identity/epoch mismatch")
    if best.get("best_metric") != state.get("best_metric"):
        raise ValueError("best snapshot metric mismatch")
    if not isinstance(state.get("rng_state"), dict) or not {"python", "numpy", "torch"} <= state["rng_state"].keys():
        raise ValueError("complete RNG state is required")


@contextmanager
def output_lock(output_root: Path):
    """Kernel-released lock, including after SIGKILL; never delete another PID's lock."""
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / ".training.lock").open("a+b") as handle:
        if os.name == "nt":
            import msvcrt
            handle.seek(0, 2)
            if handle.tell() == 0:
                handle.write(b"0")
                handle.flush()
            handle.seek(0)
            lock = lambda: msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            unlock = lambda: msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            lock = lambda: fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            unlock = lambda: fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        try:
            lock()
        except OSError as error:
            raise RuntimeError("output directory is owned by another training process") from error
        try:
            yield
        finally:
            handle.seek(0)
            unlock()
