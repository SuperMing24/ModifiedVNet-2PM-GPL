# SPDX-License-Identifier: GPL-3.0-only
"""Protocol loading and immutable scientific guards."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_protocol(path: str | Path) -> dict[str, Any]:
    protocol = json.loads(Path(path).read_text(encoding="utf-8"))
    frozen = {
        "schema_version": 1,
        "model_key": "tian_damseh_modified_vnet",
        "epochs": 199,
        "seed": 42,
        "patch_size": [128, 128, 128],
        "train_stride": [64, 64, 64],
        "batch_size": 4,
        "precision": "fp32",
        "augmentation": "none",
        "training_sampling": "released_code_grid",
        "learning_rate": 0.0001,
        "weight_decay": 0.01,
        "weight_decay_exclude_bias_norm": True,
        "beta1": 0.9,
        "beta2": 0.999,
        "loss_alpha": 5e-9,
        "inference_overlap": 0.0,
        "inference_sw_batch_size": 4,
        "threshold": 0.5,
        "validation_interval_epochs": 1,
        "non_authoritative_validation_patch_loss_logged": False,
        "checkpoint_metric": "harmonic_mean_dice3d_cldice3d",
        "outer_test_access": False,
    }
    for key, expected in frozen.items():
        if protocol.get(key) != expected:
            raise ValueError(f"frozen protocol drift for {key}: {protocol.get(key)!r}")
    if protocol.get("upstream_revision") != "e56562d303aedfc90124d5a25fba96f325562a82":
        raise ValueError("upstream revision drift")
    if protocol.get("license") != "GPL-3.0-only":
        raise ValueError("license drift")
    return protocol


def validate_split_inputs(
    train_root: str | Path,
    validation_root: str | Path,
    assignment_manifest: str | Path,
    expected_assignment_sha256: str,
    fold: int,
) -> None:
    train_path = Path(train_root).resolve()
    validation_path = Path(validation_root).resolve()
    if train_path.name != "train" or validation_path.name != "val":
        raise ValueError("only explicit train and val split roots are accepted")
    if train_path.parent != validation_path.parent:
        raise ValueError("train and val roots must belong to the same fold")
    if train_path.parent.name != f"fold_{fold}":
        raise ValueError(f"fold path mismatch: {train_path.parent}")
    for split_path in (train_path, validation_path):
        for leaf in ("images", "masks"):
            if not (split_path / leaf).is_dir():
                raise FileNotFoundError(f"missing {split_path.name}/{leaf}")
    assignment = json.loads(Path(assignment_manifest).read_text(encoding="utf-8"))
    if assignment.get("assignment_sha256") != expected_assignment_sha256:
        raise ValueError("assignment manifest digest field mismatch")
