# SPDX-License-Identifier: GPL-3.0-only

import json
from pathlib import Path

import numpy as np
import pytest

from modified_vnet_2pm.checkpoint_metrics import (
    checkpoint_score,
    cldice_3d,
    dice_3d,
)
from modified_vnet_2pm.protocol import load_protocol, validate_split_inputs


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs" / "frozen_protocol.json"


def test_frozen_protocol_loads_and_rejects_drift(tmp_path):
    protocol = load_protocol(PROTOCOL)
    assert protocol["epochs"] == 199
    assert protocol["precision"] == "fp32"
    assert protocol["training_sampling"] == "released_code_grid"
    assert protocol["non_authoritative_validation_patch_loss_logged"] is False
    assert protocol["outer_test_access"] is False
    drifted = dict(protocol, threshold=0.4)
    path = tmp_path / "drifted.json"
    path.write_text(json.dumps(drifted), encoding="utf-8")
    with pytest.raises(ValueError, match="threshold"):
        load_protocol(path)


def test_checkpoint_metrics_match_frozen_formulas():
    target = np.zeros((4, 4, 4), dtype=np.float32)
    target[1:3, 1:3, 1:3] = 1
    probability = target.copy()
    assert dice_3d(probability, target) == 1.0
    assert cldice_3d(probability, target) > 0.999999
    score = checkpoint_score(
        [
            {"dice_3d": 0.5, "cl_dice_3d": 1.0},
            {"dice_3d": 1.0, "cl_dice_3d": 0.5},
        ]
    )
    assert score == {
        "dice_3d": 0.75,
        "cl_dice_3d": 0.75,
        "checkpoint_score": 0.75,
    }


def test_split_guard_accepts_only_explicit_train_and_val_roots(tmp_path):
    fold_root = tmp_path / "fold_1"
    for split in ("train", "val"):
        for leaf in ("images", "masks"):
            (fold_root / split / leaf).mkdir(parents=True)
    assignment = tmp_path / "split_manifest.json"
    assignment.write_text(
        json.dumps({"assignment_sha256": "expected"}),
        encoding="utf-8",
    )
    validate_split_inputs(
        fold_root / "train",
        fold_root / "val",
        assignment,
        "expected",
        1,
    )
    with pytest.raises(ValueError, match="only explicit train and val"):
        validate_split_inputs(
            fold_root,
            fold_root / "val",
            assignment,
            "expected",
            1,
        )


def test_all_python_files_are_gpl_marked_and_host_independent():
    for path in sorted((ROOT / "modified_vnet_2pm").glob("*.py")):
        value = path.read_text(encoding="utf-8")
        assert value.startswith("# SPDX-License-Identifier: GPL-3.0-only"), path
        assert "from src." not in value
        assert "import src." not in value
    assert "GNU GENERAL PUBLIC LICENSE" in (ROOT / "LICENSE").read_text(
        encoding="utf-8"
    )
