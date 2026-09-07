# SPDX-License-Identifier: GPL-3.0-only

from __future__ import annotations

import json
from pathlib import Path

import torch

from modified_vnet_2pm.model import TianDamsehVNet
from modified_vnet_2pm.predict_cli import (
    _authorization_guard,
    _image_paths,
    _load_model,
)


ASSIGNMENT_SHA256 = "a" * 64


def _authorization_manifest(path: Path) -> Path:
    payload = {
        "status": "registered_ready_to_submit",
        "authority_boundary": {
            "outer_test_access_authorized": True,
            "new_gpu_budget_authorized": True,
            "authorized_gpu_hours_total": 10.0,
            "automatic_retry": False,
        },
        "global_registration": {
            "status": "registered_by_research_mainline",
            "fold_run_ids": {
                "tian_damseh_vnet": [
                    "RUN-1000",
                    "RUN-1001",
                    "RUN-1002",
                    "RUN-1003",
                    "RUN-1004",
                ]
            },
        },
        "reference_protocol": {
            "assignment_sha256": ASSIGNMENT_SHA256,
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_authorization_guard_requires_registered_fold_run(tmp_path):
    manifest = _authorization_manifest(tmp_path / "authorization.json")
    assignment = tmp_path / "split_manifest.json"
    assignment.write_text(
        json.dumps({"assignment_sha256": ASSIGNMENT_SHA256}),
        encoding="utf-8",
    )

    loaded = _authorization_guard(
        manifest_path=manifest,
        assignment_manifest=assignment,
        expected_assignment_sha256=ASSIGNMENT_SHA256,
        fold=3,
        run_id="RUN-1003",
    )

    assert loaded["authority_boundary"]["automatic_retry"] is False


def test_image_guard_accepts_only_fourteen_images_in_requested_test_fold(tmp_path):
    images = tmp_path / "fold_2" / "test" / "images"
    images.mkdir(parents=True)
    for index in range(14):
        (images / f"mv{index:02d}.nii").touch()

    paths = _image_paths(images, 2)

    assert len(paths) == 14
    assert all(path.parent == images for path in paths)


def test_plain_interchange_checkpoint_loads_strictly(tmp_path):
    model = TianDamsehVNet(in_channels=1, out_channels=1, base_channels=4)
    checkpoint = tmp_path / "interchange.pt"
    torch.save(
        {
            "schema_version": 1,
            "epoch": 61,
            "model_state_dict": model.state_dict(),
        },
        checkpoint,
    )

    loaded, epoch = _load_model(checkpoint, torch.device("cpu"))

    assert epoch == 61
    assert set(loaded.state_dict()) == set(model.state_dict())
