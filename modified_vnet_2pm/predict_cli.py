# SPDX-License-Identifier: GPL-3.0-only
"""Inference-only CLI for frozen Modified V-Net outer-test checkpoints."""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .data import load_nii
from .model import TianDamsehVNet
from .protocol import load_protocol, sha256


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_numpy(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, value, allow_pickle=False)
    temporary.replace(path)


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), *args],
        text=True,
    ).strip()


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _authorization_guard(
    *,
    manifest_path: Path,
    assignment_manifest: Path,
    expected_assignment_sha256: str,
    fold: int,
    run_id: str,
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    authority = manifest["authority_boundary"]
    registration = manifest["global_registration"]
    if manifest["status"] not in {"registered_ready_to_submit", "submitted"}:
        raise ValueError("host authorization manifest is not executable")
    if authority["outer_test_access_authorized"] is not True:
        raise ValueError("outer-test access is not authorized")
    if authority["new_gpu_budget_authorized"] is not True:
        raise ValueError("outer-test GPU budget is not authorized")
    if authority["authorized_gpu_hours_total"] != 10.0:
        raise ValueError("outer-test GPU budget drift")
    if authority["automatic_retry"] is not False:
        raise ValueError("automatic retry must remain disabled")
    if registration["status"] != "registered_by_research_mainline":
        raise ValueError("mainline global IDs are not registered")
    run_ids = registration["fold_run_ids"]["tian_damseh_vnet"]
    if len(run_ids) != 5 or run_ids[fold] != run_id:
        raise ValueError("run ID does not match the registered fold")
    assignment = json.loads(assignment_manifest.read_text(encoding="utf-8"))
    if assignment.get("assignment_sha256") != expected_assignment_sha256:
        raise ValueError("assignment manifest digest field mismatch")
    if (
        manifest["reference_protocol"]["assignment_sha256"]
        != expected_assignment_sha256
    ):
        raise ValueError("host protocol assignment digest mismatch")
    return manifest


def _image_paths(images_root: Path, fold: int) -> list[Path]:
    root = images_root.resolve()
    if root.name != "images" or root.parent.name != "test":
        raise ValueError("only an explicit outer-test images root is accepted")
    if root.parent.parent.name != f"fold_{fold}":
        raise ValueError("outer-test fold path mismatch")
    paths = sorted(
        path for path in root.iterdir() if path.name.endswith((".nii", ".nii.gz"))
    )
    if len(paths) != 14:
        raise ValueError(f"expected 14 outer-test images, found {len(paths)}")
    return paths


def _load_model(checkpoint: Path, device: torch.device) -> tuple[torch.nn.Module, int]:
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    state = payload.get("model_state_dict")
    if not isinstance(state, dict):
        raise ValueError("checkpoint does not contain model_state_dict")
    model = TianDamsehVNet(
        in_channels=1,
        out_channels=1,
        base_channels=4,
    ).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, int(payload.get("epoch", -1))


def run_inference(
    *,
    repository_root: Path,
    protocol_path: Path,
    images_root: Path,
    checkpoint: Path,
    expected_checkpoint_sha256: str,
    authorization_manifest: Path,
    assignment_manifest: Path,
    expected_assignment_sha256: str,
    fold: int,
    run_id: str,
    output_root: Path,
    expected_commit: str,
) -> dict[str, Any]:
    protocol = load_protocol(protocol_path)
    if float(protocol["threshold"]) != 0.5:
        raise ValueError("frozen operating point drift")
    _authorization_guard(
        manifest_path=authorization_manifest,
        assignment_manifest=assignment_manifest,
        expected_assignment_sha256=expected_assignment_sha256,
        fold=fold,
        run_id=run_id,
    )
    image_paths = _image_paths(images_root, fold)
    repository_root = repository_root.resolve()
    if _git(repository_root, "status", "--porcelain"):
        raise RuntimeError("standalone GPL checkout must be clean")
    observed_commit = _git(repository_root, "rev-parse", "HEAD")
    if observed_commit != expected_commit:
        raise RuntimeError(
            f"standalone checkout mismatch: {observed_commit} != {expected_commit}"
        )
    checkpoint = checkpoint.resolve()
    if sha256(checkpoint) != expected_checkpoint_sha256:
        raise ValueError("checkpoint SHA-256 mismatch")
    output_root = output_root.resolve()
    if output_root == repository_root or repository_root in output_root.parents:
        raise ValueError("prediction output must remain outside the GPL checkout")

    result_path = output_root / "prediction_manifest.json"
    if result_path.is_file():
        existing = json.loads(result_path.read_text(encoding="utf-8"))
        if existing.get("status") == "completed":
            return existing

    result: dict[str, Any] = {
        "schema_version": 1,
        "status": "starting",
        "evidence_scope": "modified_vnet_frozen_checkpoint_outer_test_inference_only",
        "model_key": protocol["model_key"],
        "fold": fold,
        "run_id": run_id,
        "standalone_commit": observed_commit,
        "protocol": str(protocol_path.resolve()),
        "protocol_sha256": sha256(protocol_path),
        "authorization_manifest": str(authorization_manifest.resolve()),
        "authorization_manifest_sha256": sha256(authorization_manifest),
        "assignment_sha256": expected_assignment_sha256,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": expected_checkpoint_sha256,
        "started_at_utc": _utc_now(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_node": os.environ.get("SLURMD_NODENAME"),
        "outer_test_accessed": True,
        "training_performed": False,
        "checkpoint_selection_performed": False,
        "threshold_tuning_performed": False,
        "threshold": 0.5,
        "performance_superiority_claim_allowed": False,
    }
    _atomic_json(result_path, result)

    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for outer-test inference")
        device = torch.device("cuda")
        _seed_everything(int(protocol["seed"]))
        model, selected_epoch = _load_model(checkpoint, device)
        result.update(
            {
                "status": "running",
                "phase": "inference",
                "gpu_name": torch.cuda.get_device_name(0),
                "selected_epoch_from_frozen_checkpoint": selected_epoch,
                "prediction_count_expected": 14,
            }
        )
        _atomic_json(result_path, result)

        from .runner import infer_volume

        prediction_root = output_root / "predictions"
        predictions = []
        started = time.perf_counter()
        for index, image_path in enumerate(image_paths, start=1):
            name = image_path.name.removesuffix(".gz").removesuffix(".nii")
            probability = infer_volume(
                model,
                load_nii(image_path),
                protocol,
                device,
            )
            if not np.isfinite(probability).all():
                raise ValueError(f"non-finite prediction for {name}")
            prediction_path = prediction_root / f"{name}.npy"
            _atomic_numpy(prediction_path, probability)
            predictions.append(
                {
                    "name": name,
                    "path": str(prediction_path),
                    "sha256": sha256(prediction_path),
                    "dtype": str(probability.dtype),
                    "shape_hwd": list(probability.shape),
                }
            )
            result["progress"] = {
                "completed": index,
                "total": len(image_paths),
                "last_case": name,
            }
            _atomic_json(result_path, result)
        torch.cuda.synchronize()
        result.update(
            {
                "status": "completed",
                "phase": "completed",
                "elapsed_seconds": time.perf_counter() - started,
                "predictions": predictions,
            }
        )
        _atomic_json(result_path, result)
        return result
    except BaseException as error:
        result.update(
            {
                "status": "failed",
                "phase": "failed",
                "error": str(error),
                "traceback": traceback.format_exc(),
            }
        )
        _atomic_json(result_path, result)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--images-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--authorization-manifest", type=Path, required=True)
    parser.add_argument("--assignment-manifest", type=Path, required=True)
    parser.add_argument("--expected-assignment-sha256", required=True)
    parser.add_argument("--fold", type=int, choices=range(5), required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    args = parser.parse_args()
    repository_root = Path(__file__).resolve().parents[1]
    try:
        payload = run_inference(
            repository_root=repository_root,
            protocol_path=args.protocol,
            images_root=args.images_root,
            checkpoint=args.checkpoint,
            expected_checkpoint_sha256=args.expected_checkpoint_sha256,
            authorization_manifest=args.authorization_manifest,
            assignment_manifest=args.assignment_manifest,
            expected_assignment_sha256=args.expected_assignment_sha256,
            fold=args.fold,
            run_id=args.run_id,
            output_root=args.output_root,
            expected_commit=args.expected_commit,
        )
    except BaseException as error:
        print(json.dumps({"status": "failed", "error": str(error)}, sort_keys=True))
        return 2
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
