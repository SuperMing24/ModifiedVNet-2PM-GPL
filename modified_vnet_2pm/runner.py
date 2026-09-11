# SPDX-License-Identifier: GPL-3.0-only
"""Standalone training and prediction runner for the frozen comparator protocol."""

from __future__ import annotations

import json
import os
import random
import signal
import subprocess
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from monai.inferers import sliding_window_inference
from torch.utils.data import DataLoader

from .checkpoint_metrics import checkpoint_score, cldice_3d, dice_3d
from .data import TianDamsehGridPatchDataset3D, load_split
from .loss import TianDamsehTVBCELoss
from .model import TianDamsehVNet
from .protocol import load_protocol, sha256, validate_split_inputs
from .recovery import atomic_replace, atomic_torch, output_lock, project_epoch_views, validate_snapshot
from .sampling import execution_identity, load_sampling_contract


class TerminationRequested(RuntimeError):
    pass


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    atomic_replace(temporary, path)


def _atomic_torch(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    atomic_replace(temporary, path)


def _atomic_numpy(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, value)
    atomic_replace(temporary, path)


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


def _optimizer(model: torch.nn.Module, protocol: dict[str, Any]) -> torch.optim.Optimizer:
    decay: list[torch.nn.Parameter] = []
    no_decay: list[torch.nn.Parameter] = []
    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        (decay if parameter.ndim > 1 else no_decay).append(parameter)
    return torch.optim.Adam(
        [
            {"params": decay, "weight_decay": float(protocol["weight_decay"])},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=float(protocol["learning_rate"]),
        betas=(float(protocol["beta1"]), float(protocol["beta2"])),
        weight_decay=0.0,
    )


def _rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: dict[str, Any]) -> None:
    cuda_states = state.get("cuda", [])
    if torch.cuda.is_available() and not cuda_states:
        raise ValueError("resume CUDA RNG state is missing")
    if cuda_states and (not torch.cuda.is_available() or len(cuda_states) != torch.cuda.device_count()):
        raise ValueError("resume CUDA RNG device-count mismatch")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if cuda_states:
        torch.cuda.set_rng_state_all([value.cpu() for value in cuda_states])


def _training_device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the complete protocol")
    return torch.device("cuda")


def infer_volume(
    model: torch.nn.Module,
    volume_hwd: np.ndarray,
    protocol: dict[str, Any],
    device: torch.device,
) -> np.ndarray:
    minimum = float(volume_hwd.min())
    maximum = float(volume_hwd.max())
    normalized = (volume_hwd - minimum) / (maximum - minimum + 1e-8)
    tensor = (
        torch.from_numpy(normalized.transpose(2, 0, 1))
        .float()
        .unsqueeze(0)
        .unsqueeze(0)
        .to(device)
    )
    with torch.no_grad():
        logits = sliding_window_inference(
            tensor,
            tuple(int(value) for value in protocol["patch_size"]),
            int(protocol["inference_sw_batch_size"]),
            model,
            overlap=float(protocol["inference_overlap"]),
            mode="gaussian",
        )
    return (
        torch.sigmoid(logits)[0, 0]
        .cpu()
        .numpy()
        .transpose(1, 2, 0)
        .astype(np.float32)
    )


def _checkpoint_validation(
    model: torch.nn.Module,
    validation_volumes: list[dict],
    protocol: dict[str, Any],
    device: torch.device,
) -> tuple[dict[str, float], list[dict[str, float | str]]]:
    model.eval()
    rows: list[dict[str, float | str]] = []
    threshold = float(protocol["threshold"])
    for volume in validation_volumes:
        probability = infer_volume(model, volume["image"], protocol, device)
        rows.append(
            {
                "name": volume["name"],
                "dice_3d": dice_3d(probability, volume["mask"], threshold),
                "cl_dice_3d": cldice_3d(probability, volume["mask"], threshold),
            }
        )
    numeric_rows = [
        {"dice_3d": float(row["dice_3d"]), "cl_dice_3d": float(row["cl_dice_3d"])}
        for row in rows
    ]
    return checkpoint_score(numeric_rows), rows


def _checkpoint_payload(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_metric: float | None,
    protocol_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "epoch": epoch,
        "best_metric": best_metric,
        "model_state_dict": {
            key: value.detach().cpu().clone() for key, value in model.state_dict().items()
        },
        "optimizer_state_dict": optimizer.state_dict(),
        "rng_state": _rng_state(),
        "protocol_sha256": protocol_sha256,
    }


def run(**kwargs: Any) -> dict[str, Any]:
    root = Path(kwargs["repository_root"]).resolve()
    output = Path(kwargs["output_root"]).resolve()
    if output == root or root in output.parents:
        raise ValueError("result output must remain outside the GPL checkout")
    with output_lock(output):
        return _run(**kwargs)


def _run(
    *,
    repository_root: Path,
    protocol_path: Path,
    train_root: Path,
    validation_root: Path,
    assignment_manifest: Path,
    expected_assignment_sha256: str,
    fold: int,
    output_root: Path,
    expected_commit: str,
    sampling_manifest: Path | None = None,
    expected_sampling_manifest_sha256: str | None = None,
    subset_key: str | None = None,
) -> dict[str, Any]:
    protocol = load_protocol(protocol_path)
    validate_split_inputs(
        train_root,
        validation_root,
        assignment_manifest,
        expected_assignment_sha256,
        fold,
    )
    sampling = None
    sampling_args = (sampling_manifest, expected_sampling_manifest_sha256, subset_key)
    if any(value is not None for value in sampling_args):
        if not all(value is not None for value in sampling_args):
            raise ValueError("all three sampling CLI arguments must be supplied together")
        sampling = load_sampling_contract(
            manifest_path=sampling_manifest,
            expected_manifest_sha256=expected_sampling_manifest_sha256,
            subset_key=subset_key,
            assignment_manifest=assignment_manifest,
            expected_assignment_sha256=expected_assignment_sha256,
            fold=fold,
            train_root=train_root,
            validation_root=validation_root,
        )
    repository_root = repository_root.resolve()
    if _git(repository_root, "status", "--porcelain"):
        raise RuntimeError("standalone GPL checkout must be clean")
    observed_commit = _git(repository_root, "rev-parse", "HEAD")
    if observed_commit != expected_commit:
        raise RuntimeError(
            f"standalone checkout mismatch: {observed_commit} != {expected_commit}"
        )
    output_root = output_root.resolve()
    if output_root == repository_root or repository_root in output_root.parents:
        raise ValueError("result output must remain outside the GPL checkout")

    result_path = output_root / "training_result.json"
    identity = execution_identity(sampling, sha256(protocol_path), observed_commit) if sampling else None
    existing: dict[str, Any] = {}
    if result_path.is_file():
        existing = json.loads(result_path.read_text(encoding="utf-8"))
        if sampling and existing.get("execution_identity") != identity:
            raise ValueError("existing output sampling identity mismatch")
        if sampling and existing.get("last_completed_epoch", 0) > 0 and not (output_root / "checkpoints/latest_model.pt").is_file():
            raise ValueError("committed progress exists but latest checkpoint is missing")
        if existing.get("status") == "completed":
            if sampling:
                if sha256(Path(existing["checkpoint"])) != existing["checkpoint_sha256"]:
                    raise ValueError("completed checkpoint digest mismatch")
                for prediction in existing["prediction_manifest"]:
                    if sha256(Path(prediction["path"])) != prediction["sha256"]:
                        raise ValueError("completed prediction digest mismatch")
            return existing

    result: dict[str, Any] = {
        "schema_version": 1,
        "status": "starting",
        "model_key": protocol["model_key"],
        "fold": fold,
        "standalone_commit": observed_commit,
        "protocol": str(protocol_path.resolve()),
        "protocol_sha256": sha256(protocol_path),
        "assignment_sha256": expected_assignment_sha256,
        "started_at_utc": _utc_now(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_node": os.environ.get("SLURMD_NODENAME"),
        "training_uses_labels": True,
        "validation_masks_used_for_optimizer_updates": False,
        "validation_masks_used_for_checkpoint_selection": True,
        "non_authoritative_validation_patch_loss_omitted": True,
        "outer_test_accessed": False,
        "performance_superiority_claim_allowed": False,
    }
    if sampling:
        result["execution_identity"] = identity
        result["effective_training_seed"] = sampling["training_seed"]
        result["initial_started_at_utc"] = existing.get("initial_started_at_utc", result["started_at_utc"])
        result["previous_attempts"] = existing.get("previous_attempts", [])
        if existing:
            result["previous_attempts"].append({
                key: existing.get(key) for key in
                ("started_at_utc", "completed_at_utc", "slurm_job_id", "status", "last_completed_epoch")
            })
    _atomic_json(result_path, result)

    try:
        device = _training_device()
        _seed_everything(sampling["training_seed"] if sampling else int(protocol["seed"]))
        model = TianDamsehVNet(
            in_channels=1,
            out_channels=1,
            base_channels=4,
        ).to(device)
        optimizer = _optimizer(model, protocol)
        loss_function = TianDamsehTVBCELoss(
            alpha=float(protocol["loss_alpha"]),
            balance_epsilon=float(protocol["balance_epsilon"]),
        ).to(device)

        training_volumes = load_split(train_root)
        validation_volumes = load_split(validation_root)
        expected_train_count = len(sampling["membership"]["train_ids"]) if sampling else 42
        if len(training_volumes) != expected_train_count or len(validation_volumes) != 14:
            raise ValueError(
                "frozen split cardinality mismatch: "
                f"train={len(training_volumes)} val={len(validation_volumes)}"
            )
        training_dataset = TianDamsehGridPatchDataset3D(
            training_volumes,
            patch_size=protocol["patch_size"],
            stride=protocol["train_stride"],
        )
        training_loader = DataLoader(
            training_dataset,
            batch_size=int(protocol["batch_size"]),
            shuffle=True,
            num_workers=0,
        )
        if not sampling and len(training_dataset) != 2058:
            raise ValueError(
                f"official grid cardinality drift: {len(training_dataset)} != 2058"
            )

        checkpoint_root = output_root / "checkpoints"
        best_path = checkpoint_root / "best_model.pt"
        latest_path = checkpoint_root / "latest_model.pt"
        history_path = output_root / "training_history.jsonl"
        start_epoch = 1
        best_metric: float | None = None
        resumed_from: str | None = None
        history: list[dict[str, Any]] = []
        best_state: dict[str, Any] | None = None
        if latest_path.is_file():
            state = torch.load(latest_path, map_location="cpu", weights_only=False)
            if state.get("protocol_sha256") != result["protocol_sha256"]:
                raise ValueError("resume checkpoint protocol drift")
            if sampling:
                validate_snapshot(state, identity, len(training_dataset))
                history = state["history"]
                best_state = state["best_checkpoint"]
                project_epoch_views(output_root, state)
            model.load_state_dict(state["model_state_dict"], strict=True)
            optimizer.load_state_dict(state["optimizer_state_dict"])
            _restore_rng_state(state["rng_state"])
            start_epoch = int(state["epoch"]) + 1
            best_metric = state.get("best_metric")
            resumed_from = str(latest_path)
        elif sampling and (best_path.exists() or history_path.exists()):
            raise ValueError("latest checkpoint missing; refusing best fallback or silent restart")

        result.update(
            {
                "status": "running",
                "phase": "training",
                "gpu_name": torch.cuda.get_device_name(0),
                "parameter_count": sum(
                    parameter.numel() for parameter in model.parameters()
                ),
                "training_volume_count": len(training_volumes),
                "validation_volume_count": len(validation_volumes),
                "training_patch_count": len(training_dataset),
                "batches_per_epoch": len(training_loader),
                "start_epoch": start_epoch,
                "resumed_from": resumed_from,
                "last_completed_epoch": start_epoch - 1,
            }
        )
        _atomic_json(result_path, result)

        train_started = time.perf_counter()
        for epoch in range(start_epoch, int(protocol["epochs"]) + 1):
            epoch_started = time.perf_counter()
            model.train()
            losses: list[float] = []
            for data, target in training_loader:
                data = data.to(device)
                target = target.to(device)
                optimizer.zero_grad()
                logits = model(data)
                loss = loss_function(logits, target)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite training loss at epoch {epoch}")
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach().cpu()))

            selection, rows = _checkpoint_validation(
                model,
                validation_volumes,
                protocol,
                device,
            )
            is_new_best = (
                best_metric is None
                or float(selection["checkpoint_score"]) > best_metric
            )
            if is_new_best:
                best_metric = float(selection["checkpoint_score"])
            epoch_row = {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                **selection,
                "is_new_best": is_new_best,
                "best_metric": best_metric,
                "epoch_seconds": time.perf_counter() - epoch_started,
                "validation": rows,
            }
            payload = _checkpoint_payload(model, optimizer, epoch, best_metric, result["protocol_sha256"])
            if sampling:
                payload.update({
                    "schema_version": 2,
                    "execution_identity": identity,
                    "training_patch_count": len(training_dataset),
                })
                if is_new_best:
                    best_state = {key: payload[key] for key in (
                        "schema_version", "model_state_dict", "epoch", "best_metric",
                        "protocol_sha256", "execution_identity", "training_patch_count",
                    )}
                history.append(epoch_row)
                payload.update({"history": history, "best_checkpoint": best_state})
                atomic_torch(latest_path, payload)
                project_epoch_views(output_root, payload)
            else:
                if is_new_best:
                    _atomic_torch(best_path, payload)
                _atomic_torch(latest_path, payload)
                with history_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(epoch_row, sort_keys=True) + "\n")
            result.update(
                {
                    "last_completed_epoch": epoch,
                    "best_metric": best_metric,
                    "last_epoch": epoch_row,
                    "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(),
                }
            )
            _atomic_json(result_path, result)
            print(json.dumps(epoch_row, sort_keys=True), flush=True)

        if not best_path.is_file():
            raise FileNotFoundError("best checkpoint is missing")
        best = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(best["model_state_dict"], strict=True)
        model.eval()
        prediction_root = output_root / "predictions"
        prediction_manifest = []
        result["phase"] = "best_checkpoint_prediction_export"
        _atomic_json(result_path, result)
        for index, volume in enumerate(validation_volumes, start=1):
            probability = infer_volume(model, volume["image"], protocol, device)
            prediction_path = prediction_root / f"{volume['name']}.npy"
            _atomic_numpy(prediction_path, probability)
            prediction_manifest.append(
                {
                    "name": volume["name"],
                    "path": str(prediction_path),
                    "sha256": sha256(prediction_path),
                    "shape_hwd": list(probability.shape),
                }
            )
            result["prediction_export_progress"] = {
                "completed": index,
                "total": len(validation_volumes),
                "last_case": volume["name"],
            }
            _atomic_json(result_path, result)
        torch.cuda.synchronize()
        result.update(
            {
                "status": "completed",
                "phase": "completed",
                "completed_at_utc": _utc_now(),
                "training_elapsed_seconds": time.perf_counter() - train_started,
                "selected_epoch": int(best["epoch"]),
                "checkpoint_score": float(best["best_metric"]),
                "checkpoint": str(best_path),
                "checkpoint_sha256": sha256(best_path),
                "prediction_manifest": prediction_manifest,
                "outer_test_accessed": False,
            }
        )
        _atomic_json(result_path, result)
        return result
    except BaseException as error:
        result.update(
            {
                "status": "failed",
                "phase": "failed",
                "completed_at_utc": _utc_now(),
                "error": str(error),
                "traceback": traceback.format_exc(),
                "outer_test_accessed": False,
            }
        )
        _atomic_json(result_path, result)
        raise


def install_signal_handlers() -> None:
    def _terminate(signum: int, _frame: Any) -> None:
        raise TerminationRequested(f"received signal {signum}")

    signal.signal(signal.SIGTERM, _terminate)
    if hasattr(signal, "SIGUSR1"):
        signal.signal(signal.SIGUSR1, _terminate)
