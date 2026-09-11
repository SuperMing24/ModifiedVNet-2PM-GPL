# SPDX-License-Identifier: GPL-3.0-only

import copy
import json
import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from modified_vnet_2pm import runner
from modified_vnet_2pm import recovery
from modified_vnet_2pm.protocol import load_protocol, sha256
from modified_vnet_2pm.recovery import output_lock, validate_snapshot
from modified_vnet_2pm.sampling import COUNTS, load_sampling_contract


ROOT = Path(__file__).resolve().parents[1]


def fixture_ledger(tmp_path, fraction=80):
    pool = [f"tr{i:02d}" for i in range(42)]
    val = [f"va{i:02d}" for i in range(14)]
    test = [f"te{i:02d}" for i in range(14)]
    train = pool[:COUNTS[fraction]]
    for split, members in (("train", train), ("val", val)):
        for leaf in ("images", "masks"):
            directory = tmp_path / "fold_0" / split / leaf
            directory.mkdir(parents=True)
            for member in members:
                (directory / f"{member}.nii").touch()
    assignment = tmp_path / "parent.json"
    assignment.write_text(json.dumps({"assignment_sha256": "a" * 64}), encoding="utf-8")
    payload = {
        "schema_version": 1, "frozen": True, "protocol_version": "synthetic-test-v1",
        "outer_fold": 0, "training_seed": 17,
        "source_sha256": sha256(assignment), "assignment_sha256": "a" * 64,
        "fixed_splits": {"train_pool": pool, "validation": val, "outer_test": test},
        "subsets": [{"subset_key": "synthetic", "outer_fold": 0,
                     "fraction_percent": fraction, "repeat": 1,
                     "train_ids": train, "val_ids": val, "test_ids": test}],
    }
    manifest = tmp_path / "sample_usage.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    args = {
        "manifest_path": manifest, "expected_manifest_sha256": sha256(manifest),
        "subset_key": "synthetic", "assignment_manifest": assignment,
        "expected_assignment_sha256": "a" * 64, "fold": 0,
        "train_root": tmp_path / "fold_0/train", "validation_root": tmp_path / "fold_0/val",
    }
    return payload, args


@pytest.mark.parametrize("fraction", [80, 60, 40, 20])
def test_four_cardinalities_and_seed_are_bound_without_test_files(tmp_path, fraction):
    _, args = fixture_ledger(tmp_path, fraction)
    result = load_sampling_contract(**args)
    assert len(result["membership"]["train_ids"]) == COUNTS[fraction]
    assert len(result["membership"]["val_ids"]) == 14
    assert result["training_seed"] == 17
    assert len(result["membership_sha256"]) == 64
    assert not result["test_volumes_opened"]
    assert not (tmp_path / "fold_0/test").exists()


@pytest.mark.parametrize("change", ["preview", "seed_bool", "seed_float", "seed_negative", "seed_large",
                                    "train_duplicate", "train_leak", "val_drift", "test_drift",
                                    "source_hash", "assignment", "fold", "repeat", "fraction",
                                    "duplicate_subset", "unsafe_id"])
def test_rejects_unfrozen_or_drifted_membership(tmp_path, change):
    payload, args = fixture_ledger(tmp_path)
    subset = payload["subsets"][0]
    if change == "preview": payload["frozen"] = False
    elif change == "seed_bool": payload["training_seed"] = True
    elif change == "seed_float": payload["training_seed"] = 2.5
    elif change == "seed_negative": payload["training_seed"] = -1
    elif change == "seed_large": payload["training_seed"] = 2**32
    elif change == "train_duplicate": subset["train_ids"][0] = subset["train_ids"][1]
    elif change == "train_leak": subset["train_ids"][0] = "te00"
    elif change == "val_drift": subset["val_ids"] = ["drift"] + subset["val_ids"][1:]
    elif change == "test_drift": subset["test_ids"] = ["drift"] + subset["test_ids"][1:]
    elif change == "source_hash": payload["source_sha256"] = "b" * 64
    elif change == "assignment": payload["assignment_sha256"] = "b" * 64
    elif change == "fold": subset["outer_fold"] = 1
    elif change == "repeat": subset["repeat"] = 0
    elif change == "fraction": subset["fraction_percent"] = 100
    elif change == "duplicate_subset": payload["subsets"].append(copy.deepcopy(subset))
    elif change == "unsafe_id": subset["train_ids"][0] = "../te00"
    args["manifest_path"].write_text(json.dumps(payload), encoding="utf-8")
    args["expected_manifest_sha256"] = sha256(args["manifest_path"])
    with pytest.raises(ValueError):
        load_sampling_contract(**args)


@pytest.mark.parametrize("leaf,kind", [("images", "extra"), ("masks", "missing"), ("images", "duplicate")])
def test_exact_on_disk_membership(tmp_path, leaf, kind):
    _, args = fixture_ledger(tmp_path)
    directory = args["train_root"] / leaf
    if kind == "missing": (directory / "tr00.nii").unlink()
    elif kind == "duplicate": (directory / "tr00.nii.gz").touch()
    else: (directory / "te00.nii").touch()
    with pytest.raises(ValueError, match="membership mismatch"):
        load_sampling_contract(**args)


def test_rejects_manifest_bytes_changed_after_pin(tmp_path):
    _, args = fixture_ledger(tmp_path)
    with args["manifest_path"].open("a") as handle:
        handle.write(" ")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        load_sampling_contract(**args)


def test_cuda_rng_states_are_returned_to_cpu_without_using_a_gpu(monkeypatch):
    state = runner._rng_state()
    calls = []

    class MappedTensor:
        def cpu(self):
            calls.append("cpu")
            return torch.zeros(8, dtype=torch.uint8)

    state["cuda"] = [MappedTensor()]
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    received = []
    monkeypatch.setattr(torch.cuda, "set_rng_state_all", lambda values: received.extend(values))
    runner._restore_rng_state(state)
    assert calls == ["cpu"]
    assert received[0].device.type == "cpu"
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    with pytest.raises(ValueError, match="device-count"):
        runner._restore_rng_state(state)
    state.pop("cuda")
    with pytest.raises(ValueError, match="state is missing"):
        runner._restore_rng_state(state)


def tiny_cpu_runner(tmp_path, monkeypatch):
    _, sample_args = fixture_ledger(tmp_path)
    original = load_protocol(ROOT / "configs/frozen_protocol.json")
    monkeypatch.setattr(runner, "load_protocol", lambda path: dict(original, epochs=3))
    monkeypatch.setattr(runner, "_git", lambda root, *args: "" if args[0] == "status" else "f" * 40)
    monkeypatch.setattr(runner, "_training_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda index: "synthetic CPU fixture")
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 0)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(runner, "TianDamsehVNet", lambda **kwargs: torch.nn.Sequential(
        torch.nn.Linear(2, 3), torch.nn.Dropout(0.2), torch.nn.Linear(3, 1)))
    monkeypatch.setattr(runner, "TianDamsehTVBCELoss", lambda **kwargs: torch.nn.MSELoss())
    monkeypatch.setattr(runner, "load_split", lambda path: [
        {"name": file.stem, "image": np.zeros((1, 1, 1))}
        for file in sorted((path / "images").glob("*.nii"))])

    class TinyDataset:
        def __init__(self, volumes, **kwargs): self.count = len(volumes)
        def __len__(self): return self.count
        def __getitem__(self, index):
            x = torch.tensor([index / 100, random.random() + np.random.random()], dtype=torch.float32)
            return x, torch.tensor([index / 80], dtype=torch.float32)

    monkeypatch.setattr(runner, "TianDamsehGridPatchDataset3D", TinyDataset)
    def validation(model, volumes, protocol, device):
        model.eval()
        score = float(next(model.parameters()).detach().mean())
        return {"checkpoint_score": score, "dice_3d": score, "cl_dice_3d": score}, []
    monkeypatch.setattr(runner, "_checkpoint_validation", validation)
    monkeypatch.setattr(runner, "infer_volume", lambda *args: np.zeros((1, 1, 1), dtype=np.float32))
    return {
        "repository_root": ROOT, "protocol_path": ROOT / "configs/frozen_protocol.json",
        "train_root": sample_args["train_root"], "validation_root": sample_args["validation_root"],
        "assignment_manifest": sample_args["assignment_manifest"], "expected_assignment_sha256": "a" * 64,
        "fold": 0, "expected_commit": "f" * 40,
        "sampling_manifest": sample_args["manifest_path"],
        "expected_sampling_manifest_sha256": sample_args["expected_manifest_sha256"],
        "subset_key": "synthetic",
    }


def assert_equal_nested(left, right):
    if isinstance(left, torch.Tensor): assert torch.equal(left, right)
    elif isinstance(left, np.ndarray): np.testing.assert_array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left: assert_equal_nested(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right): assert_equal_nested(a, b)
    else: assert left == right


@pytest.mark.parametrize("phase", ["before_latest", "after_latest", "after_views", "mid_epoch", "export"])
@pytest.mark.parametrize("best_pattern", ["improving", "declining"])
def test_interrupted_cpu_training_matches_uninterrupted(tmp_path, monkeypatch, phase, best_pattern):
    args = tiny_cpu_runner(tmp_path / "input", monkeypatch)
    if best_pattern == "improving":
        validation = runner._checkpoint_validation
        def improving(*args):
            selection, rows = validation(*args)
            return {key: -value for key, value in selection.items()}, rows
        monkeypatch.setattr(runner, "_checkpoint_validation", improving)
    reference = tmp_path / "reference"
    resumed = tmp_path / "resumed"
    runner.run(**args, output_root=reference)
    target = {
        "before_latest": "atomic_torch", "after_latest": "project_epoch_views",
        "after_views": "_atomic_json", "mid_epoch": "_checkpoint_validation", "export": "infer_volume",
    }[phase]
    original = getattr(runner, target)
    seen = 0
    def interrupt(*pos, **kw):
        nonlocal seen
        seen += 1
        fire = ((phase == "before_latest" and pos[1]["epoch"] == 2)
                or (phase == "after_latest" and pos[1]["epoch"] == 2)
                or (phase == "after_views" and pos[1].get("last_completed_epoch") == 2)
                or (phase == "mid_epoch" and seen == 2)
                or (phase == "export" and seen == 1))
        if fire: raise RuntimeError("synthetic interruption")
        return original(*pos, **kw)
    monkeypatch.setattr(runner, target, interrupt)
    with pytest.raises(RuntimeError, match="synthetic interruption"):
        runner.run(**args, output_root=resumed)
    monkeypatch.setattr(runner, target, original)
    # Simulate an interrupted/stale log projection. The committed history repairs it.
    (resumed / "training_history.jsonl").write_text('{"epoch":999}\npartial', encoding="utf-8")
    result = runner.run(**args, output_root=resumed)
    assert result["status"] == "completed"
    assert result["effective_training_seed"] == 17
    assert result["training_patch_count"] == 34
    assert result["batches_per_epoch"] == 9
    left = torch.load(reference / "checkpoints/latest_model.pt", weights_only=False)
    right = torch.load(resumed / "checkpoints/latest_model.pt", weights_only=False)
    for state in (left, right):
        for row in state["history"]: row.pop("epoch_seconds")
    assert_equal_nested(left, right)
    best_left = torch.load(reference / "checkpoints/best_model.pt", weights_only=False)
    best_right = torch.load(resumed / "checkpoints/best_model.pt", weights_only=False)
    assert_equal_nested(best_left, best_right)
    assert best_right["epoch"] == (3 if best_pattern == "improving" else 1)
    history = [json.loads(line) for line in (resumed / "training_history.jsonl").read_text().splitlines()]
    assert [row["epoch"] for row in history] == [1, 2, 3]
    assert runner.run(**args, output_root=resumed)["status"] == "completed"
    changed = copy.deepcopy(right["execution_identity"])
    changed["sampling"]["training_seed"] += 1
    with pytest.raises(ValueError, match="identity"):
        validate_snapshot(right, changed, 34)
    with pytest.raises(ValueError, match="patch count"):
        validate_snapshot(right, right["execution_identity"], 35)


def test_missing_latest_never_silently_falls_back(tmp_path, monkeypatch):
    args = tiny_cpu_runner(tmp_path / "input", monkeypatch)
    output = tmp_path / "output"
    (output / "checkpoints").mkdir(parents=True)
    (output / "checkpoints/best_model.pt").touch()
    with pytest.raises(ValueError, match="refusing best fallback"):
        runner.run(**args, output_root=output)


def test_output_lock_rejects_concurrent_writer_and_releases(tmp_path):
    with output_lock(tmp_path):
        with pytest.raises(RuntimeError, match="another training process"):
            with output_lock(tmp_path): pass
    with output_lock(tmp_path): pass


def test_recipe_and_public_cli_remain_explicit():
    protocol = load_protocol(ROOT / "configs/frozen_protocol.json")
    assert (protocol["epochs"], protocol["seed"], protocol["batch_size"]) == (199, 42, 4)
    result = subprocess.run([sys.executable, "-B", "-m", "modified_vnet_2pm.cli", "--help"],
                            cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0
    for flag in ("--sampling-manifest", "--expected-sampling-manifest-sha256", "--subset-key"):
        assert flag in result.stdout


@pytest.mark.skipif(os.name != "nt", reason="Windows sharing-error compatibility")
def test_atomic_replace_retries_only_bounded_windows_permission_error(tmp_path, monkeypatch):
    path = tmp_path / "state.json"
    temporary = tmp_path / "state.tmp"
    temporary.write_text("committed", encoding="utf-8")
    original = Path.replace
    calls = []
    def busy_once(self, target):
        calls.append(1)
        if len(calls) < 3:
            error = PermissionError("synthetic Windows sharing lock")
            error.winerror = 32
            raise error
        return original(self, target)
    monkeypatch.setattr(Path, "replace", busy_once)
    monkeypatch.setattr(recovery.time, "sleep", lambda delay: None)
    recovery.atomic_replace(temporary, path)
    assert len(calls) == 3
    assert path.read_text() == "committed"
    calls.clear()
    def always_busy(self, target):
        calls.append(1)
        error = PermissionError("synthetic persistent sharing lock")
        error.winerror = 32
        raise error
    monkeypatch.setattr(Path, "replace", always_busy)
    with pytest.raises(PermissionError):
        recovery.atomic_replace(temporary, path)
    assert len(calls) == 5


def test_kernel_releases_lock_when_child_is_killed(tmp_path):
    code = ("from pathlib import Path; import time; "
            "from modified_vnet_2pm.recovery import output_lock; "
            "lock=output_lock(Path(__import__('sys').argv[1])); "
            "lock.__enter__(); print('locked', flush=True); time.sleep(60)")
    child = subprocess.Popen([sys.executable, "-B", "-c", code, str(tmp_path)],
                             cwd=ROOT, stdout=subprocess.PIPE, text=True)
    try:
        assert child.stdout.readline().strip() == "locked"
        with pytest.raises(RuntimeError, match="another training process"):
            with output_lock(tmp_path): pass
    finally:
        child.kill()
        child.wait(timeout=10)
        child.stdout.close()
    with output_lock(tmp_path): pass


def test_changed_identity_and_lost_latest_rejected_before_result_rewrite(tmp_path, monkeypatch):
    args = tiny_cpu_runner(tmp_path / "input", monkeypatch)
    output = tmp_path / "output"
    runner.run(**args, output_root=output)
    path = output / "training_result.json"
    before = path.read_bytes()
    changed = dict(args, expected_commit="e" * 40)
    monkeypatch.setattr(runner, "_git", lambda root, *args: "" if args[0] == "status" else "e" * 40)
    with pytest.raises(ValueError, match="identity mismatch"):
        runner.run(**changed, output_root=output)
    assert path.read_bytes() == before
    monkeypatch.setattr(runner, "_git", lambda root, *args: "" if args[0] == "status" else "f" * 40)
    (output / "checkpoints/latest_model.pt").unlink()
    with pytest.raises(ValueError, match="latest checkpoint is missing"):
        runner.run(**args, output_root=output)
    assert path.read_bytes() == before
