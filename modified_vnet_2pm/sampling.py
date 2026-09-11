# SPDX-License-Identifier: GPL-3.0-only
"""Frozen, hash-pinned subset ledger boundary; never an execution authorization."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from .protocol import sha256


COUNTS = {80: 34, 60: 25, 40: 17, 20: 8}


def canonical_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    ).hexdigest()


def _ids(value: Any, label: str, count: int) -> list[str]:
    if not isinstance(value, list) or len(value) != count:
        raise ValueError(f"{label}: expected {count} members")
    if any(not isinstance(item, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", item) for item in value):
        raise ValueError(f"{label}: unsafe member ID")
    if len(set(value)) != count:
        raise ValueError(f"{label}: duplicate members")
    return sorted(value)


def _split_members(root: Path, expected: list[str]) -> None:
    for leaf in ("images", "masks"):
        directory = root / leaf
        observed: list[str] = []
        for path in directory.iterdir():
            if not path.name.endswith((".nii", ".nii.gz")):
                continue
            if not path.is_file():
                raise ValueError(f"not a regular volume: {path}")
            if "test" in path.resolve().parts or "outer-test" in path.resolve().parts:
                raise ValueError("train/val volume resolves into a test directory")
            observed.append(path.name.removesuffix(".gz").removesuffix(".nii"))
        if sorted(observed) != expected:
            raise ValueError(f"{root.name}/{leaf}: exact subset membership mismatch")


def load_sampling_contract(
    *,
    manifest_path: Path,
    expected_manifest_sha256: str,
    subset_key: str,
    assignment_manifest: Path,
    expected_assignment_sha256: str,
    fold: int,
    train_root: Path,
    validation_root: Path,
) -> dict[str, Any]:
    if not re.fullmatch(r"[0-9a-f]{64}", expected_manifest_sha256 or ""):
        raise ValueError("expected sampling manifest SHA256 is required")
    if sha256(manifest_path) != expected_manifest_sha256:
        raise ValueError("sampling manifest SHA256 mismatch")
    ledger = json.loads(manifest_path.read_text(encoding="utf-8"))
    if ledger.get("schema_version") != 1 or ledger.get("frozen") is not True:
        raise ValueError("sampling ledger must be schema 1 and explicitly frozen")
    if not isinstance(ledger.get("protocol_version"), str) or not ledger["protocol_version"]:
        raise ValueError("sampling protocol version is required")
    if ledger.get("outer_fold") != fold or ledger.get("assignment_sha256") != expected_assignment_sha256:
        raise ValueError("sampling parent assignment/fold mismatch")
    if ledger.get("source_sha256") != sha256(assignment_manifest):
        raise ValueError("sampling parent manifest file SHA256 mismatch")
    seed = ledger.get("training_seed")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("training_seed must be an integer in [0, 2**32)")
    matches = [row for row in ledger.get("subsets", []) if row.get("subset_key") == subset_key]
    if len(matches) != 1:
        raise ValueError("subset_key must identify exactly one registered subset")
    subset = matches[0]
    fraction = subset.get("fraction_percent")
    if type(fraction) is not int or fraction not in COUNTS:
        raise ValueError("supported fraction_percent values are 80, 60, 40, 20")
    if subset.get("outer_fold") != fold or type(subset.get("repeat")) is not int or subset["repeat"] < 1:
        raise ValueError("subset fold/repeat mismatch")
    fixed = ledger["fixed_splits"]
    pool = _ids(fixed.get("train_pool"), "parent train pool", 42)
    val = _ids(fixed.get("validation"), "fixed validation", 14)
    test = _ids(fixed.get("outer_test"), "excluded test IDs", 14)
    if set(pool) & set(val) or set(pool) & set(test) or set(val) & set(test):
        raise ValueError("parent train/val/test IDs overlap")
    train = _ids(subset.get("train_ids"), "subset training", COUNTS[fraction])
    if not set(train) <= set(pool):
        raise ValueError("subset contains members outside the parent training pool")
    if _ids(subset.get("val_ids"), "subset validation", 14) != val:
        raise ValueError("validation members must remain fixed")
    if _ids(subset.get("test_ids"), "subset excluded test IDs", 14) != test:
        raise ValueError("excluded test members drift")
    _split_members(train_root, train)
    _split_members(validation_root, val)
    membership = {"fold": fold, "train_ids": train, "val_ids": val}
    return {
        "schema_version": 1,
        "sampling_manifest_sha256": expected_manifest_sha256,
        "sampling_protocol_version": ledger["protocol_version"],
        "parent_manifest_sha256": ledger["source_sha256"],
        "parent_assignment_sha256": expected_assignment_sha256,
        "subset_key": subset_key,
        "fraction_percent": fraction,
        "repeat": subset["repeat"],
        "training_seed": seed,
        "membership": membership,
        "membership_sha256": canonical_hash(membership),
        "test_volumes_opened": False,
        "execution_authority": "caller must independently validate stage budget and registration",
    }


def execution_identity(contract: dict[str, Any], protocol_sha256: str, commit: str) -> dict[str, Any]:
    identity = {
        "schema_version": 1,
        "sampling": contract,
        "protocol_sha256": protocol_sha256,
        "standalone_commit": commit,
    }
    return {**identity, "sha256": canonical_hash(identity)}
