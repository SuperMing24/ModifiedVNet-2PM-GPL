# SPDX-License-Identifier: GPL-3.0-only
"""Command-line boundary for the standalone GPL comparator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .runner import install_signal_handlers, run


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument("--validation-root", type=Path, required=True)
    parser.add_argument("--assignment-manifest", type=Path, required=True)
    parser.add_argument("--expected-assignment-sha256", required=True)
    parser.add_argument("--fold", type=int, choices=range(5), required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    args = parser.parse_args()
    install_signal_handlers()
    repository_root = Path(__file__).resolve().parents[1]
    try:
        payload = run(
            repository_root=repository_root,
            protocol_path=args.protocol,
            train_root=args.train_root,
            validation_root=args.validation_root,
            assignment_manifest=args.assignment_manifest,
            expected_assignment_sha256=args.expected_assignment_sha256,
            fold=args.fold,
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
