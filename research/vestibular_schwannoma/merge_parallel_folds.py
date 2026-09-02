#!/usr/bin/env python3
"""Merge parallel fold-training manifests into one inference selection."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
RESEARCH_ROOT = PROJECT_ROOT.parent
if str(RESEARCH_ROOT) not in sys.path:
    sys.path.insert(0, str(RESEARCH_ROOT))

from vestibular_schwannoma.workflow.run_selection import (  # noqa: E402
    merge_fold_run_selections,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate and merge disjoint fold manifests produced by parallel "
            "training jobs."
        )
    )
    parser.add_argument(
        "selection_files",
        nargs="+",
        type=Path,
        help=(
            "Source completed_run_ids.json or inference_run_ids.json files to merge."
        ),
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model key whose best fold runs should be merged, for example unet.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="New directory in which to write the merged inference_run_ids.json.",
    )
    return parser


def _project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    destination = merge_fold_run_selections(
        [_project_path(path) for path in args.selection_files],
        model_key=args.model,
        output_root=_project_path(args.output_root),
    )
    print(f"Merged inference run selection: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
