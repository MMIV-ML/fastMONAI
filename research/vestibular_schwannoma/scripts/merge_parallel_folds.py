#!/usr/bin/env python3
"""Merge parallel fold-training manifests into one inference selection."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESEARCH_ROOT = PROJECT_ROOT.parent
if str(RESEARCH_ROOT) not in sys.path:
    sys.path.insert(0, str(RESEARCH_ROOT))

from vestibular_schwannoma.workflow.results import (  # noqa: E402
    build_model_comparison,
    combine_fold_results,
    write_cv_summary,
)
from vestibular_schwannoma.workflow.run_selection import (  # noqa: E402
    CROSS_VALIDATION_FOLDS,
    merge_fold_run_selections,
    read_inference_manifest,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate and merge disjoint fold manifests produced by parallel "
            "training jobs, and pool their held-out results."
        )
    )
    parser.add_argument(
        "selection_files",
        nargs="+",
        type=Path,
        help=(
            "Source completed_run_ids.json or inference_run_ids.json files to merge, "
            "relative to the VS project directory unless absolute."
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
        help=(
            "New directory in which to write the merged inference_run_ids.json, "
            "relative to the VS project directory unless absolute."
        ),
    )
    parser.add_argument(
        "--data-csv",
        type=Path,
        default=Path("data/ml_dataset.csv"),
        help="Dataset index, relative to the VS project directory unless absolute.",
    )
    return parser


def _project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _listed_fold_results(
    selection_files: list[Path], model_key: str
) -> dict[int, pd.DataFrame]:
    """Read ``<model>/fold_<n>/results.csv`` for the folds each selection lists."""

    fold_results = {}
    for path in selection_files:
        _, manifest = read_inference_manifest(path)
        for member in manifest["models"][model_key]["best"]:
            fold = int(member.removeprefix("fold_"))
            csv_path = path.parent / model_key / member / "results.csv"
            if csv_path.is_file():
                fold_results[fold] = pd.read_csv(csv_path)
    return fold_results


def _write_pooled_summary(
    selection_files: list[Path], model_key: str, output_root: Path, data_csv: Path
) -> None:
    combined = combine_fold_results(
        _listed_fold_results(selection_files, model_key),
        expected_folds=CROSS_VALIDATION_FOLDS,
        dataset_df=pd.read_csv(data_csv),
    )
    write_cv_summary(combined, output_root / model_key)
    comparison_path = output_root / "cv_model_comparison.csv"
    build_model_comparison({model_key: combined}).to_csv(comparison_path)
    print(f"Cross-model comparison: {comparison_path}")


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    selection_files = [_project_path(path) for path in args.selection_files]
    output_root = _project_path(args.output_root)
    destination = merge_fold_run_selections(
        selection_files,
        model_key=args.model,
        output_root=output_root,
    )
    print(f"Merged inference run selection: {destination}")
    try:
        _write_pooled_summary(
            selection_files, args.model, output_root, _project_path(args.data_csv)
        )
    except (OSError, ValueError, KeyError) as error:
        print(f"Pooled summary skipped: {error}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
