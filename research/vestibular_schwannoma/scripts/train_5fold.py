#!/usr/bin/env python3
"""Run reproducible five-fold VS training without using the notebook.

This is the supported command-line entry point for cross-validation experiments.
Reusable training logic lives in ``workflow/`` and is shared with
``notebooks/01_five_fold_cross_validation.ipynb``; do not duplicate that logic here.

Setup
-----
- Run in a fastMONAI environment with a CUDA GPU. SegMamba is optional.
- The default index is ``data/ml_dataset.csv``. Its image and mask paths expect
  ``research/nii_data/`` beside ``research/vestibular_schwannoma/``. Pass another
  index with ``--data-csv``; its relative paths resolve from the VS project directory.

Usage (from ``research/vestibular_schwannoma``)
------------------------------------------------
Inspect all options or run a short single-fold check::

    python scripts/train_5fold.py --help
    python scripts/train_5fold.py --models unet --folds 1 --epochs 5 --no-compile

Run one model across all five folds::

    python scripts/train_5fold.py --models unet

Run the default five folds and skip an optional model if it is not installed::

    python scripts/train_5fold.py --skip-unavailable

The defaults request three models, folds 1-5, and 500 epochs. Models and folds run
sequentially so only one model occupies GPU memory at a time. Each held-out fold is
evaluated with TTA.

Performance controls
--------------------
- ``torch.compile`` is enabled for supported models. It costs startup time but can
  speed long runs; use ``--no-compile`` for quick checks or compiler problems.
- ``--preprocess-workers N`` controls one-time preprocessing (default: up to 32).
- The patch queue defaults to 4 extraction workers and 300 buffered patches. If the
  GPU waits for data and CPU/RAM are available, raise ``--queue-workers`` first and
  then ``--queue-length``; a larger queue consumes more RAM.

Preprocessing is cached in ``preprocessed/`` and outputs go below
``cv_results/<UTC timestamp>/`` unless overridden. Every independently launched job must
use a new results root. Warm the preprocessing cache with one process before launching
multiple training jobs. Generated data, caches, MLflow state, predictions, and weights
stay outside Git through the project ``.gitignore``.

This launcher performs cross-validation only. Use the shared workflow directly (or
extend the CLI explicitly) if an all-data final/deployment model is required.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESEARCH_ROOT = PROJECT_ROOT.parent
if str(RESEARCH_ROOT) not in sys.path:
    sys.path.insert(0, str(RESEARCH_ROOT))

from vestibular_schwannoma.workflow.config import (  # noqa: E402
    ExperimentConfig,
    make_patch_config,
)
from vestibular_schwannoma.workflow.models import (  # noqa: E402
    TRAINING_MODEL_CONFIGS,
    get_training_model_configs,
)
from vestibular_schwannoma.workflow.results import (  # noqa: E402
    aggregate_results,
    build_model_comparison,
)
from vestibular_schwannoma.workflow.training import (  # noqa: E402
    load_training_index,
    preprocess_training_data,
    run_training_sweep,
)


_DEFAULTS = ExperimentConfig()
DEFAULT_MODELS = _DEFAULTS.model_keys


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train and evaluate the VS models with fixed five-fold cross-validation. "
            "Models and folds run sequentially to bound GPU memory use."
        )
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=tuple(TRAINING_MODEL_CONFIGS),
        default=list(DEFAULT_MODELS),
        help="Model keys to train in order (default: all three).",
    )
    parser.add_argument(
        "--folds",
        nargs="+",
        type=int,
        default=list(_DEFAULTS.folds),
        help="Held-out folds to run (default: 1 2 3 4 5).",
    )
    parser.add_argument("--epochs", type=int, default=_DEFAULTS.epochs)
    parser.add_argument("--batch-size", type=int, default=_DEFAULTS.batch_size)
    parser.add_argument(
        "--learning-rate", type=float, default=_DEFAULTS.learning_rate
    )
    parser.add_argument("--seed", type=int, default=_DEFAULTS.training_seed)
    parser.add_argument(
        "--foreground-probability",
        type=float,
        default=_DEFAULTS.foreground_sampling_probability,
        help=(
            "Probability of sampling a foreground-centred patch (default: "
            "%(default)s). Pass 0.7 for the planned 70%% / 30%% comparison."
        ),
    )
    parser.add_argument(
        "--samples-per-volume",
        type=int,
        default=_DEFAULTS.samples_per_volume,
        help="Patches extracted from each volume per epoch (default: %(default)s).",
    )
    parser.add_argument(
        "--queue-workers",
        type=int,
        default=_DEFAULTS.queue_num_workers,
        help="CPU workers that fill the training patch queue (default: %(default)s).",
    )
    parser.add_argument(
        "--queue-length",
        type=int,
        default=_DEFAULTS.queue_length,
        help="Maximum patches buffered in RAM (default: %(default)s).",
    )
    parser.add_argument(
        "--preprocess-workers",
        type=int,
        default=_DEFAULTS.preprocess_workers,
        help="CPU workers used for one-time preprocessing (default: up to 32).",
    )
    parser.add_argument(
        "--data-csv",
        type=Path,
        default=Path("data/ml_dataset.csv"),
        help="Dataset index, relative to the VS project directory unless absolute.",
    )
    parser.add_argument(
        "--preprocessed-dir",
        type=Path,
        default=Path("preprocessed"),
        help="Versioned preprocessing cache directory.",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        help=(
            "New output directory; it must not already exist "
            "(default: cv_results/<UTC timestamp>)."
        ),
    )
    parser.add_argument(
        "--no-compile",
        action="store_true",
        help="Disable torch.compile for models that support it.",
    )
    parser.add_argument(
        "--skip-unavailable",
        action="store_true",
        help="Skip an unavailable optional model such as SegMamba instead of failing early.",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop immediately if one fold fails instead of continuing to the next run.",
    )
    return parser


def _project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _print_plan(
    experiment: ExperimentConfig,
    data_csv: Path,
    results_root: Path,
) -> None:
    print("\nFive-fold training plan")
    print(f"  Project: {PROJECT_ROOT}")
    print(f"  Dataset: {data_csv}")
    print(f"  Models: {', '.join(experiment.model_keys)}")
    print(f"  Folds: {experiment.folds}")
    print(f"  Epochs: {experiment.epochs}")
    print(f"  Batch size: {experiment.batch_size}")
    print(f"  Learning rate: {experiment.learning_rate}")
    print(
        "  Patch sampling: "
        f"{experiment.foreground_sampling_probability:.0%} foreground / "
        f"{1 - experiment.foreground_sampling_probability:.0%} background"
    )
    print("  Held-out inference TTA: ON (fixed)")
    print("  Execution: sequential (one model/fold on the GPU at a time)")
    print(f"  Results: {results_root}\n")


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    os.chdir(PROJECT_ROOT)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for this training launcher")

    experiment = ExperimentConfig(
        model_keys=tuple(args.models),
        folds=tuple(args.folds),
        run_cross_validation=True,
        train_all_data=False,
        training_seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        use_tta=True,
        compile_models=not args.no_compile,
        preprocess_workers=args.preprocess_workers,
        samples_per_volume=args.samples_per_volume,
        queue_num_workers=args.queue_workers,
        queue_length=args.queue_length,
        foreground_sampling_probability=args.foreground_probability,
        continue_on_error=not args.stop_on_error,
    )

    data_csv = _project_path(args.data_csv)
    preprocessed_dir = _project_path(args.preprocessed_dir)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    results_root = _project_path(args.results_root or Path("cv_results") / timestamp)
    _print_plan(experiment, data_csv, results_root)

    model_configs = get_training_model_configs(
        experiment.model_keys,
        skip_unavailable=args.skip_unavailable,
    )
    train_df = load_training_index(data_csv, experiment.folds)
    print(f"Validated {len(train_df)} cases")
    print(train_df["fold"].value_counts().sort_index().to_string())

    normalization, dataset_version, preprocessing_result = preprocess_training_data(
        train_df, experiment, preprocessed_dir
    )
    print(f"Dataset version: {dataset_version}")
    print(f"Preprocessing cache: {preprocessing_result.cache_version}")
    print(f"Cache reused: {preprocessing_result.reused}")
    print(f"Manifest: {preprocessing_result.manifest_path}")

    torch.backends.cudnn.benchmark = True
    sweep = run_training_sweep(
        model_configs,
        train_df,
        experiment=experiment,
        patch_config=make_patch_config(experiment, normalization),
        preprocessing_manifest=preprocessing_result.manifest_path,
        results_root=results_root,
    )

    combined = {}
    for model_key in model_configs:
        result = aggregate_results(
            results_root / model_key,
            experiment.folds,
            train_df,
        )
        if result is not None:
            combined[model_key] = result
    comparison = build_model_comparison(combined)
    if not comparison.empty:
        comparison_path = results_root / "cv_model_comparison.csv"
        comparison.to_csv(comparison_path)
        print(f"Cross-model comparison: {comparison_path}")

    if sweep.failures:
        print(f"Training finished with {len(sweep.failures)} failed run(s):")
        for failure in sweep.failures:
            location = f" fold {failure.fold}" if failure.fold is not None else ""
            print(
                f"  {failure.model_key}{location}: "
                f"{failure.error_type}: {failure.message}"
            )
        return 1

    print(f"All requested runs completed: {results_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
