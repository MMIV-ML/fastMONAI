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

    python train_5fold.py --help
    python train_5fold.py --models unet --folds 1 --epochs 5 --no-compile

Run one model across all five folds::

    python train_5fold.py --models unet

Run the default five folds and skip an optional model if it is not installed::

    python train_5fold.py --skip-unavailable

The defaults request four models, folds 1-5, and 500 epochs. Models and folds run
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
``cv_results/<UTC timestamp>/`` unless overridden. Generated data, caches, MLflow
state, predictions, and weights stay outside Git through the project ``.gitignore``.

This launcher performs cross-validation only. Use the shared workflow directly (or
extend the CLI explicitly) if an all-data final/deployment model is required.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch

from fastMONAI.vision_all import MedDataset, MedMask, ZNormalization, preprocess_dataset


PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from workflow.config import ExperimentConfig, make_patch_config  # noqa: E402
from workflow.models import (  # noqa: E402
    TRAINING_MODEL_CONFIGS,
    get_training_model_configs,
)
from workflow.results import aggregate_results, build_model_comparison  # noqa: E402
from workflow.training import run_training_sweep  # noqa: E402


DEFAULT_MODELS = ("unet", "dynunet", "dynunet_small", "segmamba")


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
        help="Model keys to train in order (default: all four).",
    )
    parser.add_argument(
        "--folds",
        nargs="+",
        type=int,
        default=[1, 2, 3, 4, 5],
        help="Held-out folds to run (default: 1 2 3 4 5).",
    )
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--foreground-probability",
        type=float,
        default=0.8,
        help=(
            "Probability of sampling a foreground-centred patch. The default 0.8 "
            "preserves the original 80%% foreground / 20%% background control. "
            "Pass 0.7 for the planned 70%% / 30%% comparison."
        ),
    )
    parser.add_argument(
        "--samples-per-volume",
        type=int,
        default=4,
        help="Patches extracted from each volume per epoch (default: 4).",
    )
    parser.add_argument(
        "--queue-workers",
        type=int,
        default=4,
        help="CPU workers that fill the training patch queue (default: 4).",
    )
    parser.add_argument(
        "--queue-length",
        type=int,
        default=300,
        help="Maximum patches buffered in RAM (default: 300).",
    )
    parser.add_argument(
        "--preprocess-workers",
        type=int,
        default=min(32, os.cpu_count() or 1),
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
        help="Output directory (default: cv_results/<UTC timestamp>).",
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


def _load_and_validate_dataset(data_csv: Path, folds: tuple[int, ...]) -> pd.DataFrame:
    if not data_csv.is_file():
        raise FileNotFoundError(f"Dataset CSV not found: {data_csv}")
    frame = pd.read_csv(data_csv)
    required = {"case_id", "t1_img_path", "t1_seg_path", "fold"}
    missing_columns = sorted(required - set(frame.columns))
    if missing_columns:
        raise ValueError(f"Dataset CSV is missing columns: {missing_columns}")
    if frame["case_id"].isna().any() or frame["case_id"].duplicated().any():
        raise ValueError("case_id values must be present and unique")
    if frame[["t1_img_path", "t1_seg_path", "fold"]].isna().any().any():
        raise ValueError("Image paths, mask paths, and fold values must be present")

    available_folds = set(map(int, frame["fold"].unique()))
    missing_folds = sorted(set(folds) - available_folds)
    if missing_folds:
        raise ValueError(f"Dataset does not contain requested folds: {missing_folds}")

    missing_files = []
    for column in ("t1_img_path", "t1_seg_path"):
        for value in frame[column]:
            path = Path(value)
            if not path.is_file():
                missing_files.append(str(path))
                if len(missing_files) == 10:
                    break
        if len(missing_files) == 10:
            break
    if missing_files:
        shown = "\n  ".join(missing_files)
        raise FileNotFoundError(f"Missing dataset files (first 10):\n  {shown}")
    return frame


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
        target_spacing=(0.4102, 0.4102, 1.5),
        patch_size=(192, 192, 48),
        preprocess_workers=args.preprocess_workers,
        samples_per_volume=args.samples_per_volume,
        queue_num_workers=args.queue_workers,
        queue_length=args.queue_length,
        foreground_sampling_probability=args.foreground_probability,
        continue_on_error=not args.stop_on_error,
    )

    data_csv = _project_path(args.data_csv)
    preprocessed_dir = _project_path(args.preprocessed_dir)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    results_root = _project_path(args.results_root or Path("cv_results") / timestamp)
    _print_plan(experiment, data_csv, results_root)

    model_configs = get_training_model_configs(
        experiment.model_keys,
        skip_unavailable=args.skip_unavailable,
    )
    train_df = _load_and_validate_dataset(data_csv, experiment.folds)
    print(f"Validated {len(train_df)} cases")
    print(train_df["fold"].value_counts().sort_index().to_string())

    normalization = [ZNormalization(masking_method="foreground")]
    label_dataset = MedDataset(
        img_list=train_df["t1_seg_path"].tolist(),
        dtype=MedMask,
        max_workers=experiment.preprocess_workers,
    )
    dataset_version = label_dataset.fingerprint
    if dataset_version is None:
        raise RuntimeError("Could not fingerprint the segmentation masks")

    preprocessing_result = preprocess_dataset(
        train_df,
        img_col="t1_img_path",
        mask_col="t1_seg_path",
        output_dir=str(preprocessed_dir),
        target_spacing=list(experiment.target_spacing),
        apply_reorder=True,
        transforms=normalization,
        max_workers=experiment.preprocess_workers,
        dataset_version=dataset_version,
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
