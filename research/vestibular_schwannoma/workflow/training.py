"""Training and held-out evaluation orchestration for the VS project."""

from __future__ import annotations

import gc
import json
import traceback
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchio as tio
from fastai.learner import Learner
from fastai.torch_core import set_seed
from scipy.ndimage import label

from fastMONAI.vision_all import (
    AccumulatedDice,
    EMACheckpoint,
    MedPatchDataLoaders,
    PatchConfig,
    create_mlflow_callback,
    evaluate_segmentations,
    patch_inference,
)
from .config import (
    INFERENCE_RUN_IDS_FILENAME,
    INFERENCE_RUN_IDS_SCHEMA,
    VS_OUTPUT_SPEC,
    ExperimentConfig,
    make_gpu_augmentation,
)
from .models import TrainingModelConfig, build_training_model


@dataclass(frozen=True)
class FoldRun:
    """Completed held-out fold and its persisted results."""

    model_key: str
    fold: int
    run_id: str
    results_dir: Path
    results: pd.DataFrame


@dataclass(frozen=True)
class RunFailure:
    """A failed fold or all-data run recorded without retaining an exception object."""

    model_key: str
    stage: str
    error_type: str
    message: str
    fold: int | None = None


@dataclass
class TrainingSweep:
    """Structured outcome from all requested model runs."""

    fold_runs: dict[str, dict[int, FoldRun]] = field(default_factory=dict)
    all_data_run_ids: dict[str, str] = field(default_factory=dict)
    failures: list[RunFailure] = field(default_factory=list)
    inference_run_ids_path: Path | None = None


def _write_inference_run_ids(
    sweep: TrainingSweep,
    experiment: ExperimentConfig,
    results_root: Path,
) -> Path:
    """Persist exact MLflow runs for each fully completed inference role."""

    models = {}
    requested_folds = set(experiment.folds)
    for model_key in experiment.model_keys:
        roles = {}
        fold_runs = sweep.fold_runs.get(model_key, {})
        if experiment.run_cross_validation and set(fold_runs) == requested_folds:
            roles["best"] = {
                f"fold_{fold}": fold_runs[fold].run_id for fold in experiment.folds
            }
        if experiment.train_all_data and model_key in sweep.all_data_run_ids:
            roles["final"] = {"all_data": sweep.all_data_run_ids[model_key]}
        if roles:
            models[model_key] = roles

    manifest = {
        "schema_version": INFERENCE_RUN_IDS_SCHEMA,
        "run_group": results_root.name,
        "models": models,
    }
    results_root.mkdir(parents=True, exist_ok=True)
    destination = results_root / INFERENCE_RUN_IDS_FILENAME
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    temporary.replace(destination)
    return destination


def _mark_failed(callback) -> None:
    """Mark a callback-owned run failed through its public lifecycle API."""

    if callback is None:
        return
    callback.mark_failed()


def _release_training_resources(dls) -> None:
    try:
        if dls is not None:
            dls.close()
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


def _set_training_seed(seed: int) -> None:
    # Seed Python, NumPy, PyTorch, and CUDA while retaining cuDNN performance optimizations.
    set_seed(seed, reproducible=False)


def _gpu_augmentations_json(gpu_augmentation) -> str:
    """Serialize the applied GPU augmentation pipeline for MLflow."""

    transformations = [
        {"name": name, "params": params}
        for name, params in vars(gpu_augmentation).items()
        if params is not None
    ]
    return json.dumps(transformations, indent=2, separators=(",", ": "))


def _component_counts(prediction, ground_truth_path: str | Path) -> dict[str, int]:
    """Count predicted regions and regions with no ground-truth overlap."""

    prediction_array = np.asarray(
        prediction.detach().cpu()
        if isinstance(prediction, torch.Tensor)
        else prediction
    ).squeeze().astype(bool)
    ground_truth = (
        np.asarray(tio.LabelMap(ground_truth_path).data).squeeze().astype(bool)
    )
    if prediction_array.shape != ground_truth.shape:
        raise ValueError(
            "prediction and ground truth must share a voxel grid; got "
            f"{prediction_array.shape} and {ground_truth.shape}"
        )
    labels, component_count = label(prediction_array)
    false_positive_count = sum(
        not np.any(ground_truth[labels == component])
        for component in range(1, component_count + 1)
    )
    return {
        "predicted_component_count": int(component_count),
        "false_positive_component_count": int(false_positive_count),
    }


def evaluate_fold(
    learn,
    *,
    patch_config: PatchConfig,
    dls: MedPatchDataLoaders,
    fold: int,
    use_tta: bool,
    mlflow_callback,
    results_dir: str | Path,
) -> pd.DataFrame:
    """Run native-space inference and metrics for one held-out fold."""

    fold_dir = Path(results_dir) / f"fold_{fold}"
    prediction_dir = fold_dir / "predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)

    learn.cuda()
    validation_df = dls.split_df.query("is_valid").reset_index(drop=True)
    image_paths = validation_df["t1_img_path"].tolist()
    mask_paths = validation_df["t1_seg_path"].tolist()

    print(f"[Fold {fold}] Running inference on {len(image_paths)} validation images...")
    predictions = patch_inference(
        learner=learn,
        config=patch_config,
        file_paths=image_paths,
        save_dir=str(prediction_dir),
        progress=True,
        tta=use_tta,
    )
    results = evaluate_segmentations(
        predictions,
        mask_paths,
        case_ids=validation_df["case_id"].tolist(),
    )
    component_metrics = pd.DataFrame(
        [
            _component_counts(prediction, mask_path)
            for prediction, mask_path in zip(predictions, mask_paths)
        ]
    )
    results = pd.concat(
        [results.reset_index(drop=True), component_metrics], axis=1
    )
    results.insert(1, "image", [Path(path).name for path in image_paths])
    results.to_csv(fold_dir / "results.csv", index=False)

    numeric = results.select_dtypes(include="number").replace(
        [np.inf, -np.inf], np.nan
    )
    mlflow_callback.log_metrics_table(results, display=False)
    mlflow_callback.log_metrics(
        {f"val_{metric}": numeric[metric].mean() for metric in numeric.columns}
    )
    mlflow_callback.log_dataframe(results)

    print(f"[Fold {fold}] Results saved to {fold_dir / 'results.csv'}")
    print(f"[Fold {fold}] DSC: {results['dsc'].mean():.4f} +/- {results['dsc'].std():.4f}")
    return results


def train_one_fold(
    model_config: TrainingModelConfig,
    fold: int,
    train_df: pd.DataFrame,
    *,
    experiment: ExperimentConfig,
    patch_config: PatchConfig,
    preprocessing_manifest: str | Path,
    results_dir: str | Path,
) -> FoldRun:
    """Train a fresh model and evaluate it on exactly one held-out fold."""

    _set_training_seed(experiment.training_seed)
    fold_df = train_df.copy()
    fold_df["is_val"] = fold_df["fold"] == fold
    gpu_augmentation = make_gpu_augmentation(experiment)
    dls = MedPatchDataLoaders.from_df(
        df=fold_df,
        img_col="t1_img_path_preprocessed",
        mask_col="t1_seg_path_preprocessed",
        valid_col="is_val",
        patch_config=patch_config,
        gpu_augmentation=gpu_augmentation,
        bs=experiment.batch_size,
    )

    print(f"\n{'=' * 60}")
    print(f"  {model_config.display_name.upper()}  |  FOLD {fold}")
    print(f"{'=' * 60}\n")
    print(
        f"[{model_config.key} fold {fold}] Train: {len(dls.train.subjects_dataset)}, "
        f"Val: {len(dls.valid.subjects_dataset)}"
    )

    mlflow_callback = None
    try:
        model = build_training_model(
            model_config, compile_model=experiment.compile_models
        )
        learn = Learner(
            dls,
            model,
            loss_func=model_config.make_loss(),
            metrics=[AccumulatedDice(n_classes=2)],
        ).to_bf16()
        save_best = EMACheckpoint(
            monitor="accumulated_dice",
            momentum=0.9,
            comp=np.greater,
            fname=model_config.checkpoint_name,
            with_opt=False,
        )
        mlflow_callback = create_mlflow_callback(
            learn,
            experiment_name=model_config.experiment_name,
            run_name=f"fold_{fold}",
            extra_tags={"fold": str(fold), "run_group": Path(results_dir).parent.name},
            extra_params={
                "training_seed": experiment.training_seed,
                "foreground_sampling_probability": (
                    experiment.foreground_sampling_probability
                ),
                "gpu_augmentations": _gpu_augmentations_json(gpu_augmentation),
            },
            preprocessing_manifest=preprocessing_manifest,
            sample_id_col="case_id",
            model_spec=model_config.model_spec,
            output_spec=VS_OUTPUT_SPEC,
        )
        learn.fit_one_cycle(
            experiment.epochs,
            experiment.learning_rate,
            cbs=[mlflow_callback, save_best],
        )
        learn.load(model_config.checkpoint_name)
        results = evaluate_fold(
            learn,
            patch_config=patch_config,
            dls=dls,
            fold=fold,
            use_tta=experiment.use_tta,
            mlflow_callback=mlflow_callback,
            results_dir=results_dir,
        )
        return FoldRun(
            model_key=model_config.key,
            fold=fold,
            run_id=mlflow_callback.run_id,
            results_dir=Path(results_dir) / f"fold_{fold}",
            results=results,
        )
    except Exception:
        _mark_failed(mlflow_callback)
        raise
    finally:
        _release_training_resources(dls)


def train_all_data_model(
    model_config: TrainingModelConfig,
    train_df: pd.DataFrame,
    *,
    experiment: ExperimentConfig,
    patch_config: PatchConfig,
    preprocessing_manifest: str | Path,
    run_group: str,
) -> str:
    """Train every case, duplicating one stable case only for internal monitoring."""

    _set_training_seed(experiment.training_seed)
    if train_df.empty:
        raise ValueError("No cases are available for all-data training")
    if "case_id" not in train_df.columns or train_df["case_id"].isna().any():
        raise ValueError("case_id is required for stable all-data monitoring")
    monitor_df = train_df.sort_values("case_id", kind="stable").iloc[[0]].copy()
    monitor_case_id = str(monitor_df.iloc[0]["case_id"])
    print(
        f"[{model_config.key} all_data] duplicated case {monitor_case_id!r} for internal "
        "monitoring; it remains in training"
    )

    all_train_df = train_df.copy()
    all_train_df["is_val"] = False
    monitor_df["is_val"] = True
    fit_df = pd.concat([all_train_df, monitor_df], ignore_index=True)
    gpu_augmentation = make_gpu_augmentation(experiment)
    dls = MedPatchDataLoaders.from_df(
        df=fit_df,
        img_col="t1_img_path_preprocessed",
        mask_col="t1_seg_path_preprocessed",
        valid_col="is_val",
        patch_config=patch_config,
        gpu_augmentation=gpu_augmentation,
        bs=experiment.batch_size,
    )

    mlflow_callback = None
    try:
        model = build_training_model(
            model_config, compile_model=experiment.compile_models
        )
        learn = Learner(
            dls,
            model,
            loss_func=model_config.make_loss(),
            metrics=[AccumulatedDice(n_classes=2)],
        ).to_bf16()
        mlflow_callback = create_mlflow_callback(
            learn,
            experiment_name=model_config.experiment_name,
            run_name="all_data",
            extra_tags={
                "training_scope": "all_data",
                "run_group": run_group,
                "monitor_case_id": monitor_case_id,
                "monitor_is_training_duplicate": "true",
            },
            extra_params={
                "training_seed": experiment.training_seed,
                "foreground_sampling_probability": (
                    experiment.foreground_sampling_probability
                ),
                "gpu_augmentations": _gpu_augmentations_json(gpu_augmentation),
            },
            preprocessing_manifest=preprocessing_manifest,
            sample_id_col="case_id",
            model_spec=model_config.model_spec,
            output_spec=VS_OUTPUT_SPEC,
        )
        learn.fit_one_cycle(
            experiment.epochs,
            experiment.learning_rate,
            cbs=[mlflow_callback],
        )
        return mlflow_callback.run_id
    except Exception:
        _mark_failed(mlflow_callback)
        raise
    finally:
        _release_training_resources(dls)


def run_training_sweep(
    model_configs: dict[str, TrainingModelConfig],
    train_df: pd.DataFrame,
    *,
    experiment: ExperimentConfig,
    patch_config: PatchConfig,
    preprocessing_manifest: str | Path,
    results_root: str | Path,
) -> TrainingSweep:
    """Run every requested fold and/or all-data model with structured failures."""

    results_root = Path(results_root)
    sweep = TrainingSweep()

    for model_key, model_config in model_configs.items():
        model_results_dir = results_root / model_key
        sweep.fold_runs[model_key] = {}
        if experiment.run_cross_validation:
            model_results_dir.mkdir(parents=True, exist_ok=True)
            for fold in experiment.folds:
                try:
                    run = train_one_fold(
                        model_config,
                        fold,
                        train_df,
                        experiment=experiment,
                        patch_config=patch_config,
                        preprocessing_manifest=preprocessing_manifest,
                        results_dir=model_results_dir,
                    )
                    sweep.fold_runs[model_key][fold] = run
                except Exception as error:
                    sweep.failures.append(
                        RunFailure(
                            model_key=model_key,
                            fold=fold,
                            stage="cross_validation",
                            error_type=type(error).__name__,
                            message=str(error),
                        )
                    )
                    print(
                        f"[FAILED] {model_key} fold {fold}: "
                        f"{type(error).__name__}: {error}"
                    )
                    traceback.print_exc()
                    if not experiment.continue_on_error:
                        raise

        if experiment.train_all_data:
            try:
                sweep.all_data_run_ids[model_key] = train_all_data_model(
                    model_config,
                    train_df,
                    experiment=experiment,
                    patch_config=patch_config,
                    preprocessing_manifest=preprocessing_manifest,
                    run_group=results_root.name,
                )
                print(
                    f"[{model_key}] all-data MLflow run: "
                    f"{sweep.all_data_run_ids[model_key]}"
                )
            except Exception as error:
                sweep.failures.append(
                    RunFailure(
                        model_key=model_key,
                        stage="all_data",
                        error_type=type(error).__name__,
                        message=str(error),
                    )
                )
                print(
                    f"[FAILED] {model_key} all_data: "
                    f"{type(error).__name__}: {error}"
                )
                traceback.print_exc()
                if not experiment.continue_on_error:
                    raise

    sweep.inference_run_ids_path = _write_inference_run_ids(
        sweep, experiment, results_root
    )
    print(f"Inference run selection: {sweep.inference_run_ids_path}")
    print("\nSweep complete.")
    return sweep
