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
from fastai.learner import Learner

from fastMONAI.vision_all import (
    AccumulatedDice,
    EMACheckpoint,
    MedPatchDataLoaders,
    PatchConfig,
    create_mlflow_callback,
    evaluate_segmentations,
    make_output_spec,
    patch_inference,
)
from fastMONAI.vision_metrics import _SURFACE_DISTANCE_SOURCE

from .config import ExperimentConfig, make_gpu_augmentation
from .models import ModelRecipe, build_training_model


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


def _write_benchmark_metadata(results_df: pd.DataFrame, destination: Path) -> None:
    payload = {
        "surface_distance_source": _SURFACE_DISTANCE_SOURCE,
        "nsd_tolerances_mm": [0.5, 1.0, 2.0],
        "nsd_headline_tau_mm": 1.0,
        "status_counts": results_df["surface_status"].value_counts().to_dict(),
        "per_case": [
            {
                "case_id": row["case_id"],
                "spacing_mm": list(row["spacing_mm"]),
                "surface_status": row["surface_status"],
            }
            for row in results_df.to_dict("records")
        ],
    }
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def evaluate_fold(
    learn,
    *,
    patch_config: PatchConfig,
    dls: MedPatchDataLoaders,
    pre_inference_transforms: list,
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
        pre_inference_tfms=pre_inference_transforms,
        save_dir=str(prediction_dir),
        progress=True,
        tta=use_tta,
    )
    results = evaluate_segmentations(
        predictions,
        mask_paths,
        case_ids=validation_df["case_id"].tolist(),
    )
    results.insert(1, "image", [Path(path).name for path in image_paths])
    results.to_csv(fold_dir / "results.csv", index=False)
    _write_benchmark_metadata(results, fold_dir / "benchmark_meta.json")

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
    recipe: ModelRecipe,
    fold: int,
    train_df: pd.DataFrame,
    *,
    experiment: ExperimentConfig,
    patch_config: PatchConfig,
    pre_inference_transforms: list,
    preprocessing_manifest: str | Path,
    results_dir: str | Path,
    output_spec: dict | None = None,
) -> FoldRun:
    """Train a fresh model and evaluate it on exactly one held-out fold."""

    output_spec = output_spec or make_output_spec(
        "multiclass_segmentation", classes=2
    )
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
    print(f"  {recipe.display_name.upper()}  |  FOLD {fold}")
    print(f"{'=' * 60}\n")
    print(
        f"[{recipe.key} fold {fold}] Train: {len(dls.train.subjects_dataset)}, "
        f"Val: {len(dls.valid.subjects_dataset)}"
    )

    mlflow_callback = None
    try:
        model = build_training_model(
            recipe, compile_model=experiment.compile_models
        )
        learn = Learner(
            dls,
            model,
            loss_func=recipe.make_loss(),
            metrics=[AccumulatedDice(n_classes=2)],
        ).to_bf16()
        save_best = EMACheckpoint(
            monitor="accumulated_dice",
            momentum=0.9,
            comp=np.greater,
            fname=recipe.checkpoint_name,
            with_opt=False,
        )
        mlflow_callback = create_mlflow_callback(
            learn,
            experiment_name=recipe.experiment_name,
            run_name=f"fold_{fold}",
            extra_tags={"fold": str(fold), "run_group": Path(results_dir).parent.name},
            preprocessing_manifest=preprocessing_manifest,
            sample_id_col="t1_img_path",
            model_spec=recipe.model_spec,
            output_spec=output_spec,
        )
        learn.fit_one_cycle(
            experiment.epochs,
            experiment.learning_rate,
            cbs=[mlflow_callback, save_best],
        )
        learn.load(recipe.checkpoint_name)
        results = evaluate_fold(
            learn,
            patch_config=patch_config,
            dls=dls,
            pre_inference_transforms=pre_inference_transforms,
            fold=fold,
            use_tta=experiment.use_tta,
            mlflow_callback=mlflow_callback,
            results_dir=results_dir,
        )
        return FoldRun(
            model_key=recipe.key,
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
    recipe: ModelRecipe,
    train_df: pd.DataFrame,
    *,
    experiment: ExperimentConfig,
    patch_config: PatchConfig,
    preprocessing_manifest: str | Path,
    run_group: str,
    output_spec: dict | None = None,
) -> str:
    """Train every case, using duplicated rows only for monitoring during fitting."""

    output_spec = output_spec or make_output_spec(
        "multiclass_segmentation", classes=2
    )
    available_folds = sorted(map(int, train_df["fold"].dropna().unique()))
    if not available_folds:
        raise ValueError("No folds are available for all-data monitoring")
    monitor_fold = np.random.default_rng(
        experiment.all_data_monitor_seed
    ).choice(available_folds).item()
    monitor_df = train_df[train_df["fold"] == monitor_fold].copy()
    print(
        f"[{recipe.key} all_data] selected fold {monitor_fold} for duplicated monitoring "
        f"(seed={experiment.all_data_monitor_seed}); these cases remain in training"
    )

    all_train_df = train_df.copy()
    all_train_df["is_val"] = False
    monitor_df["is_val"] = True
    fit_df = pd.concat([all_train_df, monitor_df], ignore_index=True)
    dls = MedPatchDataLoaders.from_df(
        df=fit_df,
        img_col="t1_img_path_preprocessed",
        mask_col="t1_seg_path_preprocessed",
        valid_col="is_val",
        patch_config=patch_config,
        gpu_augmentation=make_gpu_augmentation(experiment),
        bs=experiment.batch_size,
    )

    mlflow_callback = None
    try:
        model = build_training_model(
            recipe, compile_model=experiment.compile_models
        )
        learn = Learner(
            dls,
            model,
            loss_func=recipe.make_loss(),
            metrics=[AccumulatedDice(n_classes=2)],
        ).to_bf16()
        mlflow_callback = create_mlflow_callback(
            learn,
            experiment_name=recipe.experiment_name,
            run_name="all_data",
            extra_tags={
                "training_scope": "all_data",
                "run_group": run_group,
                "monitor_fold": str(monitor_fold),
                "monitor_is_training_duplicate": "true",
            },
            preprocessing_manifest=preprocessing_manifest,
            sample_id_col="t1_img_path",
            model_spec=recipe.model_spec,
            output_spec=output_spec,
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
    recipes: dict[str, ModelRecipe],
    train_df: pd.DataFrame,
    *,
    experiment: ExperimentConfig,
    patch_config: PatchConfig,
    pre_inference_transforms: list,
    preprocessing_manifest: str | Path,
    results_root: str | Path,
    output_spec: dict | None = None,
) -> TrainingSweep:
    """Run every requested fold and/or all-data model with structured failures."""

    results_root = Path(results_root)
    output_spec = output_spec or make_output_spec(
        "multiclass_segmentation", classes=2
    )
    sweep = TrainingSweep()

    for model_key, recipe in recipes.items():
        model_results_dir = results_root / model_key
        sweep.fold_runs[model_key] = {}
        if experiment.run_cross_validation:
            model_results_dir.mkdir(parents=True, exist_ok=True)
            for fold in experiment.folds:
                try:
                    run = train_one_fold(
                        recipe,
                        fold,
                        train_df,
                        experiment=experiment,
                        patch_config=patch_config,
                        pre_inference_transforms=pre_inference_transforms,
                        preprocessing_manifest=preprocessing_manifest,
                        results_dir=model_results_dir,
                        output_spec=output_spec,
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
                    recipe,
                    train_df,
                    experiment=experiment,
                    patch_config=patch_config,
                    preprocessing_manifest=preprocessing_manifest,
                    run_group=results_root.name,
                    output_spec=output_spec,
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

    print("\nSweep complete.")
    return sweep
