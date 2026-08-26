import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd
import torch

from vestibular_schwannoma.workflow.config import VS_OUTPUT_SPEC, ExperimentConfig
from vestibular_schwannoma.workflow.models import TRAINING_MODEL_CONFIGS
from vestibular_schwannoma.workflow import training


class TrainingOrchestrationTests(unittest.TestCase):
    def test_public_failure_hook_is_used(self):
        callback = MagicMock()
        training._mark_failed(callback)
        callback.mark_failed.assert_called_once_with()

    def test_training_seed_retains_cudnn_performance_settings(self):
        with patch.object(training, "set_seed") as set_seed:
            training._set_training_seed(42)

        set_seed.assert_called_once_with(42, reproducible=False)

    def test_gpu_augmentations_json_matches_the_applied_object(self):
        augmentation = SimpleNamespace(
            affine={"scales": (0.7, 1.4), "p": 0.2},
            flip={"axes": (0, 1, 2), "p": 0.5},
            noise=None,
        )

        self.assertEqual(
            json.loads(training._gpu_augmentations_json(augmentation)),
            [
                {
                    "name": "affine",
                    "params": {"scales": [0.7, 1.4], "p": 0.2},
                },
                {
                    "name": "flip",
                    "params": {"axes": [0, 1, 2], "p": 0.5},
                },
            ],
        )

    def test_fold_evaluation_reads_normalization_from_patch_config(self):
        patch_config = SimpleNamespace(normalization=[{"name": "ZNormalization"}])
        dls = MagicMock()
        dls.split_df = pd.DataFrame(
            {
                "case_id": ["case-1"],
                "t1_img_path": ["raw.nii.gz"],
                "t1_seg_path": ["raw_mask.nii.gz"],
                "is_valid": [True],
            }
        )
        metrics = pd.DataFrame(
            {
                "case_id": ["case-1"],
                "dsc": [0.8],
                "spacing_mm": [(0.5, 0.5, 1.0)],
                "surface_status": ["ok"],
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            with (
                patch.object(training, "patch_inference", return_value=[object()]) as infer,
                patch.object(training, "evaluate_segmentations", return_value=metrics),
                patch.object(
                    training,
                    "_component_counts",
                    return_value={
                        "predicted_component_count": 2,
                        "false_positive_component_count": 1,
                    },
                ) as component_counts,
                redirect_stdout(StringIO()),
            ):
                results = training.evaluate_fold(
                    MagicMock(),
                    patch_config=patch_config,
                    dls=dls,
                    fold=1,
                    use_tta=False,
                    mlflow_callback=MagicMock(),
                    results_dir=directory,
                )
                self.assertFalse(
                    (Path(directory) / "fold_1" / "benchmark_meta.json").exists()
                )

        self.assertIs(infer.call_args.kwargs["config"], patch_config)
        self.assertNotIn("pre_inference_tfms", infer.call_args.kwargs)
        component_counts.assert_called_once_with(
            infer.return_value[0], "raw_mask.nii.gz"
        )
        self.assertEqual(results["predicted_component_count"].tolist(), [2])
        self.assertEqual(results["false_positive_component_count"].tolist(), [1])

    def test_component_counts_preserve_disconnected_predictions(self):
        prediction = torch.zeros((1, 4, 4, 4), dtype=torch.long)
        prediction[0, 0, 0, 0] = 1
        prediction[0, 3, 3, 3] = 1
        ground_truth = torch.zeros_like(prediction)
        ground_truth[0, 0, 0, 0] = 1

        with patch.object(
            training.tio,
            "LabelMap",
            return_value=SimpleNamespace(data=ground_truth),
        ):
            counts = training._component_counts(prediction, "mask.nii.gz")

        self.assertEqual(
            counts,
            {
                "predicted_component_count": 2,
                "false_positive_component_count": 1,
            },
        )

    def test_fold_dataloaders_close_when_model_construction_fails(self):
        experiment = ExperimentConfig(
            model_keys=("unet",),
            folds=(1,),
            epochs=1,
            compile_models=False,
        )
        train_df = pd.DataFrame(
            {
                "case_id": ["a", "b"],
                "fold": [1, 2],
                "t1_img_path_preprocessed": ["a.nii.gz", "b.nii.gz"],
                "t1_seg_path_preprocessed": ["a_seg.nii.gz", "b_seg.nii.gz"],
            }
        )
        dls = MagicMock()
        dls.train.subjects_dataset = [object()]
        dls.valid.subjects_dataset = [object()]
        with (
            patch.object(
                training.MedPatchDataLoaders, "from_df", return_value=dls
            ),
            patch.object(training, "make_gpu_augmentation", return_value=object()),
            patch.object(
                training, "build_training_model", side_effect=RuntimeError("build failed")
            ),
            redirect_stdout(StringIO()),
        ):
            with self.assertRaisesRegex(RuntimeError, "build failed"):
                training.train_one_fold(
                    TRAINING_MODEL_CONFIGS["unet"],
                    1,
                    train_df,
                    experiment=experiment,
                    patch_config=SimpleNamespace(),
                    preprocessing_manifest="manifest.json",
                    results_dir="results/unet",
                )
        dls.close.assert_called_once_with()

    def test_fold_tracking_uses_fixed_model_and_output_contract(self):
        experiment = ExperimentConfig(
            model_keys=("unet",),
            folds=(1,),
            epochs=1,
            compile_models=False,
        )
        train_df = pd.DataFrame(
            {
                "case_id": ["a", "b"],
                "fold": [1, 2],
                "t1_img_path_preprocessed": ["a.nii.gz", "b.nii.gz"],
                "t1_seg_path_preprocessed": ["a_seg.nii.gz", "b_seg.nii.gz"],
            }
        )
        dls = MagicMock()
        dls.train.subjects_dataset = [object()]
        dls.valid.subjects_dataset = [object()]
        learner = MagicMock()
        callback = MagicMock(run_id="run-1")
        results = pd.DataFrame({"case_id": ["a"], "dsc": [0.8]})
        model_config = TRAINING_MODEL_CONFIGS["unet"]
        gpu_augmentation = SimpleNamespace(
            affine={"scales": (0.7, 1.4), "p": 0.2},
            flip={"axes": (0, 1, 2), "p": 0.5},
        )

        with (
            patch.object(training, "_set_training_seed") as set_training_seed,
            patch.object(training.MedPatchDataLoaders, "from_df", return_value=dls),
            patch.object(
                training,
                "make_gpu_augmentation",
                return_value=gpu_augmentation,
            ),
            patch.object(training, "build_training_model", return_value=object()),
            patch.object(training, "Learner") as learner_factory,
            patch.object(training, "EMACheckpoint", return_value=object()),
            patch.object(
                training, "create_mlflow_callback", return_value=callback
            ) as create_callback,
            patch.object(training, "evaluate_fold", return_value=results),
            redirect_stdout(StringIO()),
        ):
            learner_factory.return_value.to_bf16.return_value = learner
            training.train_one_fold(
                model_config,
                1,
                train_df,
                experiment=experiment,
                patch_config=SimpleNamespace(),
                preprocessing_manifest="manifest.json",
                results_dir="results/unet",
            )

        set_training_seed.assert_called_once_with(42)
        self.assertEqual(
            create_callback.call_args.kwargs["experiment_name"],
            "vestibular_schwannoma_unet",
        )
        self.assertEqual(
            create_callback.call_args.kwargs["model_spec"], model_config.model_spec
        )
        self.assertIs(
            create_callback.call_args.kwargs["output_spec"], VS_OUTPUT_SPEC
        )
        extra_params = create_callback.call_args.kwargs["extra_params"]
        self.assertEqual(extra_params["training_seed"], 42)
        self.assertEqual(extra_params["foreground_sampling_probability"], 0.8)
        self.assertEqual(
            json.loads(extra_params["gpu_augmentations"]),
            [
                {
                    "name": "affine",
                    "params": {"scales": [0.7, 1.4], "p": 0.2},
                },
                {
                    "name": "flip",
                    "params": {"axes": [0, 1, 2], "p": 0.5},
                },
            ],
        )
        self.assertEqual(create_callback.call_args.kwargs["sample_id_col"], "case_id")
        learner.load.assert_called_once_with("best_unet")
        dls.close.assert_called_once_with()

    def test_all_data_training_uses_one_stable_duplicated_monitor_case(self):
        experiment = ExperimentConfig(
            model_keys=("unet",),
            run_cross_validation=False,
            train_all_data=True,
            epochs=1,
            compile_models=False,
        )
        train_df = pd.DataFrame(
            {
                "case_id": ["case-b", "case-a"],
                "fold": [1, 2],
                "t1_img_path_preprocessed": ["b.nii.gz", "a.nii.gz"],
                "t1_seg_path_preprocessed": ["b_seg.nii.gz", "a_seg.nii.gz"],
            }
        )
        dls = MagicMock()
        learner = MagicMock()
        callback = MagicMock(run_id="run-final")
        gpu_augmentation = SimpleNamespace(
            affine={"scales": (0.7, 1.4), "p": 0.2},
            flip={"axes": (0, 1, 2), "p": 0.5},
        )
        events = []

        def seed_run(seed):
            events.append(("seed", seed))

        def make_dls(**kwargs):
            events.append(("dls", None))
            return dls

        def build_model(*args, **kwargs):
            events.append(("model", None))
            return object()

        with (
            patch.object(training, "_set_training_seed", side_effect=seed_run),
            patch.object(
                training.MedPatchDataLoaders,
                "from_df",
                side_effect=make_dls,
            ) as from_df,
            patch.object(
                training,
                "make_gpu_augmentation",
                return_value=gpu_augmentation,
            ),
            patch.object(training, "build_training_model", side_effect=build_model),
            patch.object(training, "Learner") as learner_factory,
            patch.object(
                training, "create_mlflow_callback", return_value=callback
            ) as create_callback,
            redirect_stdout(StringIO()),
        ):
            learner_factory.return_value.to_bf16.return_value = learner
            run_id = training.train_all_data_model(
                TRAINING_MODEL_CONFIGS["unet"],
                train_df,
                experiment=experiment,
                patch_config=SimpleNamespace(),
                preprocessing_manifest="manifest.json",
                run_group="run-group",
            )

        fit_df = from_df.call_args.kwargs["df"]
        training_rows = fit_df.loc[~fit_df["is_val"]]
        monitor_rows = fit_df.loc[fit_df["is_val"]]
        self.assertEqual(run_id, "run-final")
        self.assertEqual(events[:3], [("seed", 42), ("dls", None), ("model", None)])
        self.assertEqual(training_rows["case_id"].tolist(), ["case-b", "case-a"])
        self.assertEqual(monitor_rows["case_id"].tolist(), ["case-a"])
        self.assertEqual(
            create_callback.call_args.kwargs["extra_tags"],
            {
                "training_scope": "all_data",
                "run_group": "run-group",
                "monitor_case_id": "case-a",
                "monitor_is_training_duplicate": "true",
            },
        )
        extra_params = create_callback.call_args.kwargs["extra_params"]
        self.assertEqual(extra_params["training_seed"], 42)
        self.assertEqual(extra_params["foreground_sampling_probability"], 0.8)
        self.assertEqual(
            json.loads(extra_params["gpu_augmentations"]),
            [
                {
                    "name": "affine",
                    "params": {"scales": [0.7, 1.4], "p": 0.2},
                },
                {
                    "name": "flip",
                    "params": {"axes": [0, 1, 2], "p": 0.5},
                },
            ],
        )
        self.assertEqual(create_callback.call_args.kwargs["sample_id_col"], "case_id")
        dls.close.assert_called_once_with()

    def test_sweep_records_fold_failures_and_continues(self):
        experiment = ExperimentConfig(
            model_keys=("unet",),
            folds=(1, 2),
            epochs=1,
            compile_models=False,
        )
        train_df = pd.DataFrame({"case_id": ["a", "b"], "fold": [1, 2]})

        def fake_train(model_config, fold, *args, **kwargs):
            if fold == 1:
                raise RuntimeError("expected failure")
            return training.FoldRun(
                model_key=model_config.key,
                fold=fold,
                run_id="run-2",
                results_dir=Path("fold_2"),
                results=pd.DataFrame({"case_id": ["b"], "dsc": [0.8]}),
            )

        with tempfile.TemporaryDirectory() as directory:
            with (
                patch.object(training, "train_one_fold", side_effect=fake_train),
                redirect_stdout(StringIO()),
                redirect_stderr(StringIO()),
            ):
                sweep = training.run_training_sweep(
                    {"unet": TRAINING_MODEL_CONFIGS["unet"]},
                    train_df,
                    experiment=experiment,
                    patch_config=SimpleNamespace(),
                    preprocessing_manifest="manifest.json",
                    results_root=directory,
                )
            selection = json.loads(sweep.inference_run_ids_path.read_text())
        self.assertEqual(list(sweep.fold_runs["unet"]), [2])
        self.assertEqual(len(sweep.failures), 1)
        self.assertEqual(sweep.failures[0].fold, 1)
        self.assertEqual(sweep.failures[0].error_type, "RuntimeError")
        self.assertEqual(selection["models"], {})

    def test_sweep_writes_complete_fold_and_all_data_run_selection(self):
        experiment = ExperimentConfig(
            model_keys=("unet",),
            folds=(2, 4),
            run_cross_validation=True,
            train_all_data=True,
            epochs=1,
            compile_models=False,
        )
        train_df = pd.DataFrame({"case_id": ["a"], "fold": [2]})

        def fake_train(model_config, fold, *args, **kwargs):
            return training.FoldRun(
                model_key=model_config.key,
                fold=fold,
                run_id=f"run-{fold}",
                results_dir=Path(f"fold_{fold}"),
                results=pd.DataFrame({"case_id": ["a"], "dsc": [0.8]}),
            )

        with tempfile.TemporaryDirectory() as directory:
            with (
                patch.object(training, "train_one_fold", side_effect=fake_train),
                patch.object(
                    training, "train_all_data_model", return_value="run-final"
                ),
                redirect_stdout(StringIO()),
            ):
                sweep = training.run_training_sweep(
                    {"unet": TRAINING_MODEL_CONFIGS["unet"]},
                    train_df,
                    experiment=experiment,
                    patch_config=SimpleNamespace(),
                    preprocessing_manifest="manifest.json",
                    results_root=directory,
                )
            selection = json.loads(sweep.inference_run_ids_path.read_text())

        self.assertEqual(selection["schema_version"], 1)
        self.assertEqual(
            selection["models"],
            {
                "unet": {
                    "best": {"fold_2": "run-2", "fold_4": "run-4"},
                    "final": {"all_data": "run-final"},
                }
            },
        )


if __name__ == "__main__":
    unittest.main()
