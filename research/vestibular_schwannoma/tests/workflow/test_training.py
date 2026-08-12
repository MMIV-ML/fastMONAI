import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd
from fastMONAI import utils as fastmonai_utils
from fastMONAI.utils import ModelTrackingCallback

from vestibular_schwannoma.workflow.config import ExperimentConfig
from vestibular_schwannoma.workflow.models import MODEL_RECIPES
from vestibular_schwannoma.workflow import training


class TrainingOrchestrationTests(unittest.TestCase):
    def test_public_failure_hook_is_used(self):
        callback = MagicMock()
        training._mark_failed(callback)
        callback.mark_failed.assert_called_once_with()

    def test_owned_active_mlflow_run_is_failed_idempotently(self):
        callback = object.__new__(ModelTrackingCallback)
        callback._owns_run = True
        callback._auto_started = True
        callback._run_failed = False
        callback.run_id = "owned-run"
        active = SimpleNamespace(info=SimpleNamespace(run_id="owned-run"))
        with (
            patch.object(fastmonai_utils.mlflow, "active_run", return_value=active),
            patch.object(fastmonai_utils.mlflow, "end_run") as end_run,
        ):
            callback.mark_failed()
            callback.mark_failed()
        end_run.assert_called_once_with(status="FAILED")
        self.assertFalse(callback._auto_started)
        self.assertTrue(callback._run_failed)

    def test_owned_closed_mlflow_run_is_updated_without_closing_another_run(self):
        callback = object.__new__(ModelTrackingCallback)
        callback._owns_run = True
        callback._auto_started = False
        callback._run_failed = False
        callback.run_id = "owned-run"
        client = MagicMock()
        with (
            patch.object(fastmonai_utils.mlflow, "active_run", return_value=None),
            patch.object(
                fastmonai_utils.mlflow.tracking,
                "MlflowClient",
                return_value=client,
            ),
            patch.object(fastmonai_utils.mlflow, "end_run") as end_run,
        ):
            callback.mark_failed()
        client.set_terminated.assert_called_once_with("owned-run", status="FAILED")
        end_run.assert_not_called()

    def test_callback_does_not_fail_a_caller_owned_run(self):
        callback = object.__new__(ModelTrackingCallback)
        callback._owns_run = False
        callback._auto_started = False
        callback._run_failed = False
        callback.run_id = "external-run"
        with patch.object(fastmonai_utils.mlflow, "end_run") as end_run:
            callback.mark_failed()
        end_run.assert_not_called()

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
                    MODEL_RECIPES["unet"],
                    1,
                    train_df,
                    experiment=experiment,
                    patch_config=SimpleNamespace(),
                    pre_inference_transforms=[],
                    preprocessing_manifest="manifest.json",
                    results_dir="results/unet",
                )
        dls.close.assert_called_once_with()

    def test_sweep_records_fold_failures_and_continues(self):
        experiment = ExperimentConfig(
            model_keys=("unet",),
            folds=(1, 2),
            epochs=1,
            compile_models=False,
        )
        train_df = pd.DataFrame({"case_id": ["a", "b"], "fold": [1, 2]})

        def fake_train(recipe, fold, *args, **kwargs):
            if fold == 1:
                raise RuntimeError("expected failure")
            return training.FoldRun(
                model_key=recipe.key,
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
                    {"unet": MODEL_RECIPES["unet"]},
                    train_df,
                    experiment=experiment,
                    patch_config=SimpleNamespace(),
                    pre_inference_transforms=[],
                    preprocessing_manifest="manifest.json",
                    results_root=directory,
                )
        self.assertEqual(list(sweep.fold_runs["unet"]), [2])
        self.assertEqual(len(sweep.failures), 1)
        self.assertEqual(sweep.failures[0].fold, 1)
        self.assertEqual(sweep.failures[0].error_type, "RuntimeError")


if __name__ == "__main__":
    unittest.main()
