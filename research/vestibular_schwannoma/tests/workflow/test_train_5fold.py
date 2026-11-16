import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from vestibular_schwannoma.scripts import train_5fold
from vestibular_schwannoma.workflow.config import ExperimentConfig


class FiveFoldLauncherTests(unittest.TestCase):
    def test_defaults_select_all_models_and_preserve_80_20_control(self):
        args = train_5fold._parser().parse_args([])
        self.assertEqual(train_5fold.DEFAULT_MODELS, ("unet", "dynunet", "segmamba"))

        self.assertEqual(tuple(args.models), train_5fold.DEFAULT_MODELS)
        self.assertEqual(args.folds, [1, 2, 3, 4, 5])
        self.assertEqual(args.foreground_probability, 0.8)
        self.assertNotIn("use_tta", vars(args))

    def test_70_30_sampling_remains_an_explicit_comparison(self):
        args = train_5fold._parser().parse_args(["--foreground-probability", "0.7"])

        self.assertEqual(args.foreground_probability, 0.7)

    def test_smaller_dynunet_variants_are_selectable(self):
        args = train_5fold._parser().parse_args(
            ["--models", "dynunet_small", "dynunet_xs"]
        )

        self.assertEqual(args.models, ["dynunet_small", "dynunet_xs"])

    def run_main(self, argv, failures=()):
        train_df = pd.DataFrame({"case_id": ["a"], "fold": [2]})
        preprocessing = SimpleNamespace(
            cache_version="cache", reused=True, manifest_path=Path("manifest.json")
        )
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(train_5fold.os, "chdir") as chdir,
            patch.object(train_5fold.torch.cuda, "is_available", return_value=True),
            patch.object(
                train_5fold, "get_training_model_configs", return_value={"unet": None}
            ) as get_configs,
            patch.object(
                train_5fold, "load_training_index", return_value=train_df
            ) as load_index,
            patch.object(
                train_5fold,
                "preprocess_training_data",
                return_value=([], "fp", preprocessing),
            ) as preprocess,
            patch.object(
                train_5fold,
                "run_training_sweep",
                return_value=SimpleNamespace(failures=list(failures)),
            ) as sweep,
            patch.object(train_5fold, "aggregate_results", return_value=None),
            redirect_stdout(StringIO()),
        ):
            results_root = Path(directory) / "run"
            status = train_5fold.main([*argv, "--results-root", str(results_root)])
        chdir.assert_called_once_with(train_5fold.PROJECT_ROOT)
        return status, results_root, get_configs, load_index, preprocess, sweep

    def test_main_maps_arguments_onto_the_experiment(self):
        status, results_root, get_configs, load_index, preprocess, sweep = (
            self.run_main(
                [
                    "--models",
                    "dynunet",
                    "unet",
                    "--folds",
                    "2",
                    "4",
                    "--epochs",
                    "3",
                    "--batch-size",
                    "2",
                    "--learning-rate",
                    "0.01",
                    "--seed",
                    "7",
                    "--foreground-probability",
                    "0.7",
                    "--samples-per-volume",
                    "2",
                    "--queue-workers",
                    "7",
                    "--queue-length",
                    "50",
                    "--preprocess-workers",
                    "3",
                    "--data-csv",
                    "data/other.csv",
                    "--preprocessed-dir",
                    "other_cache",
                    "--no-compile",
                    "--skip-unavailable",
                    "--stop-on-error",
                ]
            )
        )

        self.assertEqual(status, 0)
        experiment = sweep.call_args.kwargs["experiment"]
        self.assertEqual(
            experiment,
            ExperimentConfig(
                model_keys=("dynunet", "unet"),
                folds=(2, 4),
                run_cross_validation=True,
                train_all_data=False,
                training_seed=7,
                epochs=3,
                batch_size=2,
                learning_rate=0.01,
                use_tta=True,
                compile_models=False,
                target_spacing=(0.4102, 0.4102, 1.5),
                patch_size=(256, 256, 48),
                preprocess_workers=3,
                samples_per_volume=2,
                queue_num_workers=7,
                queue_length=50,
                foreground_sampling_probability=0.7,
                continue_on_error=False,
            ),
        )
        get_configs.assert_called_once_with(
            ("dynunet", "unet"), skip_unavailable=True
        )
        load_index.assert_called_once_with(
            train_5fold.PROJECT_ROOT / "data/other.csv", (2, 4)
        )
        train_df = load_index.return_value
        preprocess.assert_called_once_with(
            train_df, experiment, train_5fold.PROJECT_ROOT / "other_cache"
        )
        self.assertEqual(
            sweep.call_args.kwargs["preprocessing_manifest"], Path("manifest.json")
        )
        self.assertEqual(sweep.call_args.kwargs["results_root"], results_root)

    def test_main_returns_one_when_a_fold_fails(self):
        failure = SimpleNamespace(
            model_key="unet", fold=1, error_type="RuntimeError", message="boom"
        )
        status, *_ = self.run_main(["--models", "unet", "--folds", "1"], [failure])
        self.assertEqual(status, 1)


if __name__ == "__main__":
    unittest.main()
