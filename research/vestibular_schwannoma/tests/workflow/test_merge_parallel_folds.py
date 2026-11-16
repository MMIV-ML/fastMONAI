import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from vestibular_schwannoma.scripts import merge_parallel_folds
from vestibular_schwannoma.workflow.run_selection import (
    make_training_contract,
    merge_fold_run_selections,
)


class MergeParallelFoldsCliTests(unittest.TestCase):
    def test_main_passes_resolved_inputs_to_fixed_fold_merger(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "folds_123" / "inference_run_ids.json"
            second = root / "folds_45" / "inference_run_ids.json"
            output_root = root / "merged"
            destination = output_root / "inference_run_ids.json"
            with (
                patch.object(
                    merge_parallel_folds,
                    "merge_fold_run_selections",
                    return_value=destination,
                ) as merge,
                redirect_stdout(StringIO()),
            ):
                status = merge_parallel_folds.main(
                    [
                        str(first),
                        str(second),
                        "--model",
                        "unet",
                        "--output-root",
                        str(output_root),
                    ]
                )

        self.assertEqual(status, 0)
        merge.assert_called_once_with(
            [first, second],
            model_key="unet",
            output_root=output_root,
        )

    def write_job(self, root, name, folds, result_folds=None):
        """Write one job root: a registry listing ``folds`` and fold results."""

        job = root / name
        for fold in folds if result_folds is None else result_folds:
            fold_dir = job / "unet" / f"fold_{fold}"
            fold_dir.mkdir(parents=True)
            pd.DataFrame(
                {"case_id": [f"case_{fold}"], "dsc": [0.5 + fold / 10]}
            ).to_csv(fold_dir / "results.csv", index=False)
        manifest = {
            "schema_version": 1,
            "manifest_kind": "completed_registry",
            "run_group": name,
            "training_contracts": {"unet": make_training_contract({"campaign": "t"})},
            "models": {
                "unet": {"best": {f"fold_{fold}": f"run-{fold}" for fold in folds}}
            },
        }
        selection = job / "completed_run_ids.json"
        selection.write_text(json.dumps(manifest), encoding="utf-8")
        return selection

    def write_dataset(self, root):
        path = root / "ml_dataset.csv"
        pd.DataFrame(
            {"case_id": [f"case_{fold}" for fold in range(1, 6)], "fold": range(1, 6)}
        ).to_csv(path, index=False)
        return path

    def run_merge(self, root, selections):
        output_root = root / "merged"
        with redirect_stdout(StringIO()) as output:
            status = merge_parallel_folds.main(
                [
                    *map(str, selections),
                    "--model",
                    "unet",
                    "--output-root",
                    str(output_root),
                    "--data-csv",
                    str(self.write_dataset(root)),
                ]
            )
        return status, output_root, output.getvalue()

    def test_merge_writes_selection_and_pooled_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            selections = [
                self.write_job(root, "folds_123", [1, 2, 3]),
                self.write_job(root, "folds_45", [4, 5]),
            ]
            status, output_root, _ = self.run_merge(root, selections)
            selection = (output_root / "inference_run_ids.json").read_bytes()
            direct = merge_fold_run_selections(
                selections, model_key="unet", output_root=root / "direct" / "merged"
            )
            summary = pd.read_csv(output_root / "unet" / "cv_summary.csv")
            comparison = pd.read_csv(
                output_root / "cv_model_comparison.csv", index_col="model"
            )
            self.assertEqual(selection, direct.read_bytes())

        self.assertEqual(status, 0)
        self.assertEqual(
            json.loads(selection)["models"]["unet"]["best"],
            {f"fold_{fold}": f"run-{fold}" for fold in range(1, 6)},
        )
        self.assertEqual(summary.fold.tolist(), [1, 2, 3, 4, 5])
        self.assertEqual(
            summary.case_id.tolist(), [f"case_{fold}" for fold in range(1, 6)]
        )
        self.assertEqual(comparison.loc["unet", "n_cases"], 5)
        self.assertAlmostEqual(comparison.loc["unet", "dsc_mean"], 0.8)

    def test_summary_uses_only_the_folds_each_selection_lists(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            selections = [
                self.write_job(root, "fold_1", [1], result_folds=[1, 2]),
                self.write_job(root, "folds_2345", [2, 3, 4, 5]),
            ]
            stale = root / "fold_1" / "unet" / "fold_2" / "results.csv"
            pd.DataFrame({"case_id": ["stale"], "dsc": [0.0]}).to_csv(
                stale, index=False
            )
            status, output_root, _ = self.run_merge(root, selections)
            self.assertTrue((output_root / "inference_run_ids.json").is_file())
            summary = pd.read_csv(output_root / "unet" / "cv_summary.csv")

        self.assertEqual(status, 0)
        self.assertEqual(
            summary.case_id.tolist(), [f"case_{fold}" for fold in range(1, 6)]
        )

    def test_summary_error_keeps_the_merged_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            selections = [
                self.write_job(root, "folds_123", [1, 2, 3]),
                self.write_job(root, "folds_45", [4, 5], result_folds=[4]),
            ]
            status, output_root, output = self.run_merge(root, selections)
            self.assertTrue((output_root / "inference_run_ids.json").is_file())
            self.assertFalse((output_root / "unet").exists())
            self.assertFalse((output_root / "cv_model_comparison.csv").exists())

        self.assertEqual(status, 0)
        self.assertIn(
            "Pooled summary skipped: Incomplete cross-validation results", output
        )

    def test_fold_listed_by_two_selections_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            selections = [
                self.write_job(root, "folds_123", [1, 2, 3]),
                self.write_job(root, "folds_345", [3, 4, 5]),
            ]
            with self.assertRaisesRegex(ValueError, "Duplicate model member"):
                self.run_merge(root, selections)
            self.assertFalse((root / "merged").exists())


if __name__ == "__main__":
    unittest.main()
