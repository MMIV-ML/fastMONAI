import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from vestibular_schwannoma import merge_parallel_folds


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

    def test_custom_fold_set_is_rejected(self):
        with (
            redirect_stderr(StringIO()),
            self.assertRaises(SystemExit),
        ):
            merge_parallel_folds._parser().parse_args(
                [
                    "fold_1/completed_run_ids.json",
                    "--model",
                    "unet",
                    "--folds",
                    "1",
                    "--output-root",
                    "merged",
                ]
            )


if __name__ == "__main__":
    unittest.main()
