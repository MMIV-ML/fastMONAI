import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from vestibular_schwannoma.workflow.results import (
    combine_fold_results,
    load_fold_results,
    select_qualitative_cases,
    summarize_metrics,
)


class CrossValidationResultTests(unittest.TestCase):
    def setUp(self):
        self.dataset = pd.DataFrame(
            {"case_id": ["a", "b", "c", "d"], "fold": [1, 1, 2, 2]}
        )
        self.fold_results = {
            1: pd.DataFrame({"case_id": ["a", "b"], "dsc": [0.9, 0.7]}),
            2: pd.DataFrame({"case_id": ["c", "d"], "dsc": [0.8, 0.6]}),
        }

    def test_combines_each_held_out_case_exactly_once(self):
        combined = combine_fold_results(
            self.fold_results, expected_folds=(1, 2), dataset_df=self.dataset
        )
        self.assertEqual(combined.case_id.tolist(), ["a", "b", "c", "d"])
        self.assertEqual(combined.fold.tolist(), [1, 1, 2, 2])

    def test_rejects_incomplete_duplicate_and_mismatched_cases(self):
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            combine_fold_results(
                {1: self.fold_results[1]},
                expected_folds=(1, 2),
                dataset_df=self.dataset,
            )

        duplicate = dict(self.fold_results)
        duplicate[2] = duplicate[2].assign(case_id=["a", "d"])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            combine_fold_results(
                duplicate, expected_folds=(1, 2), dataset_df=self.dataset
            )

        missing = dict(self.fold_results)
        missing[2] = missing[2].assign(case_id=["c", "unexpected"])
        with self.assertRaisesRegex(ValueError, "case mismatch"):
            combine_fold_results(
                missing, expected_folds=(1, 2), dataset_df=self.dataset
            )

    def test_summary_excludes_nonfinite_values(self):
        frame = pd.DataFrame({"dsc": [0.5, 1.0], "hd95_mm": [2.0, np.inf]})
        summary = summarize_metrics(frame, metrics=("dsc", "hd95_mm"))
        hd95 = summary.set_index("metric").loc["hd95_mm"]
        self.assertEqual(hd95["mean"], 2.0)
        self.assertEqual(hd95["n_finite"], 1)
        self.assertEqual(hd95["n_excluded"], 1)

    def test_loads_only_valid_fold_directories(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for fold, frame in self.fold_results.items():
                fold_dir = root / f"fold_{fold}"
                fold_dir.mkdir()
                frame.to_csv(fold_dir / "results.csv", index=False)
            invalid = root / "fold_notes"
            invalid.mkdir()
            self.assertEqual(sorted(load_fold_results(root)), [1, 2])

    def test_qualitative_selection_is_distinct_for_one_case(self):
        selected = select_qualitative_cases(
            pd.DataFrame({"case_id": ["a"], "dsc": [0.8]})
        )
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0][0], "median DSC")


if __name__ == "__main__":
    unittest.main()
