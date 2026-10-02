import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACS_DIR = PROJECT_ROOT / "deployment" / "pacs"
sys.path.insert(0, str(PACS_DIR))

import pacs_inference as pacs  # noqa: E402
import parallel_inference as parallel  # noqa: E402


class CpuAssignmentTests(unittest.TestCase):
    def test_assigns_disjoint_three_thread_worker_sets(self):
        assignments = parallel.assign_worker_cpus(range(16), worker_count=5)

        self.assertEqual(
            assignments,
            [
                (0, 1, 2),
                (3, 4, 5),
                (6, 7, 8),
                (9, 10, 11),
                (12, 13, 14),
            ],
        )
        self.assertEqual(len(set().union(*map(set, assignments))), 15)

    def test_rejects_insufficient_cpu_affinity(self):
        with self.assertRaisesRegex(RuntimeError, "requires at least 15"):
            parallel.assign_worker_cpus(range(14), worker_count=5)

    def test_probability_activation_matches_output_channels(self):
        binary = torch.tensor([[[[[0.0]]]]])
        multiclass = torch.tensor([[[[[1.0]]], [[[2.0]]]]])

        self.assertTrue(
            torch.equal(
                parallel._logits_to_probabilities(binary),
                torch.sigmoid(binary),
            )
        )
        self.assertTrue(
            torch.equal(
                parallel._logits_to_probabilities(multiclass),
                torch.softmax(multiclass, dim=1),
            )
        )


class EngineSelectionTests(unittest.TestCase):
    def test_packaged_ensemble_uses_parallel_engine_with_three_threads(self):
        deployment = {
            "members": [{"member_id": f"fold_{index}"} for index in range(1, 6)],
            "model_paths": [Path(f"fold_{index}.safetensors") for index in range(1, 6)],
            "patch_config": SimpleNamespace(),
            "output_channels": 2,
        }
        engine = object()
        with patch.object(
            pacs, "ParallelEnsemblePatchInferenceEngine", return_value=engine
        ) as parallel_engine:
            actual = pacs._create_inference_engine(deployment)

        self.assertIs(actual, engine)
        parallel_engine.assert_called_once_with(
            deployment["model_paths"],
            [f"fold_{index}" for index in range(1, 6)],
            deployment["patch_config"],
            output_channels=2,
            threads_per_model=3,
            sw_batch_size=1,
        )

    def test_injected_in_memory_predictor_uses_standard_engine(self):
        predictor = object()
        deployment = {
            "members": [{"member_id": "test"}],
            "patch_config": SimpleNamespace(),
            "predictor": predictor,
        }
        engine = object()
        with patch.object(
            pacs, "PatchInferenceEngine", return_value=engine
        ) as standard:
            actual = pacs._create_inference_engine(deployment)

        self.assertIs(actual, engine)
        standard.assert_called_once_with(
            predictor,
            deployment["patch_config"],
            sw_batch_size=1,
        )


if __name__ == "__main__":
    unittest.main()
