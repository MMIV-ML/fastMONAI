import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import nibabel as nib
import numpy as np
import torch
from fastMONAI.vision_all import (
    PatchConfig,
    PatchInferenceEngine,
    make_model_spec,
    make_output_spec,
    patch_config_to_dict,
    save_safetensors_model,
)
from monai.networks.nets import UNet

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACS_DIR = PROJECT_ROOT / "deployment" / "pacs"
sys.path.insert(0, str(PACS_DIR))

import pacs_inference as pacs  # noqa: E402
import parallel_inference as parallel  # noqa: E402


class CpuAssignmentTests(unittest.TestCase):
    def test_assigns_disjoint_three_thread_worker_sets(self):
        assignments = parallel.assign_worker_cpus(range(16), worker_count=5)
        self.assertEqual(
            assignments, [(0, 1, 2), (3, 4, 5), (6, 7, 8), (9, 10, 11), (12, 13, 14)]
        )
        self.assertEqual(len(set().union(*map(set, assignments))), 15)

    def test_rejects_insufficient_cpu_affinity(self):
        with self.assertRaisesRegex(RuntimeError, "requires at least 15"):
            parallel.assign_worker_cpus(range(14), worker_count=5)


class ParallelEnsembleTests(unittest.TestCase):
    @unittest.skipUnless(
        hasattr(os, "sched_getaffinity") and len(os.sched_getaffinity(0)) >= 2,
        "Real parallel workers require Linux and two available CPUs",
    )
    def test_real_workers_match_sequential_inference_with_and_without_tta(self):
        previous_threads = torch.get_num_threads()
        self.addCleanup(torch.set_num_threads, previous_threads)
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory, torch.random.fork_rng():
            torch.manual_seed(42)
            root = Path(directory)
            kwargs = dict(
                spatial_dims=3,
                in_channels=1,
                out_channels=2,
                channels=(2, 4),
                strides=(2,),
            )
            models = [UNet(**kwargs).eval() for _ in range(2)]
            config = PatchConfig(
                patch_size=[16, 16, 16],
                patch_overlap=8,
                apply_reorder=False,
                normalization=None,
                aggregation_mode="average",
            )
            metadata = dict(
                config_schema="1",
                workflow="patch",
                patch_config=patch_config_to_dict(config, inference_only=True),
                output=make_output_spec("multiclass_segmentation", classes=2),
            )
            image = root / "synthetic.nii.gz"
            values = (
                np.random.default_rng(42).normal(size=(18, 17, 13)).astype(np.float32)
            )
            nib.save(nib.Nifti1Image(values, np.eye(4)), image)
            paths = [
                save_safetensors_model(
                    model,
                    root / f"fold_{i}.safetensors",
                    make_model_spec("monai.unet", kwargs),
                    metadata,
                    "best",
                )
                for i, model in enumerate(models)
            ]
            deployment = dict(
                model_paths=paths,
                output_channels=2,
                patch_config=config,
                members=[{"member_id": f"fold_{i}"} for i in range(2)],
            )
            # Small CI runners use one thread per model; production allocation is
            # covered by CpuAssignmentTests above.
            with patch.object(pacs, "THREADS_PER_ENSEMBLE_MODEL", 1):
                engine = pacs._create_inference_engine(deployment)
            sequential = PatchInferenceEngine(models, config)
            for use_tta in (False, True):
                with self.subTest(tta=use_tta):
                    actual_mask, actual = engine.predict_mask_and_probabilities(
                        image, tta=use_tta
                    )
                    expected_mask, expected = sequential.predict_mask_and_probabilities(
                        image, tta=use_tta
                    )
                    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
                    torch.testing.assert_close(
                        actual_mask, expected_mask, rtol=0, atol=0
                    )


if __name__ == "__main__":
    unittest.main()
