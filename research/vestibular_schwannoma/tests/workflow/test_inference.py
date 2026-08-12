import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastMONAI.vision_all import (
    PatchConfig,
    make_output_spec,
    patch_config_to_dict,
    prediction_filename,
)

from vestibular_schwannoma.workflow import inference


class InferenceArtifactTests(unittest.TestCase):
    def test_public_prediction_filename_matches_patch_inference_outputs(self):
        self.assertEqual(prediction_filename("case.nii.gz"), "case_pred.nii.gz")
        self.assertEqual(prediction_filename("case.nii"), "case_pred.nii")
        self.assertEqual(prediction_filename("case.mha"), "case_pred.nii.gz")

    def metadata(self, run_id, *, role="best", patch_size=None):
        config = PatchConfig(
            patch_size=patch_size or [16, 16, 16],
            target_spacing=[1.0, 1.0, 1.0],
        )
        return {
            "mlflow_run": run_id,
            "artifact_role": role,
            "inference_config": {
                "config_schema": "1",
                "workflow": "patch",
                "patch_config": patch_config_to_dict(config, inference_only=True),
                "output": make_output_spec("multiclass_segmentation", classes=2),
            },
        }

    def test_resolve_requires_exactly_one_source(self):
        for runs, local in (({}, {}), ({"a": "run"}, {"a": "model"})):
            with self.subTest(runs=runs, local=local), self.assertRaises(ValueError):
                inference.resolve_model_artifacts(
                    member_run_ids=runs,
                    local_model_artifacts=local,
                    artifact_role="best",
                )

    def test_mlflow_resolution_reuses_fastmonai_discovery(self):
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "member.safetensors"
            artifact.touch()
            with patch.object(
                inference,
                "find_model_artifacts",
                return_value={"fold_1": artifact},
            ) as find:
                resolved = inference.resolve_model_artifacts(
                    member_run_ids={"fold_1": "run-1"}, artifact_role="best"
                )
        find.assert_called_once_with(
            run_ids={"fold_1": "run-1"},
            artifact_role="best",
            expected_members=["fold_1"],
        )
        self.assertEqual(list(resolved), ["fold_1"])

    def test_loads_compatible_ensemble_after_validating_all_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = {
                member: Path(directory) / f"{member}.safetensors"
                for member in ("fold_1", "fold_2")
            }
            for path in paths.values():
                path.touch()
            metadata = {
                str(paths["fold_1"]): self.metadata("run-1"),
                str(paths["fold_2"]): self.metadata("run-2"),
            }
            loaded_models = [object(), object()]
            with (
                patch.object(
                    inference,
                    "read_safetensors_metadata",
                    side_effect=lambda path: metadata[str(path)],
                ),
                patch.object(
                    inference,
                    "load_safetensors_model",
                    side_effect=loaded_models,
                ) as load,
            ):
                loaded = inference.load_predictor_set(
                    paths,
                    mode="ensemble",
                    artifact_role="best",
                    device="cpu",
                    expected_run_ids={"fold_1": "run-1", "fold_2": "run-2"},
                )
        self.assertEqual(loaded.member_ids, ("fold_1", "fold_2"))
        self.assertEqual(loaded.patch_config.patch_size, [16, 16, 16])
        self.assertEqual(loaded.predictor, loaded_models)
        self.assertEqual(load.call_count, 2)

    def test_rejects_incompatible_ensemble_before_loading_models(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = {
                member: Path(directory) / f"{member}.safetensors"
                for member in ("fold_1", "fold_2")
            }
            for path in paths.values():
                path.touch()
            values = [
                self.metadata("run-1"),
                self.metadata("run-2", patch_size=[32, 32, 32]),
            ]
            with (
                patch.object(
                    inference, "read_safetensors_metadata", side_effect=values
                ),
                patch.object(inference, "load_safetensors_model") as load,
            ):
                with self.assertRaisesRegex(ValueError, "different inference"):
                    inference.load_predictor_set(
                        paths,
                        mode="ensemble",
                        artifact_role="best",
                        device="cpu",
                    )
        load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
