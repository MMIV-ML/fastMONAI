import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastMONAI.vision_all import (
    PatchConfig,
    patch_config_to_dict,
    prediction_filename,
)

from vestibular_schwannoma.workflow import inference
from vestibular_schwannoma.workflow.config import VS_OUTPUT_SPEC
from vestibular_schwannoma.workflow.run_selection import (
    make_training_contract,
    read_inference_run_ids,
)


class InferenceArtifactTests(unittest.TestCase):
    def test_public_prediction_filename_matches_patch_inference_outputs(self):
        self.assertEqual(prediction_filename("case.nii.gz"), "case_pred.nii.gz")
        self.assertEqual(prediction_filename("case.nii"), "case_pred.nii")
        self.assertEqual(prediction_filename("case.mha"), "case_pred.nii.gz")

    def metadata(
        self,
        run_id,
        *,
        role="best",
        patch_size=None,
        keep_largest_component=False,
    ):
        config = PatchConfig(
            patch_size=patch_size or [16, 16, 16],
            target_spacing=[1.0, 1.0, 1.0],
            keep_largest_component=keep_largest_component,
        )
        return {
            "mlflow_run": run_id,
            "artifact_role": role,
            "inference_config": {
                "config_schema": "1",
                "workflow": "patch",
                "patch_config": patch_config_to_dict(config, inference_only=True),
                "output": dict(VS_OUTPUT_SPEC),
            },
        }

    def write_selection(
        self,
        path,
        *,
        model_key="unet",
        role="best",
        run_ids=None,
        contract_payload=None,
        manifest_kind="inference_selection",
        include_contract=True,
    ):
        manifest = {
            "schema_version": 1,
            "manifest_kind": manifest_kind,
            "run_group": "test-group",
            "models": {
                model_key: {
                    role: run_ids if run_ids is not None else {"fold_1": "run-1"}
                },
            },
        }
        if include_contract:
            manifest["training_contracts"] = {
                model_key: make_training_contract(
                    contract_payload or {"campaign": "test"}
                )
            }
        path.write_text(json.dumps(manifest), encoding="utf-8")
        return path

    def test_merges_disjoint_fold_run_selections(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = self.write_selection(
                root / "folds_123.json",
                manifest_kind="completed_registry",
                run_ids={
                    "fold_1": "run-1",
                    "fold_2": "run-2",
                    "fold_3": "run-3",
                },
            )
            second = self.write_selection(
                root / "folds_45.json",
                manifest_kind="completed_registry",
                run_ids={"fold_4": "run-4", "fold_5": "run-5"},
            )
            destination = inference.merge_fold_run_selections(
                [first, second],
                model_key="unet",
                output_root=root / "merged",
            )
            manifest = json.loads(destination.read_text(encoding="utf-8"))

        self.assertEqual(destination.name, "inference_run_ids.json")
        self.assertEqual(manifest["manifest_kind"], "inference_selection")
        self.assertEqual(manifest["run_group"], "merged")
        self.assertEqual(
            manifest["training_contracts"]["unet"]["payload"],
            {"campaign": "test"},
        )
        self.assertEqual(
            manifest["models"]["unet"]["best"],
            {
                "fold_1": "run-1",
                "fold_2": "run-2",
                "fold_3": "run-3",
                "fold_4": "run-4",
                "fold_5": "run-5",
            },
        )

    def test_completed_registry_cannot_be_loaded_directly_for_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            selection = self.write_selection(
                Path(directory) / "completed_run_ids.json",
                manifest_kind="completed_registry",
            )
            with self.assertRaisesRegex(ValueError, "partial registry"):
                inference.load_inference_models(
                    run_selection_file=selection,
                    model_key="unet",
                    artifact_role="best",
                    device="cpu",
                )

    def test_legacy_schema_one_selection_without_contract_remains_readable(self):
        with tempfile.TemporaryDirectory() as directory:
            selection = self.write_selection(
                Path(directory) / "legacy.json",
                manifest_kind="inference_selection",
                include_contract=False,
            )
            manifest = json.loads(selection.read_text())
            manifest.pop("manifest_kind")
            selection.write_text(json.dumps(manifest), encoding="utf-8")

            self.assertEqual(
                read_inference_run_ids(
                    selection,
                    model_key="unet",
                    artifact_role="best",
                ),
                {"fold_1": "run-1"},
            )

    def test_reader_rejects_invalid_or_duplicate_run_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            empty = self.write_selection(
                root / "empty.json",
                run_ids={"fold_1": ""},
            )
            duplicate = self.write_selection(
                root / "duplicate.json",
                run_ids={"fold_1": "same-run", "fold_2": "same-run"},
            )
            with self.assertRaisesRegex(ValueError, "non-empty string"):
                read_inference_run_ids(
                    empty,
                    model_key="unet",
                    artifact_role="best",
                )
            with self.assertRaisesRegex(ValueError, "duplicate run IDs"):
                read_inference_run_ids(
                    duplicate,
                    model_key="unet",
                    artifact_role="best",
                )

    def test_merge_rejects_training_contract_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = self.write_selection(
                root / "first.json",
                run_ids={"fold_1": "run-1"},
            )
            second = self.write_selection(
                root / "second.json",
                run_ids={"fold_2": "run-2"},
                contract_payload={"campaign": "different"},
            )
            with self.assertRaisesRegex(ValueError, "Training contract mismatch"):
                inference.merge_fold_run_selections(
                    [first, second],
                    model_key="unet",
                    output_root=root / "mismatch-output",
                )
            self.assertFalse((root / "mismatch-output").exists())

    def test_merge_rejects_overlapping_or_incomplete_fold_selections(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = self.write_selection(
                root / "first.json",
                run_ids={"fold_1": "run-1"},
            )
            overlap = self.write_selection(
                root / "overlap.json",
                run_ids={"fold_1": "other-run"},
            )
            with self.assertRaisesRegex(ValueError, "Duplicate model member 'fold_1'"):
                inference.merge_fold_run_selections(
                    [first, overlap],
                    model_key="unet",
                    output_root=root / "overlap-output",
                )
            with self.assertRaisesRegex(ValueError, r"missing=\['fold_2'"):
                inference.merge_fold_run_selections(
                    [first],
                    model_key="unet",
                    output_root=root / "incomplete-output",
                )

            self.assertFalse((root / "overlap-output").exists())
            self.assertFalse((root / "incomplete-output").exists())

    def test_merge_rejects_duplicate_run_ids_and_existing_output_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = self.write_selection(
                root / "first.json",
                run_ids={"fold_1": "same-run"},
            )
            second = self.write_selection(
                root / "second.json",
                run_ids={
                    "fold_2": "same-run",
                    "fold_3": "run-3",
                    "fold_4": "run-4",
                    "fold_5": "run-5",
                },
            )
            with self.assertRaisesRegex(ValueError, "duplicate MLflow run IDs"):
                inference.merge_fold_run_selections(
                    [first, second],
                    model_key="unet",
                    output_root=root / "duplicate-output",
                )

            existing = root / "existing"
            existing.mkdir()
            self.write_selection(
                second,
                run_ids={
                    "fold_2": "different-run",
                    "fold_3": "run-3",
                    "fold_4": "run-4",
                    "fold_5": "run-5",
                },
            )
            with self.assertRaisesRegex(FileExistsError, "new --output-root"):
                inference.merge_fold_run_selections(
                    [first, second],
                    model_key="unet",
                    output_root=existing,
                )

    def test_loader_requires_exactly_one_source(self):
        for selection, local in ((None, {}), ("selection.json", {"a": "model"})):
            with (
                self.subTest(selection=selection, local=local),
                self.assertRaises(ValueError),
            ):
                inference.load_inference_models(
                    run_selection_file=selection,
                    model_key="unet",
                    local_model_artifacts=local,
                    artifact_role="best",
                    device="cpu",
                )

    def test_mlflow_source_is_resolved_validated_and_loaded_once(self):
        with tempfile.TemporaryDirectory() as directory:
            selection = self.write_selection(Path(directory) / "selection.json")
            artifact = Path(directory) / "member.safetensors"
            artifact.touch()
            model = object()
            with (
                patch.object(
                    inference,
                    "find_model_artifacts",
                    return_value={"fold_1": artifact},
                ) as find,
                patch.object(
                    inference,
                    "read_safetensors_metadata",
                    return_value=self.metadata("run-1"),
                ),
                patch.object(
                    inference,
                    "load_safetensors_model",
                    return_value=model,
                ) as load,
            ):
                loaded = inference.load_inference_models(
                    run_selection_file=selection,
                    model_key="unet",
                    artifact_role="best",
                    device="cpu",
                )
        find.assert_called_once_with(
            run_ids={"fold_1": "run-1"},
            artifact_role="best",
        )
        load.assert_called_once_with(artifact, device="cpu")
        self.assertIs(loaded.predictor, model)
        self.assertEqual(loaded.artifacts, {"fold_1": artifact})

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
                loaded = inference.load_inference_models(
                    local_model_artifacts=paths,
                    artifact_role="best",
                    device="cpu",
                )
        self.assertEqual(loaded.patch_config.patch_size, [16, 16, 16])
        self.assertEqual(loaded.predictor, loaded_models)
        self.assertEqual(loaded.artifacts, paths)
        self.assertEqual(load.call_count, 2)

    def test_single_predictor_is_inferred_from_one_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "all_data.safetensors"
            path.touch()
            model = object()
            with (
                patch.object(
                    inference,
                    "read_safetensors_metadata",
                    return_value=self.metadata("run-final", role="final"),
                ),
                patch.object(
                    inference,
                    "load_safetensors_model",
                    return_value=model,
                ),
            ):
                loaded = inference.load_inference_models(
                    local_model_artifacts={"all_data": path},
                    artifact_role="final",
                    device="cpu",
                )
        self.assertIs(loaded.predictor, model)

    def test_loading_requires_at_least_one_declared_source(self):
        with self.assertRaisesRegex(ValueError, "Declare exactly one"):
            inference.load_inference_models(
                artifact_role="best",
                device="cpu",
            )

    def test_mlflow_run_and_artifact_role_must_match_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            selection = self.write_selection(Path(directory) / "selection.json")
            artifact = Path(directory) / "member.safetensors"
            artifact.touch()
            cases = (
                (self.metadata("different-run"), "declares MLflow run"),
                (self.metadata("run-1", role="final"), "has role"),
            )
            for metadata, message in cases:
                with (
                    self.subTest(message=message),
                    patch.object(
                        inference,
                        "find_model_artifacts",
                        return_value={"fold_1": artifact},
                    ),
                    patch.object(
                        inference,
                        "read_safetensors_metadata",
                        return_value=metadata,
                    ),
                    patch.object(inference, "load_safetensors_model") as load,
                    self.assertRaisesRegex(ValueError, message),
                ):
                    inference.load_inference_models(
                        run_selection_file=selection,
                        model_key="unet",
                        artifact_role="best",
                        device="cpu",
                    )
                load.assert_not_called()

    def test_selection_requires_supported_schema_and_ready_model_role(self):
        with tempfile.TemporaryDirectory() as directory:
            selection = Path(directory) / "selection.json"
            cases = (
                (
                    {"schema_version": 2, "run_group": "group", "models": {}},
                    "Unsupported inference run selection schema",
                ),
                (
                    {"schema_version": 1, "run_group": "group", "models": {}},
                    "is not ready",
                ),
                (
                    {
                        "schema_version": 1,
                        "run_group": "group",
                        "models": {"unet": {"final": {"all_data": "run"}}},
                    },
                    "Role 'best' is not ready",
                ),
            )
            for manifest, message in cases:
                with self.subTest(message=message):
                    selection.write_text(json.dumps(manifest), encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, message):
                        inference.load_inference_models(
                            run_selection_file=selection,
                            model_key="unet",
                            artifact_role="best",
                            device="cpu",
                        )

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
                    inference.load_inference_models(
                        local_model_artifacts=paths,
                        artifact_role="best",
                        device="cpu",
                    )
        load.assert_not_called()

    def test_rejects_artifacts_that_remove_disconnected_predictions(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fold_1.safetensors"
            path.touch()
            with (
                patch.object(
                    inference,
                    "read_safetensors_metadata",
                    return_value=self.metadata("run-1", keep_largest_component=True),
                ),
                patch.object(inference, "load_safetensors_model") as load,
                self.assertRaisesRegex(
                    ValueError, "must preserve all predicted components"
                ),
            ):
                inference.load_inference_models(
                    local_model_artifacts={"fold_1": path},
                    artifact_role="best",
                    device="cpu",
                )
            load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
