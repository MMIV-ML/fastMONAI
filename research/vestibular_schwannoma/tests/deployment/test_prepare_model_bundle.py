import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACS_DIR = PROJECT_ROOT / "deployment" / "pacs"
sys.path.insert(0, str(PACS_DIR))

import prepare_model_bundle as bundle  # noqa: E402
from deployment_models import (  # noqa: E402
    DICOM_OUTPUT_CODES,
    MODEL_CONFIGS,
    validate_registered_uid_prefix,
)
from vestibular_schwannoma.workflow.config import VS_OUTPUT_SPEC  # noqa: E402


REGISTERED_TEST_PREFIX = "1.2.826.0.1.3680043.10.9999"


class ArtifactSelectionTests(unittest.TestCase):
    def parse_run(self, role, *extra):
        return bundle.make_parser().parse_args([
            "--model-type", "unet",
            "--run", "all_data=run-id",
            "--artifact-role", role,
            *extra,
        ])

    def test_pacs_and_research_expect_the_same_output_contract(self):
        self.assertEqual(
            bundle.make_output_spec("multiclass_segmentation", classes=2),
            VS_OUTPUT_SPEC,
        )

    def metadata(
        self,
        *,
        role="final",
        run_id="run-id",
        arch_id="monai.unet",
        arch_kwargs=None,
        patch_config=None,
    ):
        return {
            "artifact_schema": "1",
            "arch_id": arch_id,
            "arch_kwargs": {} if arch_kwargs is None else arch_kwargs,
            "wrapper_spec": [],
            "artifact_role": role,
            "mlflow_run": run_id,
            "inference_config": {
                "config_schema": "1",
                "workflow": "patch",
                "patch_config": (
                    {"keep_largest_component": False}
                    if patch_config is None else patch_config
                ),
                "output": bundle.make_output_spec(
                    "multiclass_segmentation", classes=2
                ),
            },
        }

    def local_args(self, root: Path, *member_ids: str):
        argv = ["--model-type", "unet"]
        for member_id in member_ids:
            artifact = root / f"source-{member_id}.safetensors"
            artifact.write_bytes(member_id.encode())
            argv.extend(["--artifact", f"{member_id}={artifact}"])
        argv.extend([
            "--artifact-role", "final",
            "--out", str(root / "bundle"),
        ])
        return bundle.make_parser().parse_args(argv)

    def test_role_selects_standard_artifact_path(self):
        self.assertEqual(
            bundle._requested_artifact_path(self.parse_run("final")),
            "model/final_model.safetensors",
        )
        self.assertEqual(
            bundle._requested_artifact_path(self.parse_run("best")),
            "model/best_model.safetensors",
        )

    def test_explicit_artifact_path_overrides_role_default(self):
        args = self.parse_run(
            "final", "--artifact-path", "custom/export.safetensors"
        )
        self.assertEqual(
            bundle._requested_artifact_path(args),
            "custom/export.safetensors",
        )

    def test_artifact_role_is_required(self):
        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                bundle.make_parser().parse_args([
                    "--model-type", "unet",
                    "--run", "all_data=run-id",
                ])

    def test_uid_prefix_is_optional(self):
        default_args = self.parse_run("final")
        registered_args = self.parse_run(
            "final", "--dicom-uid-prefix", REGISTERED_TEST_PREFIX
        )
        self.assertIsNone(default_args.dicom_uid_prefix)
        self.assertEqual(registered_args.dicom_uid_prefix, REGISTERED_TEST_PREFIX)

    def test_registered_prefix_validation(self):
        self.assertEqual(
            validate_registered_uid_prefix(REGISTERED_TEST_PREFIX),
            REGISTERED_TEST_PREFIX,
        )
        invalid = (
            "",
            " 1.2.3",
            "1.2.3.",
            "1.02.3",
            "1.40.3",
            "3.1.2",
            "1.2.alpha",
            "2.25",
            "2.25.123",
            "1.2.840.10008",
            "1.2.840.10008.1",
            "1.2." + "1" * 31,
        )
        for value in invalid:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    validate_registered_uid_prefix(value)

    def test_invalid_prefix_fails_before_download_or_output_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "bundle"
            args = self.parse_run(
                "final",
                "--dicom-uid-prefix", "1.02.3",
                "--out", str(output),
            )
            with patch.object(bundle, "_download_run_artifact") as download:
                with self.assertRaisesRegex(ValueError, "canonical numeric"):
                    bundle.build_bundle(args)
            download.assert_not_called()
            self.assertFalse(output.exists())

    def test_registry_preserves_existing_models_and_unique_codes(self):
        self.assertEqual(
            MODEL_CONFIGS["unet"]["arch_ids"], frozenset({"monai.unet"})
        )
        self.assertEqual(
            MODEL_CONFIGS["dynunet"]["arch_ids"],
            frozenset({"monai.dynunet"}),
        )
        self.assertEqual(MODEL_CONFIGS["unet"]["dicom_model_code"], 1)
        self.assertEqual(MODEL_CONFIGS["dynunet"]["dicom_model_code"], 2)

        model_codes = [
            config["dicom_model_code"] for config in MODEL_CONFIGS.values()
        ]
        self.assertTrue(all(type(code) is int and code > 0 for code in model_codes))
        self.assertEqual(len(model_codes), len(set(model_codes)))
        self.assertEqual(
            DICOM_OUTPUT_CODES,
            {"segmentation_mask": 1, "probability_map": 2},
        )

    def test_wrong_architecture_is_rejected_during_bundle_preparation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self.local_args(root, "all_data")
            with (
                patch.object(
                    bundle,
                    "read_safetensors_metadata",
                    return_value=self.metadata(arch_id="monai.dynunet"),
                ),
                patch.object(bundle, "load_safetensors_model") as strict_load,
                self.assertRaisesRegex(ValueError, "does not match model type"),
            ):
                bundle.build_bundle(args)
            strict_load.assert_not_called()
            self.assertFalse((root / "bundle").exists())

    def test_duplicate_mlflow_runs_are_rejected_during_bundle_preparation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self.local_args(root, "fold_1", "fold_2")
            with (
                patch.object(
                    bundle,
                    "read_safetensors_metadata",
                    side_effect=[
                        self.metadata(run_id="same-run"),
                        self.metadata(run_id="same-run"),
                    ],
                ),
                patch.object(bundle, "load_safetensors_model"),
                self.assertRaisesRegex(ValueError, "declared more than once"),
            ):
                bundle.build_bundle(args)
            self.assertFalse((root / "bundle").exists())

    def test_requested_mlflow_run_mismatch_is_rejected_during_preparation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "downloaded.safetensors"
            artifact.touch()
            args = self.parse_run("final", "--out", str(root / "bundle"))
            with (
                patch.object(
                    bundle, "_download_run_artifact", return_value=artifact
                ),
                patch.object(
                    bundle,
                    "read_safetensors_metadata",
                    return_value=self.metadata(run_id="different-run"),
                ),
                patch.object(bundle, "load_safetensors_model") as strict_load,
                self.assertRaisesRegex(ValueError, "does not match.*requested run"),
            ):
                bundle.build_bundle(args)
            strict_load.assert_not_called()
            self.assertFalse((root / "bundle").exists())

    def test_wrong_output_contract_is_rejected_during_bundle_preparation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self.local_args(root, "all_data")
            metadata = self.metadata()
            metadata["inference_config"]["output"] = bundle.make_output_spec(
                "binary_segmentation", threshold=0.5
            )
            with (
                patch.object(
                    bundle, "read_safetensors_metadata", return_value=metadata
                ),
                patch.object(bundle, "load_safetensors_model") as strict_load,
                self.assertRaisesRegex(ValueError, "two-logit multiclass segmentation"),
            ):
                bundle.build_bundle(args)
            strict_load.assert_not_called()
            self.assertFalse((root / "bundle").exists())

    def test_incompatible_ensemble_is_rejected_during_bundle_preparation(self):
        incompatible = (
            (
                self.metadata(run_id="run-2", arch_kwargs={"channels": 8}),
                "different model specification",
            ),
            (
                self.metadata(
                    run_id="run-2",
                    patch_config={
                        "keep_largest_component": False,
                        "patch_size": [32, 32, 32],
                    },
                ),
                "different inference configuration",
            ),
        )
        for second_metadata, message in incompatible:
            with self.subTest(message=message), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                args = self.local_args(root, "fold_1", "fold_2")
                with (
                    patch.object(
                        bundle,
                        "read_safetensors_metadata",
                        side_effect=[self.metadata(run_id="run-1"), second_metadata],
                    ),
                    patch.object(bundle, "load_safetensors_model"),
                    self.assertRaisesRegex(ValueError, message),
                ):
                    bundle.build_bundle(args)
                self.assertFalse((root / "bundle").exists())

    def test_largest_component_filter_is_rejected_before_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self.local_args(root, "all_data")
            metadata = self.metadata(
                patch_config={"keep_largest_component": True}
            )
            with (
                patch.object(
                    bundle, "read_safetensors_metadata", return_value=metadata
                ),
                patch.object(bundle, "load_safetensors_model") as strict_load,
                self.assertRaisesRegex(ValueError, "keep_largest_component=False"),
            ):
                bundle.build_bundle(args)
            strict_load.assert_not_called()
            self.assertFalse((root / "bundle").exists())

    def test_strict_load_failure_prevents_bundle_publication(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self.local_args(root, "all_data")
            with (
                patch.object(
                    bundle,
                    "read_safetensors_metadata",
                    return_value=self.metadata(),
                ),
                patch.object(
                    bundle,
                    "load_safetensors_model",
                    side_effect=RuntimeError("strict load failed"),
                ),
                self.assertRaisesRegex(RuntimeError, "strict load failed"),
            ):
                bundle.build_bundle(args)
            self.assertFalse((root / "bundle").exists())

    def test_final_run_writes_minimal_schema_one_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "downloaded.safetensors"
            artifact.touch()
            output = root / "bundle"
            output.mkdir()
            output.chmod(0o750)
            args = self.parse_run("final", "--out", str(output))
            with (
                patch.object(
                    bundle,
                    "_download_run_artifact",
                    return_value=artifact,
                ) as download,
                patch.object(
                    bundle,
                    "read_safetensors_metadata",
                    return_value=self.metadata(),
                ),
                patch.object(bundle, "load_safetensors_model") as strict_load,
                redirect_stdout(io.StringIO()),
            ):
                manifest_path = bundle.build_bundle(args)

            download.assert_called_once_with(
                "run-id", "model/final_model.safetensors"
            )
            strict_load.assert_called_once_with(artifact.resolve(), device="cpu")
            declaration = json.loads(manifest_path.read_text())
            self.assertEqual(
                set(declaration),
                {"schema_version", "model_type", "members", "bundle_sha256"},
            )
            self.assertEqual(
                set(declaration["members"][0]),
                {"member_id", "sha256"},
            )
            self.assertEqual(declaration["schema_version"], 1)
            self.assertEqual(declaration["model_type"], "unet")
            self.assertTrue((manifest_path.parent / "all_data.safetensors").is_file())
            self.assertEqual(manifest_path.parent.stat().st_mode & 0o777, 0o750)

    def build_local_manifest(
        self,
        root: Path,
        output_name: str,
        *,
        member_id="all_data",
        model_bytes=b"model",
        prefix=None,
    ) -> dict:
        artifact = root / f"source-{output_name}.safetensors"
        artifact.write_bytes(model_bytes)
        argv = [
            "--model-type", "unet",
            "--artifact", f"{member_id}={artifact}",
            "--artifact-role", "final",
            "--out", str(root / output_name),
        ]
        if prefix is not None:
            argv.extend(["--dicom-uid-prefix", prefix])
        args = bundle.make_parser().parse_args(argv)
        with (
            patch.object(
                bundle,
                "read_safetensors_metadata",
                return_value=self.metadata(),
            ),
            patch.object(bundle, "load_safetensors_model"),
            redirect_stdout(io.StringIO()),
        ):
            manifest_path = bundle.build_bundle(args)
        return json.loads(manifest_path.read_text())

    def test_prefix_and_member_rename_do_not_change_bundle_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = self.build_local_manifest(root, "original")
            renamed = self.build_local_manifest(
                root,
                "renamed",
                member_id="renamed_member",
                prefix=REGISTERED_TEST_PREFIX,
            )

        self.assertNotIn("registered_prefix", original)
        self.assertEqual(renamed["registered_prefix"], REGISTERED_TEST_PREFIX)
        self.assertEqual(original["members"][0]["member_id"], "all_data")
        self.assertEqual(renamed["members"][0]["member_id"], "renamed_member")
        self.assertEqual(original["bundle_sha256"], renamed["bundle_sha256"])

    def test_model_bytes_change_bundle_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = self.build_local_manifest(root, "first", model_bytes=b"first")
            second = self.build_local_manifest(root, "second", model_bytes=b"second")
        self.assertNotEqual(first["bundle_sha256"], second["bundle_sha256"])

    def test_bundle_identity_changes_with_model_type_and_member_order(self):
        hashes = ["a" * 64, "b" * 64]
        original = bundle._bundle_sha256(1, "unet", hashes)
        self.assertNotEqual(original, bundle._bundle_sha256(1, "dynunet", hashes))
        self.assertNotEqual(original, bundle._bundle_sha256(1, "unet", hashes[::-1]))

    def test_copy_failure_leaves_no_partial_bundle_or_staging_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sources = []
            for member in ("fold_1", "fold_2"):
                artifact = root / f"{member}.safetensors"
                artifact.write_bytes(member.encode())
                sources.extend(["--artifact", f"{member}={artifact}"])
            output = root / "bundle"
            args = bundle.make_parser().parse_args([
                "--model-type", "unet",
                *sources,
                "--artifact-role", "final",
                "--out", str(output),
            ])
            real_copy = bundle.shutil.copy2
            copy_count = 0

            def fail_second_copy(source, destination):
                nonlocal copy_count
                copy_count += 1
                if copy_count == 2:
                    raise OSError("copy failed")
                return real_copy(source, destination)

            with (
                patch.object(
                    bundle,
                    "read_safetensors_metadata",
                    side_effect=[
                        self.metadata(run_id="run-1"),
                        self.metadata(run_id="run-2"),
                    ],
                ),
                patch.object(bundle, "load_safetensors_model"),
                patch.object(bundle.shutil, "copy2", side_effect=fail_second_copy),
                self.assertRaisesRegex(OSError, "copy failed"),
            ):
                bundle.build_bundle(args)

            self.assertFalse(output.exists())
            self.assertEqual(list(root.glob(".bundle-build-*")), [])

    def test_manifest_failure_leaves_preexisting_empty_target_empty(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "model.safetensors"
            artifact.write_bytes(b"model")
            output = root / "bundle"
            output.mkdir()
            args = bundle.make_parser().parse_args([
                "--model-type", "unet",
                "--artifact", f"all_data={artifact}",
                "--artifact-role", "final",
                "--out", str(output),
            ])
            with (
                patch.object(
                    bundle,
                    "read_safetensors_metadata",
                    return_value=self.metadata(),
                ),
                patch.object(bundle, "load_safetensors_model"),
                patch.object(Path, "write_text", side_effect=OSError("write failed")),
                self.assertRaisesRegex(OSError, "write failed"),
            ):
                bundle.build_bundle(args)

            self.assertTrue(output.is_dir())
            self.assertEqual(list(output.iterdir()), [])
            self.assertEqual(list(root.glob(".bundle-build-*")), [])

    def test_metadata_role_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "model.safetensors"
            artifact.touch()
            args = bundle.make_parser().parse_args([
                "--model-type", "unet",
                "--artifact", f"all_data={artifact}",
                "--artifact-role", "final",
                "--out", str(root / "bundle"),
            ])
            with patch.object(
                bundle,
                "read_safetensors_metadata",
                return_value=self.metadata(role="best"),
            ):
                with self.assertRaisesRegex(
                    ValueError, "has role .best..*expected .final."
                ):
                    bundle.build_bundle(args)


if __name__ == "__main__":
    unittest.main()
