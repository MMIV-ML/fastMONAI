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

import prepare_model_bundle as bundle
from deployment_models import (
    DICOM_DEPLOYMENT_CODES,
    DICOM_OUTPUT_CODES,
    MODEL_ARCH_IDS,
    MODEL_CONFIGS,
    make_dicom_uid_contract,
)


class ArtifactSelectionTests(unittest.TestCase):
    def parse_run(self, role, *extra):
        return bundle.make_parser().parse_args([
            "--mode", "single",
            "--model-type", "unet",
            "--run", "all_data=run-id",
            "--artifact-role", role,
            *extra,
        ])

    def test_final_role_selects_final_model(self):
        args = self.parse_run("final")
        self.assertEqual(
            bundle._requested_artifact_path(args),
            "model/final_model.safetensors",
        )

    def test_best_role_selects_best_model(self):
        args = self.parse_run("best")
        self.assertEqual(
            bundle._requested_artifact_path(args),
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
                    "--mode", "single",
                    "--model-type", "unet",
                    "--run", "all_data=run-id",
                ])

    def test_builder_and_runtime_registry_support_unet_and_dynunet(self):
        self.assertEqual(set(MODEL_ARCH_IDS), {"unet", "dynunet"})
        self.assertEqual(MODEL_ARCH_IDS.keys(), MODEL_CONFIGS.keys())
        args = bundle.make_parser().parse_args([
            "--mode", "ensemble",
            "--model-type", "dynunet",
            "--run", "fold_1=run-1",
            "--run", "fold_2=run-2",
            "--run", "fold_3=run-3",
            "--run", "fold_4=run-4",
            "--run", "fold_5=run-5",
            "--artifact-role", "best",
        ])
        self.assertEqual(args.model_type, "dynunet")
        self.assertEqual(len(bundle._declared_sources(args)), 5)
        self.assertEqual(
            sorted(config["dicom_model_code"] for config in MODEL_CONFIGS.values()),
            [1, 2],
        )
        self.assertEqual(DICOM_DEPLOYMENT_CODES, {"single": 1, "ensemble": 2})
        self.assertEqual(DICOM_OUTPUT_CODES, {"segmentation": 1, "probability": 2})

    def test_final_run_passes_derived_path_to_downloader(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "downloaded.safetensors"
            artifact.touch()
            args = self.parse_run("final", "--out", str(root / "bundle"))
            metadata = {
                "artifact_schema": "1",
                "arch_id": "monai.unet",
                "arch_kwargs": {},
                "wrapper_spec": [],
                "artifact_role": "final",
                "mlflow_run": "run-id",
                "inference_config": {
                    "config_schema": "1",
                    "workflow": "patch",
                    "patch_config": {},
                    "output": bundle.make_output_spec(
                        "multiclass_segmentation", classes=2
                    ),
                },
            }
            with (
                patch.object(
                    bundle,
                    "_download_run_artifact",
                    return_value=artifact,
                ) as download,
                patch.object(
                    bundle,
                    "read_safetensors_metadata",
                    return_value=metadata,
                ),
                patch.object(bundle, "load_safetensors_model"),
            ):
                with redirect_stdout(io.StringIO()):
                    manifest = bundle.build_bundle(args)

            download.assert_called_once_with(
                "run-id", "model/final_model.safetensors"
            )
            self.assertTrue(manifest.is_file())
            self.assertTrue((root / "bundle" / "all_data.safetensors").is_file())
            declaration = json.loads(manifest.read_text())
            self.assertEqual(declaration["schema_version"], 2)
            self.assertEqual(
                declaration["dicom_uid"],
                make_dicom_uid_contract("unet", "single", 1),
            )

    def test_metadata_role_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "model.safetensors"
            artifact.touch()
            args = bundle.make_parser().parse_args([
                "--mode", "single",
                "--model-type", "unet",
                "--artifact", f"all_data={artifact}",
                "--artifact-role", "final",
                "--out", str(root / "bundle"),
            ])
            metadata = {
                "arch_id": "monai.unet",
                "artifact_role": "best",
            }
            with patch.object(
                bundle, "read_safetensors_metadata", return_value=metadata
            ):
                with self.assertRaisesRegex(
                    ValueError, "has role 'best'.*expected 'final'"
                ):
                    bundle.build_bundle(args)


if __name__ == "__main__":
    unittest.main()
