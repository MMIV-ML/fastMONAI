import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, call, patch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACS_DIR = PROJECT_ROOT / "deployment" / "pacs"
sys.path.insert(0, str(PACS_DIR))

import deployment_bundle as runtime_bundle  # noqa: E402


class DeploymentLoadingTests(unittest.TestCase):
    def metadata(self, run_id="run-1", patch_config=None):
        return {
            "artifact_schema": "1",
            "arch_id": "monai.unet",
            "arch_kwargs": {},
            "wrapper_spec": [],
            "artifact_role": "final",
            "mlflow_run": run_id,
            "inference_config": {
                "config_schema": "1",
                "workflow": "patch",
                "patch_config": {} if patch_config is None else patch_config,
                "output": {},
            },
        }

    def write_bundle(
        self,
        script_dir: Path,
        *,
        member_ids=("member_1",),
        create_models=True,
        model_type="unet",
    ) -> tuple[Path, dict]:
        models_dir = script_dir / "model_bundles" / model_type
        models_dir.mkdir(parents=True)
        members = []
        for index, member_id in enumerate(member_ids, start=1):
            if create_models:
                (models_dir / f"{member_id}.safetensors").write_bytes(
                    f"model-{index}".encode()
                )
            members.append({"member_id": member_id, "sha256": f"{index:064x}"})
        deployment = {
            "schema_version": 1,
            "model_type": model_type,
            "members": members,
            "bundle_sha256": "a" * 64,
        }
        config_path = models_dir / "deployment_config.json"
        config_path.write_text(json.dumps(deployment), encoding="utf-8")
        return config_path, deployment

    def test_packaged_manifest_requires_current_schema_and_requested_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path, deployment = self.write_bundle(root)
            for field, value, message in (
                ("schema_version", 2, "unsupported deployment schema"),
                ("model_type", "dynunet", "bundle declares model_type"),
            ):
                with self.subTest(field=field):
                    changed = dict(deployment)
                    changed[field] = value
                    config_path.write_text(json.dumps(changed), encoding="utf-8")
                    with self.assertRaisesRegex(RuntimeError, message):
                        runtime_bundle._read_packaged_deployment(config_path, "unet")

    def test_packaged_manifest_requires_safe_unique_members(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path, deployment = self.write_bundle(root)
            invalid_members = (
                ([], "at least one member"),
                ([{}], "invalid member ID"),
                ([{"member_id": "../escape"}], "invalid member ID"),
                (
                    [{"member_id": "same"}, {"member_id": "same"}],
                    "duplicate member ids",
                ),
            )
            for members, message in invalid_members:
                with self.subTest(members=members):
                    changed = dict(deployment)
                    changed["members"] = members
                    config_path.write_text(json.dumps(changed), encoding="utf-8")
                    with self.assertRaisesRegex(RuntimeError, message):
                        runtime_bundle._read_packaged_deployment(config_path, "unet")

    def test_missing_manifest_and_declared_model_fail_before_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(FileNotFoundError, "no declared"):
                runtime_bundle.load_deployment("unet", script_dir=root)

            self.write_bundle(root, create_models=False)
            with (
                patch.object(runtime_bundle, "read_safetensors_metadata") as metadata,
                patch.object(runtime_bundle, "load_safetensors_model") as load,
                self.assertRaisesRegex(FileNotFoundError, "packaged model files"),
            ):
                runtime_bundle.load_deployment("unet", script_dir=root)
            metadata.assert_not_called()
            load.assert_not_called()

    def test_loads_ensemble_and_uses_first_member_patch_config(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, deployment = self.write_bundle(
                root, member_ids=("fold_1", "fold_2")
            )
            paths = [
                root / "model_bundles" / "unet" / f"fold_{index}.safetensors"
                for index in (1, 2)
            ]
            models = [MagicMock(), MagicMock()]
            with (
                patch.object(
                    runtime_bundle,
                    "read_safetensors_metadata",
                    return_value=self.metadata(patch_config={"patch_size": [16, 16, 16]}),
                ) as metadata,
                patch.object(
                    runtime_bundle,
                    "load_safetensors_model",
                    side_effect=models,
                ) as load,
                patch.object(
                    runtime_bundle, "PatchConfig", return_value="patch-config"
                ) as patch_config,
            ):
                loaded = runtime_bundle.load_deployment("unet", script_dir=root)

            self.assertEqual(loaded["members"], deployment["members"])
            self.assertNotIn("models", loaded)
            self.assertEqual(loaded["predictor"], models)
            self.assertEqual(loaded["patch_config"], "patch-config")
            metadata.assert_called_once_with(paths[0])
            self.assertEqual(
                load.call_args_list,
                [call(paths[0], device="cpu"), call(paths[1], device="cpu")],
            )
            patch_config.assert_called_once_with(patch_size=[16, 16, 16])

    def test_loads_one_final_model_from_derived_filename(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, deployment = self.write_bundle(root, member_ids=("all_data",))
            path = root / "model_bundles" / "unet" / "all_data.safetensors"
            model = MagicMock()
            with (
                patch.object(
                    runtime_bundle,
                    "read_safetensors_metadata",
                    return_value=self.metadata("run-final"),
                ),
                patch.object(
                    runtime_bundle,
                    "load_safetensors_model",
                    return_value=model,
                ) as load,
                patch.object(
                    runtime_bundle, "PatchConfig", return_value="patch-config"
                ),
            ):
                loaded = runtime_bundle.load_deployment("unet", script_dir=root)

            self.assertEqual(loaded["members"], deployment["members"])
            self.assertIs(loaded["predictor"], model)
            load.assert_called_once_with(path, device="cpu")

    def test_unknown_model_type_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown model type"):
            runtime_bundle.load_deployment("unknown")


if __name__ == "__main__":
    unittest.main()
