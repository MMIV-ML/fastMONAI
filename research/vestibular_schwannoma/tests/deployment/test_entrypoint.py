import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENTRYPOINT = PROJECT_ROOT / "deployment" / "pacs" / "entrypoint.sh"


class EntrypointTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.output = self.root / "output"
        self.output.mkdir()
        self.capture = self.root / "python_args.txt"

        self.entrypoint = self.root / "entrypoint.sh"
        script = ENTRYPOINT.read_text().replace(
            'output="/output"', f'output="{self.output}"', 1
        )
        self.entrypoint.write_text(script)

        (self.root / "stub_inference.py").write_text("# test stub\n")
        bundle = self.root / "model_bundles" / "unet"
        bundle.mkdir(parents=True)
        (bundle / "deployment_config.json").write_text(
            json.dumps({
                "mode": "single",
                "model_type": "unet",
                "expected_member_count": 1,
            })
        )

        fake_bin = self.root / "bin"
        fake_bin.mkdir()
        conda = fake_bin / "conda"
        conda.write_text("#!/bin/bash\nexit 0\n")
        conda.chmod(0o755)
        python = fake_bin / "python"
        python.write_text(
            "#!/bin/bash\nprintf '%s\\n' \"$@\" > \"$CAPTURE_PATH\"\n"
        )
        python.chmod(0o755)
        self.fake_bin = fake_bin

    def tearDown(self):
        self.tempdir.cleanup()

    def run_entrypoint(self, options):
        env = os.environ.copy()
        env.update(
            {
                "CONDA_DEFAULT_ENV": "test",
                "ROR_CONT_OPTIONS": options,
                "CAPTURE_PATH": str(self.capture),
                "PATH": f"{self.fake_bin}:{env['PATH']}",
            }
        )
        return subprocess.run(
            ["bash", str(self.entrypoint)],
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )

    def captured_args(self):
        return self.capture.read_text().splitlines()

    def test_valid_options_preserve_python_arguments(self):
        result = self.run_entrypoint(
            json.dumps(
                {
                    "model-type": "unet",
                    "tta": False,
                }
            )
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            self.captured_args(),
            [
                str(self.root / "stub_inference.py"),
                "/data",
                str(self.output),
                "--model-type",
                "unet",
            ],
        )

    def test_shell_metacharacters_are_rejected_without_execution(self):
        marker = self.root / "injected"
        result = self.run_entrypoint(
            json.dumps({"model-type": f"unet;touch {marker}"})
        )
        self.assertEqual(result.returncode, 2)
        self.assertFalse(marker.exists())
        self.assertFalse(self.capture.exists())

    def test_dynunet_is_forwarded_as_one_argument_for_a_dynunet_bundle(self):
        deployment = (
            self.root / "model_bundles" / "dynunet" / "deployment_config.json"
        )
        deployment.parent.mkdir(parents=True)
        deployment.write_text(json.dumps({
            "mode": "ensemble",
            "model_type": "dynunet",
            "expected_member_count": 5,
        }))
        result = self.run_entrypoint(json.dumps({"model-type": "dynunet"}))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            self.captured_args(),
            [
                str(self.root / "stub_inference.py"),
                "/data",
                str(self.output),
                "--model-type",
                "dynunet",
                "--tta",
            ],
        )

    def test_requested_model_type_must_match_bundle(self):
        deployment = (
            self.root / "model_bundles" / "dynunet" / "deployment_config.json"
        )
        deployment.parent.mkdir(parents=True)
        deployment.write_text(json.dumps({
            "mode": "ensemble",
            "model_type": "unet",
            "expected_member_count": 5,
        }))
        result = self.run_entrypoint(json.dumps({"model-type": "dynunet"}))
        self.assertEqual(result.returncode, 1)
        self.assertIn(
            "bundle declares model-type unet, requested dynunet", result.stderr
        )
        self.assertFalse(self.capture.exists())

    def test_unknown_option_is_rejected(self):
        result = self.run_entrypoint(json.dumps({"model-tpye": "unet"}))
        self.assertEqual(result.returncode, 2)
        self.assertIn("invalid ROR_CONT_OPTIONS", result.stderr)

    def test_malformed_json_is_rejected(self):
        result = self.run_entrypoint('{"model-type":')
        self.assertEqual(result.returncode, 2)
        self.assertIn("invalid ROR_CONT_OPTIONS", result.stderr)

    def test_invalid_boolean_is_rejected(self):
        result = self.run_entrypoint(json.dumps({"tta": "sometimes"}))
        self.assertEqual(result.returncode, 2)
        self.assertIn("invalid ROR_CONT_OPTIONS", result.stderr)

    def test_command_is_not_reinterpreted(self):
        script = ENTRYPOINT.read_text()
        self.assertNotIn("eval ", script)
        self.assertNotIn("bash -c", script)
        self.assertIn('"' + '$' + '{cmd[@]}"', script)


if __name__ == "__main__":
    unittest.main()
