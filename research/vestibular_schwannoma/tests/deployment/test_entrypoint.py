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
        self.input_dir = self.root / "data" / "input"
        self.input_dir.mkdir(parents=True)
        self.capture = self.root / "conda_args.txt"

        self.entrypoint = self.root / "entrypoint.sh"
        script = (
            ENTRYPOINT.read_text()
            .replace(
                'readonly INPUT_DIR="/data/input"',
                f'readonly INPUT_DIR="{self.input_dir}"',
                1,
            )
            .replace(
                'readonly OUTPUT_DIR="/output"',
                f'readonly OUTPUT_DIR="{self.output}"',
                1,
            )
        )
        self.entrypoint.write_text(script)
        (self.root / "pacs_inference.py").write_text("# test runner\n")

        fake_bin = self.root / "bin"
        fake_bin.mkdir()
        conda = fake_bin / "conda"
        conda.write_text(
            "#!/bin/bash\nprintf '%s\\n' \"$@\" > \"$CAPTURE_PATH\"\n"
        )
        conda.chmod(0o755)
        self.fake_bin = fake_bin

    def tearDown(self):
        self.tempdir.cleanup()

    def run_entrypoint(self, options=None):
        env = os.environ.copy()
        env.update(
            {
                "CAPTURE_PATH": str(self.capture),
                "PATH": f"{self.fake_bin}:{env['PATH']}",
            }
        )
        if options is None:
            env.pop("ROR_CONT_OPTIONS", None)
        else:
            env["ROR_CONT_OPTIONS"] = options
        return subprocess.run(
            ["bash", str(self.entrypoint)],
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )

    def captured_args(self):
        return self.capture.read_text().splitlines()

    def expected_args(self, model_type="unet", *, tta=True):
        args = [
            "run",
            "--no-capture-output",
            "-n",
            "fastmonai",
            "python",
            str(self.root / "pacs_inference.py"),
            str(self.input_dir),
            str(self.output),
            "--model-type",
            model_type,
        ]
        args.append("--tta" if tta else "--no-tta")
        return args

    def test_valid_options_are_forwarded_as_arguments(self):
        result = self.run_entrypoint(
            json.dumps({"model-type": "dynunet", "tta": False})
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            self.captured_args(),
            self.expected_args("dynunet", tta=False),
        )

    def test_defaults_to_unet_with_tta(self):
        result = self.run_entrypoint()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.captured_args(), self.expected_args())

    def test_tta_can_be_enabled_explicitly(self):
        result = self.run_entrypoint(json.dumps({"tta": True}))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            self.captured_args(), self.expected_args(tta=True)
        )

    def test_shell_metacharacters_are_rejected_without_execution(self):
        marker = self.root / "injected"
        result = self.run_entrypoint(
            json.dumps({"model-type": f"unet;touch {marker}"})
        )
        self.assertEqual(result.returncode, 2)
        self.assertFalse(marker.exists())
        self.assertFalse(self.capture.exists())

    def test_missing_input_directory_is_rejected(self):
        self.input_dir.rmdir()
        result = self.run_entrypoint(json.dumps({"model-type": "unet"}))
        self.assertEqual(result.returncode, 2)
        self.assertIn("expected an input DICOM directory", result.stderr)
        self.assertFalse(self.capture.exists())

    def test_missing_inference_script_is_rejected(self):
        (self.root / "pacs_inference.py").unlink()
        result = self.run_entrypoint(json.dumps({"model-type": "unet"}))
        self.assertEqual(result.returncode, 1)
        self.assertIn("bundled inference script is missing", result.stderr)
        self.assertFalse(self.capture.exists())

    def test_unknown_option_is_rejected(self):
        result = self.run_entrypoint(json.dumps({"model-tpye": "unet"}))
        self.assertEqual(result.returncode, 2)
        self.assertIn("invalid ROR_CONT_OPTIONS", result.stderr)
        self.assertFalse(self.capture.exists())

    def test_malformed_json_is_rejected(self):
        result = self.run_entrypoint('{"model-type":')
        self.assertEqual(result.returncode, 2)
        self.assertIn("invalid ROR_CONT_OPTIONS", result.stderr)
        self.assertFalse(self.capture.exists())

    def test_only_json_boolean_tta_is_accepted(self):
        for value in ("false", 0, 1, None, [], {}):
            with self.subTest(value=value):
                result = self.run_entrypoint(json.dumps({"tta": value}))
                self.assertEqual(result.returncode, 2)
                self.assertIn("invalid ROR_CONT_OPTIONS", result.stderr)
                self.assertFalse(self.capture.exists())


if __name__ == "__main__":
    unittest.main()
