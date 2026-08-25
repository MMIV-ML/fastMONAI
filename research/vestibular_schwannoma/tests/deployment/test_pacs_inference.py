import io
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACS_DIR = PROJECT_ROOT / "deployment" / "pacs"
sys.path.insert(0, str(PACS_DIR))

import pacs_inference as pacs  # noqa: E402


class CommandLineTests(unittest.TestCase):
    def test_main_forwards_arguments_and_image_version(self):
        argv = [
            "/data/input",
            "/output",
            "--model-type",
            "unet",
            "--tta",
        ]
        with (
            patch.dict(os.environ, {"VERSION": "20260817T120000Z"}),
            patch.object(pacs, "run_inference") as run_inference,
        ):
            pacs.main(argv)

        run_inference.assert_called_once_with(
            "/data/input",
            "/output",
            "unet",
            use_tta=True,
            version="20260817T120000Z",
        )

    def test_main_defaults_to_tta_on(self):
        argv = ["/data/input", "/output"]
        with (
            patch.dict(os.environ, {"VERSION": "20260817T120000Z"}),
            patch.object(pacs, "run_inference") as run_inference,
        ):
            pacs.main(argv)

        run_inference.assert_called_once_with(
            "/data/input",
            "/output",
            "unet",
            use_tta=True,
            version="20260817T120000Z",
        )

    def test_main_allows_tta_to_be_disabled(self):
        argv = [
            "/data/input",
            "/output",
            "--no-tta",
        ]
        with (
            patch.dict(os.environ, {"VERSION": "20260817T120000Z"}),
            patch.object(pacs, "run_inference") as run_inference,
        ):
            pacs.main(argv)

        run_inference.assert_called_once_with(
            "/data/input",
            "/output",
            "unet",
            use_tta=False,
            version="20260817T120000Z",
        )


class PredictionOutputTests(unittest.TestCase):
    def deployment(self):
        return {
            "patch_config": SimpleNamespace(patch_size=[16, 16, 16]),
            "model_type": "unet",
            "bundle_sha256": "a" * 64,
            "predictor": object(),
            "members": [{"member_id": "all_data"}],
        }

    def test_inference_writes_probability_channel_one_then_postprocesses(self):
        mask = torch.zeros((1, 3, 4, 5), dtype=torch.long)
        mask[0, 1, 1, 1] = 1
        probabilities = torch.zeros((2, 3, 4, 5), dtype=torch.float32)
        probabilities[0] = 0.1
        probabilities[1] = 0.9
        engine = MagicMock()
        engine.predict_mask_and_probabilities.return_value = (mask, probabilities)
        deployment = self.deployment()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            work_dir = root / "work"
            input_output_dir = root / "case"
            input_output_dir.mkdir()
            for name in ("source.dcm", "descr.json", "ror-state.txt"):
                (input_output_dir / name).write_text(name)
            with (
                patch.object(pacs, "load_deployment", return_value=deployment),
                patch.object(pacs, "PatchInferenceEngine", return_value=engine),
                patch.object(pacs, "write_prediction_outputs") as write_outputs,
                patch.object(pacs, "validate_dicom_input") as preflight,
                patch.object(
                    pacs,
                    "_required_pr2mask_tools",
                    return_value={"report": Path("report"), "fused": Path("fused")},
                ),
                patch.object(pacs, "_run_postprocessing") as postprocess,
                redirect_stdout(io.StringIO()),
            ):
                pacs.run_inference(
                    input_output_dir,
                    input_output_dir,
                    "unet",
                    use_tta=True,
                    version="release",
                    work_dir=work_dir,
                )
            unrelated_files_remain = all(
                (input_output_dir / name).is_file()
                for name in ("source.dcm", "descr.json", "ror-state.txt")
            )

        engine.predict_mask_and_probabilities.assert_called_once_with(
            str(input_output_dir), tta=True
        )
        preflight.assert_called_once_with(input_output_dir)
        written_mask, written_probability = write_outputs.call_args.args[:2]
        self.assertTrue(torch.equal(written_mask, mask))
        self.assertTrue(torch.equal(written_probability, probabilities[1]))
        self.assertEqual(write_outputs.call_args.args[2:], (
            input_output_dir,
            work_dir,
            deployment,
        ))
        self.assertTrue(write_outputs.call_args.kwargs["use_tta"])
        postprocess.assert_called_once()
        self.assertTrue(unrelated_files_remain)

    def test_required_pr2mask_tools_fail_before_model_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(pacs, "load_deployment") as load:
                with self.assertRaisesRegex(RuntimeError, "pr2mask tools"):
                    pacs.run_inference(
                        "/dicom",
                        "/output",
                        "unet",
                        version="release",
                        pr2mask_dir=Path(directory),
                    )
            load.assert_not_called()

    def test_each_owned_output_collision_is_rejected_before_model_loading(self):
        for owned_name in pacs.FINAL_OUTPUT_DIRS:
            with self.subTest(owned_name=owned_name):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    output = root / "output"
                    (output / owned_name).mkdir(parents=True)
                    with (
                        patch.object(pacs, "_required_pr2mask_tools", return_value={}),
                        patch.object(pacs, "validate_dicom_input") as preflight,
                        patch.object(pacs, "load_deployment") as load,
                    ):
                        with self.assertRaisesRegex(
                            RuntimeError, "owned output directories already exist"
                        ):
                            pacs.run_inference(
                                "/dicom",
                                output,
                                "unet",
                                version="release",
                                work_dir=root / "work",
                            )
                    preflight.assert_not_called()
                    load.assert_not_called()

    def test_broken_symlink_at_owned_output_name_is_a_collision(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "output"
            output.mkdir()
            (output / "mask").symlink_to(output / "missing-target")
            with self.assertRaisesRegex(
                RuntimeError, "owned output directories already exist"
            ):
                pacs._prepare_output_directory(output)

    def test_nonempty_work_directory_is_rejected_before_model_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "output"
            output.mkdir()
            work = root / "work"
            work.mkdir()
            (work / "stale.dcm").touch()
            with (
                patch.object(pacs, "_required_pr2mask_tools", return_value={}),
                patch.object(pacs, "load_deployment") as load,
            ):
                with self.assertRaisesRegex(RuntimeError, "work directory must be empty"):
                    pacs.run_inference(
                        "/dicom",
                        output,
                        "unet",
                        version="release",
                        work_dir=work,
                    )
            load.assert_not_called()

    def test_dicom_preflight_runs_before_model_loading(self):
        events = []
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with (
                patch.object(pacs, "_required_pr2mask_tools", return_value={}),
                patch.object(
                    pacs,
                    "validate_dicom_input",
                    side_effect=lambda path: events.append(("preflight", path)),
                ),
                patch.object(
                    pacs,
                    "load_deployment",
                    side_effect=lambda model: (
                        events.append(("load", model)),
                        self.deployment(),
                    )[1],
                ),
                patch.object(pacs, "PatchInferenceEngine", side_effect=RuntimeError("stop")),
            ):
                with self.assertRaisesRegex(RuntimeError, "stop"):
                    pacs.run_inference(
                        root / "input",
                        root / "output",
                        "unet",
                        version="release",
                        work_dir=root / "work",
                    )
        self.assertEqual(
            events,
            [("preflight", root / "input"), ("load", "unet")],
        )

    def test_dicom_preflight_failure_prevents_model_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with (
                patch.object(pacs, "_required_pr2mask_tools", return_value={}),
                patch.object(
                    pacs,
                    "validate_dicom_input",
                    side_effect=RuntimeError("invalid DICOM input"),
                ) as preflight,
                patch.object(pacs, "load_deployment") as load,
            ):
                with self.assertRaisesRegex(RuntimeError, "invalid DICOM input"):
                    pacs.run_inference(
                        root / "input",
                        root / "output",
                        "unet",
                        version="release",
                        work_dir=root / "work",
                    )
            preflight.assert_called_once_with(root / "input")
            load.assert_not_called()

    def test_invalid_paired_output_fails_before_dicom_writes(self):
        mask = torch.zeros((1, 3, 4, 5), dtype=torch.long)
        probabilities = torch.zeros((2, 2, 4, 5), dtype=torch.float32)
        engine = MagicMock()
        engine.predict_mask_and_probabilities.return_value = (mask, probabilities)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with (
                patch.object(pacs, "load_deployment", return_value=self.deployment()),
                patch.object(pacs, "PatchInferenceEngine", return_value=engine),
                patch.object(pacs, "write_prediction_outputs") as write_outputs,
                patch.object(pacs, "_required_pr2mask_tools", return_value={}),
                patch.object(pacs, "validate_dicom_input"),
                redirect_stdout(io.StringIO()),
            ):
                with self.assertRaisesRegex(RuntimeError, "different spatial shapes"):
                    pacs.run_inference(
                        "/dicom",
                        root / "output",
                        "unet",
                        version="release",
                        work_dir=root / "work",
                    )

        write_outputs.assert_not_called()

    def test_output_contract_rejects_invalid_values(self):
        valid_mask = torch.zeros((1, 2, 2, 2), dtype=torch.long)
        valid_probabilities = torch.full((2, 2, 2, 2), 0.5)
        invalid = [
            (valid_mask.squeeze(0), valid_probabilities, "mask shape"),
            (valid_mask.float(), valid_probabilities, "torch.long"),
            (valid_mask, valid_probabilities[:1], "two class-probability"),
            (valid_mask, valid_probabilities.long(), "floating-point"),
            (valid_mask, torch.zeros((2, 3, 2, 2)), "different spatial shapes"),
            (valid_mask, valid_probabilities.clone(), "non-finite"),
            (valid_mask, valid_probabilities.clone(), "outside"),
            (valid_mask.clone(), valid_probabilities, "labels other than"),
        ]
        invalid[5][1][0, 0, 0, 0] = torch.nan
        invalid[6][1][0, 0, 0, 0] = 1.1
        invalid[7][0][0, 0, 0, 0] = 2

        for mask, probabilities, message in invalid:
            with self.subTest(message=message):
                with self.assertRaisesRegex(RuntimeError, message):
                    pacs.validate_prediction_outputs(mask, probabilities)

    def test_tiny_probability_drift_is_clamped(self):
        mask = torch.zeros((1, 1, 1, 2), dtype=torch.long)
        probabilities = torch.tensor(
            [[[[[-5e-7, 0.5]]]], [[[[0.5, 1 + 5e-7]]]]]
        ).reshape(2, 1, 1, 2)
        _, validated = pacs.validate_prediction_outputs(mask, probabilities)
        self.assertEqual(float(validated.min()), 0.0)
        self.assertEqual(float(validated.max()), 1.0)


class PostprocessingTests(unittest.TestCase):
    def test_log_publication_replaces_symlink_without_following_it(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            work_log = root / "work.log"
            work_log.write_text("new log")
            output = root / "output"
            output.mkdir()
            protected = root / "protected.txt"
            protected.write_text("unchanged")
            published = output / pacs.LOG_NAME
            published.symlink_to(protected)

            pacs._publish_log(work_log, output)

            self.assertFalse(published.is_symlink())
            self.assertEqual(published.read_text(), "new log")
            self.assertEqual(protected.read_text(), "unchanged")

    def test_commands_bind_identity_and_copy_only_final_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_dir = root / "input"
            work_dir = root / "work"
            output_dir = root / "output"
            tools_dir = root / "pr2mask"
            input_dir.mkdir()
            output_dir.mkdir()
            (output_dir / pacs.LOG_NAME).write_text("old log")
            tools_dir.mkdir()
            for name in ("mask", "vote_map"):
                path = work_dir / name
                path.mkdir(parents=True)
                (path / "raw.dcm").write_text(name)
            tools = {
                name: tools_dir / name
                for name in ("imageAndMask2Report", "imageAndMask2Fused")
            }
            for path in tools.values():
                path.touch()

            calls = []

            def run(command, **kwargs):
                calls.append(command)
                self.assertEqual(
                    (output_dir / pacs.LOG_NAME).read_text(), "old log"
                )
                self.assertTrue((work_dir / pacs.LOG_NAME).is_file())
                kwargs["stdout"].write(f"command {len(calls)}\n")
                kwargs["stdout"].flush()
                if len(calls) == 3:
                    for name in pacs.FINAL_OUTPUT_DIRS:
                        path = work_dir / name
                        path.mkdir(exist_ok=True)
                        (path / "result.dcm").write_text(name)
                return SimpleNamespace(returncode=0)

            deployment = {
                "model_type": "unet",
                "bundle_sha256": "a" * 64,
                "members": [{"member_id": f"fold_{number}"} for number in range(5)],
            }
            with patch.object(pacs.subprocess, "run", side_effect=run):
                pacs._run_postprocessing(
                    input_dir,
                    work_dir,
                    output_dir,
                    deployment,
                    use_tta=True,
                    version="20260817T120000Z",
                    tools=tools,
                )

            self.assertEqual(len(calls), 3)
            identity = f"20260817T120000Z_m1_b{'a' * 32}_t1"
            self.assertLessEqual(len(identity), 64)
            self.assertIn(identity + "_report", calls[0])
            self.assertIn(identity + "_fused", calls[1])
            self.assertIn(identity + "_votemap", calls[2])
            self.assertIn("65535", calls[2])
            self.assertIn("0.5", calls[2])
            self.assertEqual(
                (output_dir / pacs.LOG_NAME).read_text(),
                "command 1\ncommand 2\ncommand 3\n",
            )
            self.assertEqual(
                (work_dir / pacs.LOG_NAME).read_text(),
                (output_dir / pacs.LOG_NAME).read_text(),
            )
            for name in pacs.FINAL_OUTPUT_DIRS:
                self.assertTrue((output_dir / name / "result.dcm").is_file())
            self.assertFalse((output_dir / "vote_map").exists())

    def test_missing_final_product_is_rejected_before_copy(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            work = root / "work"
            for name in ("mask", "fused"):
                (work / name).mkdir(parents=True, exist_ok=True)
            output = root / "output"
            with self.assertRaisesRegex(RuntimeError, "required output directories"):
                pacs._copy_final_outputs(work, output)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
