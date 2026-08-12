import io
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
from pydicom import dcmread
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import UID
from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACS_DIR = PROJECT_ROOT / "deployment" / "pacs"
sys.path.insert(0, str(PACS_DIR))

import stub_inference as stub
from deployment_models import make_dicom_uid_contract


class PredictionOutputTests(unittest.TestCase):
    def deployment(self):
        return {
            "patch_config": SimpleNamespace(patch_size=[16, 16, 16]),
            "expected_member_count": 1,
            "mode": "single",
            "model_type": "unet",
            "bundle_sha256": "a" * 64,
            "dicom_uid": make_dicom_uid_contract("unet", "single", 1),
            "predictor": object(),
            "members": [{"mlflow_run_id": "1234567890abcdef"}],
        }

    def test_pacs_uses_engine_mask_and_probability_channel_one(self):
        mask = torch.zeros((1, 3, 4, 5), dtype=torch.long)
        mask[0, 1, 1, 1] = 1
        probabilities = torch.zeros((2, 3, 4, 5), dtype=torch.float32)
        probabilities[0] = 0.1
        probabilities[1] = 0.9

        engine = MagicMock()
        engine.predict_mask_and_probabilities.return_value = (mask, probabilities)
        deployment = self.deployment()
        with (
            patch.object(stub, "load_deployment", return_value=deployment),
            patch.object(stub, "PatchInferenceEngine", return_value=engine),
            patch.object(stub, "create_dicom_mask") as write_mask,
            patch.object(stub, "create_dicom_prob_mask") as write_probability,
            redirect_stdout(io.StringIO()),
        ):
            stub.run_inference("/dicom", "/output", "unet", use_tta=True)

        engine.predict_mask_and_probabilities.assert_called_once_with(
            "/dicom", tta=True
        )
        written_mask = write_mask.call_args.args[0]
        written_probability = write_probability.call_args.args[0]
        self.assertTrue(torch.equal(written_mask, mask))
        self.assertTrue(torch.equal(written_probability, probabilities[1]))
        self.assertFalse(torch.equal(mask.squeeze(0), probabilities.argmax(0)))
        self.assertIs(write_mask.call_args.kwargs["deployment"], deployment)
        self.assertIn("software_versions", write_probability.call_args.kwargs)

    def test_invalid_paired_output_fails_before_dicom_writes(self):
        mask = torch.zeros((1, 3, 4, 5), dtype=torch.long)
        probabilities = torch.zeros((2, 2, 4, 5), dtype=torch.float32)
        engine = MagicMock()
        engine.predict_mask_and_probabilities.return_value = (mask, probabilities)

        with (
            patch.object(stub, "load_deployment", return_value=self.deployment()),
            patch.object(stub, "PatchInferenceEngine", return_value=engine),
            patch.object(stub, "create_dicom_mask") as write_mask,
            patch.object(stub, "create_dicom_prob_mask") as write_probability,
            redirect_stdout(io.StringIO()),
        ):
            with self.assertRaisesRegex(RuntimeError, "different spatial shapes"):
                stub.run_inference("/dicom", "/output", "unet")

        write_mask.assert_not_called()
        write_probability.assert_not_called()

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
                    stub.validate_prediction_outputs(mask, probabilities)

    def test_tiny_probability_drift_is_clamped(self):
        mask = torch.zeros((1, 1, 1, 2), dtype=torch.long)
        probabilities = torch.tensor(
            [[[[[-5e-7, 0.5]]]], [[[[0.5, 1 + 5e-7]]]]]
        ).reshape(2, 1, 1, 2)
        _, validated = stub.validate_prediction_outputs(mask, probabilities)
        self.assertEqual(float(validated.min()), 0.0)
        self.assertEqual(float(validated.max()), 1.0)


class DicomUIDTests(unittest.TestCase):
    def deployment(self, *, output_hash="a" * 64):
        return {
            "schema_version": 2,
            "model_type": "unet",
            "mode": "ensemble",
            "expected_member_count": 5,
            "bundle_sha256": output_hash,
            "dicom_uid": make_dicom_uid_contract("unet", "ensemble", 5),
            "members": [
                {"mlflow_run_id": f"{index:032d}"} for index in range(5)
            ],
        }

    def test_generated_uids_are_valid_deterministic_and_distinct(self):
        deployment = self.deployment()
        mask_series = stub.make_derived_dicom_uid(
            deployment, "source-series", "segmentation"
        )
        repeated = stub.make_derived_dicom_uid(
            deployment, "source-series", "segmentation"
        )
        probability_series = stub.make_derived_dicom_uid(
            deployment, "source-series", "probability"
        )
        instance = stub.make_derived_dicom_uid(
            deployment,
            "source-series",
            "segmentation",
            source_sop_uid="source-instance",
            slice_index=0,
        )
        changed_bundle = stub.make_derived_dicom_uid(
            self.deployment(output_hash="b" * 64),
            "source-series",
            "segmentation",
        )

        self.assertEqual(mask_series, repeated)
        self.assertEqual(len({mask_series, probability_series, instance, changed_bundle}), 4)
        for value in (mask_series, probability_series, instance, changed_bundle):
            self.assertTrue(UID(value).is_valid, value)
            self.assertLessEqual(len(value), 64)
            self.assertRegex(value, r"^2\.25\.[0-9]+$")

    def test_save_series_replaces_series_and_instance_uids(self):
        class FakeSeries:
            seriesInstanceUID = "source-series"
            SOPInstanceUIDs = {(0, 0): "source-sop-0", (0, 1): "source-sop-1"}
            patientID = "ABC123"
            slices = 2

            def __init__(self):
                self.attributes = []
                self.written = None

            def getDicomAttribute(self, keyword):
                return "fallback-sop"

            def setDicomAttribute(self, keyword, value, slice=None):
                self.attributes.append((keyword, value, slice))

            def write(self, path, opts, formats):
                self.written = (path, opts, formats)

        series = FakeSeries()
        with patch.object(stub, "_finalize_written_dicom") as finalize:
            stub.save_series_pred(
                series, "/output", self.deployment(), "segmentation"
            )
        sop_uids = [
            value for keyword, value, slice_index in series.attributes
            if keyword == "SOPInstanceUID" and slice_index is not None
        ]
        self.assertTrue(UID(series.seriesInstanceUID).is_valid)
        self.assertEqual(len(sop_uids), 2)
        self.assertEqual(len(set(sop_uids)), 2)
        self.assertTrue(all(UID(value).is_valid for value in sop_uids))
        self.assertEqual(
            series.written,
            ("/output", {"keep_uid": True}, ["dicom"]),
        )
        finalize.assert_called_once_with("/output")

    def test_written_file_meta_sop_uid_is_finalized(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "image.dcm"
            old_uid = "2.25.1"
            new_uid = "2.25.2"
            file_meta = FileMetaDataset()
            file_meta.MediaStorageSOPClassUID = MRImageStorage
            file_meta.MediaStorageSOPInstanceUID = old_uid
            file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
            dataset = FileDataset(
                str(path), {}, file_meta=file_meta, preamble=b"\0" * 128
            )
            dataset.SOPClassUID = MRImageStorage
            dataset.SOPInstanceUID = new_uid
            dataset.SeriesInstanceUID = "2.25.3"
            dataset.save_as(str(path), write_like_original=False)

            stub._finalize_written_dicom(directory)
            written = dcmread(str(path), stop_before_pixels=True)
            self.assertEqual(str(written.file_meta.MediaStorageSOPInstanceUID), new_uid)

    def test_both_outputs_receive_derived_metadata(self):
        class FakeSeries:
            def __init__(self):
                self.attributes = {}

            def getDicomAttribute(self, keyword):
                return ["ORIGINAL", "PRIMARY", "M", "ND"]

            def setDicomAttribute(self, keyword, value, **kwargs):
                self.attributes[keyword] = value

        for output_kind, marker in (
            ("segmentation", "MASK"),
            ("probability", "PROBABILITY"),
        ):
            with self.subTest(output_kind=output_kind):
                series = FakeSeries()
                stub._set_derived_metadata(
                    series, self.deployment(), output_kind, ["fastMONAI 0.10.0"]
                )
                self.assertEqual(series.attributes["ImageType"][:2], ["DERIVED", "SECONDARY"])
                self.assertEqual(series.attributes["ImageType"][-1], marker)
                self.assertIn("UNet 5-model ensemble", series.attributes["SeriesDescription"])
                self.assertIn("bundle_sha256=" + "a" * 64,
                              series.attributes["DerivationDescription"])
        self.assertIn("stored uint16 value / 65535",
                      series.attributes["DerivationDescription"])

    def test_runtime_rejects_a_uid_contract_not_matching_registry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "member.safetensors"
            artifact.write_bytes(b"model")
            member = {
                "member_id": "all_data",
                "artifact": artifact.name,
                "format": "safetensors",
                "artifact_role": "final",
                "mlflow_run_id": "run-id",
                "sha256": stub._sha256_file(artifact),
            }
            config = {
                "schema_version": 2,
                "model_type": "unet",
                "mode": "single",
                "expected_member_count": 1,
                "dicom_uid": make_dicom_uid_contract("unet", "single", 1),
                "members": [member],
                "model_spec": {"arch_id": "monai.unet"},
                "inference_config": {"canonical_sha256": "a" * 64},
            }
            config["dicom_uid"]["model_code"] = 999
            config["bundle_sha256"] = stub._sha256_bytes(
                stub._canonical_json(config).encode("utf-8")
            )
            with self.assertRaisesRegex(RuntimeError, "UID contract"):
                stub._validate_new_deployment(root, config, "unet")


if __name__ == "__main__":
    unittest.main()
