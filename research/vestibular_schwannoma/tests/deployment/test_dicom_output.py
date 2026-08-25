import sys
import tempfile
import unittest
import warnings
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from pydicom import dcmread
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, UID


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACS_DIR = PROJECT_ROOT / "deployment" / "pacs"
sys.path.insert(0, str(PACS_DIR))

import dicom_output as dicom  # noqa: E402


REGISTERED_TEST_PREFIX = "1.2.826.0.1.3680043.10.9999"
SOURCE_SERIES_UID = "1.2.3.4"
SOURCE_SOP_UIDS = ["1.2.3.4.1", "1.2.3.4.2"]
SOP_SEQUENCE_SHA256 = "b" * 64
PIXEL_SHA256 = "c" * 64


HEX_STUDY_UID = "a" * 64
HEX_SERIES_UID = "b" * 64
HEX_FRAME_UID = "d" * 64


def _write_test_image(
    path,
    index,
    *,
    study_uid="1.2.3",
    series_uid="1.2.3.4",
    sop_uid=None,
    modality="MR",
    rows=2,
    columns=3,
    pixel_spacing=(1.0, 1.0),
    orientation=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
    position=None,
    frame_uid="1.2.3.5",
    meta_sop_uid=None,
    omit=(),
):
    sop_uid = sop_uid or f"1.2.3.4.{index + 1}"
    position = position or (0.0, 0.0, float(index))
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = MRImageStorage
    if meta_sop_uid is not False:
        file_meta.MediaStorageSOPInstanceUID = (
            sop_uid if meta_sop_uid is None else meta_sop_uid
        )
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    dataset = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\0" * 128)
    values = {
        "SOPClassUID": MRImageStorage,
        "SOPInstanceUID": sop_uid,
        "StudyInstanceUID": study_uid,
        "SeriesInstanceUID": series_uid,
        "FrameOfReferenceUID": frame_uid,
        "Modality": modality,
        "Rows": rows,
        "Columns": columns,
        "PixelSpacing": list(pixel_spacing),
        "ImageOrientationPatient": list(orientation),
        "ImagePositionPatient": list(position),
    }
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"^Invalid value for VR UI:",
            category=UserWarning,
            module=r"^pydicom\.valuerep$",
        )
        for keyword, value in values.items():
            if keyword not in omit and value is not None:
                setattr(dataset, keyword, value)
        dataset.SamplesPerPixel = 1
        dataset.PhotometricInterpretation = "MONOCHROME2"
        dataset.BitsAllocated = 16
        dataset.BitsStored = 16
        dataset.HighBit = 15
        dataset.PixelRepresentation = 0
        dataset.PixelData = np.zeros((rows, columns), dtype="<u2").tobytes()
        dicom._save_dataset(dataset, path, enforce_file_format=True)
        if meta_sop_uid is not None:
            written = dcmread(str(path))
            if meta_sop_uid is False:
                del written.file_meta.MediaStorageSOPInstanceUID
            else:
                written.file_meta.MediaStorageSOPInstanceUID = meta_sop_uid
            dicom._save_dataset(written, path, enforce_file_format=False)
    return path


class DicomInputValidationTests(unittest.TestCase):
    def test_unreadable_dicom_candidate_is_fatal(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "broken.dcm"
            path.write_bytes(b"\0" * 128 + b"DICM" + b"broken")
            with patch.object(dicom, "dcmread", side_effect=OSError("broken")):
                with self.assertRaisesRegex(
                    RuntimeError,
                    r"Unable to read DICOM file: .*broken\.dcm",
                ):
                    dicom.validate_dicom_input(directory)

    def test_unrelated_pydicom_warning_remains_visible(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_test_image(root / "0.dcm", 0)
            real_dcmread = dicom.dcmread

            def read_with_warning(*args, **kwargs):
                warnings.warn("unexpected DICOM warning", UserWarning)
                return real_dcmread(*args, **kwargs)

            with (
                patch.object(dicom, "dcmread", side_effect=read_with_warning),
                warnings.catch_warnings(record=True) as caught,
            ):
                warnings.simplefilter("always")
                dicom.validate_dicom_input(root)

        self.assertEqual(
            [str(item.message) for item in caught],
            ["unexpected DICOM warning"],
        )

    def test_hex_source_identifiers_warn_once_and_are_deterministic_entropy(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sop_uids = ["c" * 63 + "1", "c" * 63 + "2"]
            for index, sop_uid in enumerate(sop_uids):
                _write_test_image(
                    root / f"{index}.dcm",
                    index,
                    study_uid=HEX_STUDY_UID,
                    series_uid=HEX_SERIES_UID,
                    sop_uid=sop_uid,
                    frame_uid=HEX_FRAME_UID,
                )
            (root / "descr.json").write_text("{}", encoding="utf-8")

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                result = dicom.validate_dicom_input(root)

        self.assertIsNone(result)
        self.assertEqual(len(caught), 1)
        self.assertIn("nonstandard source UID syntax", str(caught[0].message))

        deployment = {
            "schema_version": 1,
            "model_type": "unet",
            "bundle_sha256": "a" * 64,
            "members": [{}],
        }
        kwargs = {
            "source_sop_sequence_sha256": SOP_SEQUENCE_SHA256,
            "output_pixels_sha256": "e" * 64,
            "use_tta": False,
        }
        first = dicom.make_derived_series_uid(
            deployment, HEX_SERIES_UID, dicom.SEGMENTATION_MASK, **kwargs
        )
        repeated = dicom.make_derived_series_uid(
            deployment, HEX_SERIES_UID, dicom.SEGMENTATION_MASK, **kwargs
        )
        instance = dicom._make_derived_instance_uid(deployment, first, sop_uids[0])
        self.assertEqual(first, repeated)
        for value in (first, instance):
            self.assertTrue(UID(value).is_valid)
            self.assertLessEqual(len(value), 64)

    def test_file_meta_and_frame_issues_are_one_nonfatal_warning(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_test_image(
                root / "0.dcm",
                0,
                meta_sop_uid="1.2.9",
                frame_uid=None,
            )
            _write_test_image(
                root / "1.dcm",
                1,
                meta_sop_uid=False,
                frame_uid="1.2.3.8",
            )
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                result = dicom.validate_dicom_input(root)

        self.assertIsNone(result)
        self.assertEqual(len(caught), 1)
        message = str(caught[0].message)
        self.assertIn("file-meta SOP identity", message)
        self.assertIn("FrameOfReferenceUID", message)

    def test_non_dicom_sidecars_are_skipped_but_images_are_required(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "descr.json").write_text("{}", encoding="utf-8")
            (root / "nested").mkdir()
            with (
                patch.object(dicom, "dcmread") as read,
                self.assertRaisesRegex(RuntimeError, "no readable image DICOM"),
            ):
                dicom.validate_dicom_input(root)
            read.assert_not_called()

    def test_missing_identifiers_and_mixed_identity_are_fatal(self):
        cases = (
            ("StudyInstanceUID", {"omit": {"StudyInstanceUID"}}, "missing required"),
            ("SeriesInstanceUID", {"omit": {"SeriesInstanceUID"}}, "missing required"),
            ("SOPInstanceUID", {"omit": {"SOPInstanceUID"}}, "missing required"),
        )
        for label, changes, message in cases:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                _write_test_image(Path(directory) / "0.dcm", 0, **changes)
                with self.assertRaisesRegex(RuntimeError, message):
                    dicom.validate_dicom_input(directory)

        mixed_cases = (
            ("StudyInstanceUID", {"study_uid": "1.2.99"}, "multiple Study"),
            ("SeriesInstanceUID", {"series_uid": "1.2.99"}, "multiple Series"),
            ("Modality", {"modality": "CT"}, "one MR modality"),
            ("SOPInstanceUID", {"sop_uid": "1.2.3.4.1"}, "duplicate SOP"),
        )
        for label, changes, message in mixed_cases:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                _write_test_image(root / "0.dcm", 0)
                _write_test_image(root / "1.dcm", 1, **changes)
                with self.assertRaisesRegex(RuntimeError, message):
                    dicom.validate_dicom_input(root)

    def test_missing_or_inconsistent_geometry_is_fatal(self):
        fields = (
            "Rows",
            "Columns",
            "PixelSpacing",
            "ImageOrientationPatient",
            "ImagePositionPatient",
        )
        for field in fields:
            with (
                self.subTest(missing=field),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                _write_test_image(root / "0.dcm", 0)
                _write_test_image(root / "1.dcm", 1, omit={field})
                with self.assertRaisesRegex(RuntimeError, field):
                    dicom.validate_dicom_input(root)

        changes = (
            ({"rows": 4}, "Rows or Columns"),
            ({"columns": 4}, "Rows or Columns"),
            ({"pixel_spacing": (1.0, 2.0)}, "PixelSpacing"),
            (
                {"orientation": (0.0, 1.0, 0.0, 1.0, 0.0, 0.0)},
                "ImageOrientationPatient",
            ),
            ({"position": (0.0, 0.0, 0.0)}, "duplicate ImagePositionPatient"),
            ({"position": (1.0, 0.0, 0.0)}, "ImagePositionPatient"),
        )
        for change, message in changes:
            with (
                self.subTest(change=change),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                _write_test_image(root / "0.dcm", 0)
                _write_test_image(root / "1.dcm", 1, **change)
                with self.assertRaisesRegex(RuntimeError, message):
                    dicom.validate_dicom_input(root)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_test_image(root / "0.dcm", 0)
            _write_test_image(root / "1.dcm", 1)
            _write_test_image(root / "2.dcm", 2, position=(0.0, 0.0, 3.0))
            self.assertIsNone(dicom.validate_dicom_input(root))


class DicomUIDTests(unittest.TestCase):
    def deployment(self, *, bundle_hash="a" * 64, prefix=None, model_type="unet"):
        deployment = {
            "schema_version": 1,
            "model_type": model_type,
            "bundle_sha256": bundle_hash,
            "members": [{} for _ in range(5)],
        }
        if prefix is not None:
            deployment["registered_prefix"] = prefix
        return deployment

    def uid(self, deployment, representation=dicom.SEGMENTATION_MASK, **extra):
        kwargs = {
            "source_sop_sequence_sha256": SOP_SEQUENCE_SHA256,
            "output_pixels_sha256": PIXEL_SHA256,
            "use_tta": False,
        }
        kwargs.update(extra)
        return dicom.make_derived_series_uid(
            deployment, SOURCE_SERIES_UID, representation, **kwargs
        )

    def test_default_uid_golden_vector_and_identity_separation(self):
        deployment = self.deployment()
        mask_series = self.uid(deployment)
        repeated = self.uid(deployment)
        probability_series = self.uid(deployment, dicom.PROBABILITY_MAP)
        changed_bundle = self.uid(self.deployment(bundle_hash="d" * 64))
        changed_tta = self.uid(deployment, use_tta=True)
        changed_pixels = self.uid(deployment, output_pixels_sha256="e" * 64)
        with patch.object(dicom.fastMONAI, "__version__", "0.10.2"):
            changed_fastmonai = self.uid(deployment)

        self.assertEqual(
            mask_series,
            "2.25.279307575913214821815280391887400559925",
        )
        self.assertEqual(mask_series, repeated)
        values = {
            mask_series,
            probability_series,
            changed_bundle,
            changed_tta,
            changed_pixels,
            changed_fastmonai,
        }
        self.assertEqual(len(values), 6)
        for value in values:
            self.assertTrue(UID(value).is_valid, value)
            self.assertLessEqual(len(value), 64)
            self.assertRegex(value, r"^2\.25\.[0-9]+$")

    def test_registered_prefix_uid_golden_vector(self):
        value = self.uid(self.deployment(prefix=REGISTERED_TEST_PREFIX))
        self.assertEqual(
            value,
            "1.2.826.0.1.3680043.10.9999.702858577822640103404256011291145367",
        )
        self.assertTrue(UID(value).is_valid)
        self.assertLessEqual(len(value), 64)
        self.assertTrue(value.startswith(REGISTERED_TEST_PREFIX + "."))

    def test_instance_uid_uses_only_series_and_source_sop_identity(self):
        series_uid = self.uid(self.deployment())
        first = dicom._make_derived_instance_uid(
            self.deployment(), series_uid, SOURCE_SOP_UIDS[0]
        )
        changed_irrelevant_bundle = dicom._make_derived_instance_uid(
            self.deployment(bundle_hash="d" * 64), series_uid, SOURCE_SOP_UIDS[0]
        )
        changed_source = dicom._make_derived_instance_uid(
            self.deployment(), series_uid, SOURCE_SOP_UIDS[1]
        )
        self.assertEqual(
            first,
            "2.25.91692261918121449464758301100525040869",
        )
        self.assertEqual(first, changed_irrelevant_bundle)
        self.assertNotEqual(first, changed_source)

    def test_canonical_identity_and_representation_validation(self):
        deployment = self.deployment()
        common = {
            "source_sop_sequence_sha256": SOP_SEQUENCE_SHA256,
            "output_pixels_sha256": PIXEL_SHA256,
            "use_tta": False,
        }
        first = dicom.make_derived_series_uid(
            deployment, "1.2.3", dicom.SEGMENTATION_MASK, **common
        )
        second = dicom.make_derived_series_uid(
            deployment, "1.2.34", dicom.SEGMENTATION_MASK, **common
        )
        self.assertNotEqual(first, second)
        with self.assertRaisesRegex(ValueError, "prediction representation"):
            dicom.make_derived_series_uid(deployment, "1.2.3", "unknown", **common)

    def test_save_series_replaces_uids_and_preserves_study_id(self):
        class FakeSeries:
            patientID = "ABC123"
            slices = 2

            def __init__(self):
                self.studyID = "SOURCE-STUDY"
                self.studyInstanceUID = HEX_STUDY_UID
                self.frameOfReferenceUID = HEX_FRAME_UID
                self.seriesInstanceUID = HEX_SERIES_UID
                self.SOPInstanceUIDs = {
                    (0, 0): SOURCE_SOP_UIDS[0],
                    (0, 1): SOURCE_SOP_UIDS[1],
                }
                self.attributes = []
                self.written = None

            def setDicomAttribute(self, keyword, value, slice=None):
                self.attributes.append((keyword, value, slice))

            def write(self, path, opts, formats):
                self.written = (path, opts, formats)

        for prefix in (None, REGISTERED_TEST_PREFIX):
            with self.subTest(prefix=prefix):
                series = FakeSeries()
                with (
                    patch.object(dicom, "_finalize_written_dicom") as finalize,
                    patch.object(
                        dicom,
                        "make_derived_series_uid",
                        wraps=dicom.make_derived_series_uid,
                    ) as make_series_uid,
                ):
                    dicom.save_series_pred(
                        series,
                        "/output",
                        self.deployment(prefix=prefix),
                        dicom.SEGMENTATION_MASK,
                        output_pixels_sha256=PIXEL_SHA256,
                        use_tta=False,
                    )
                expected_sop_digest = dicom.sha256_bytes(
                    dicom.canonical_json(SOURCE_SOP_UIDS).encode("utf-8")
                )
                self.assertEqual(
                    make_series_uid.call_args.kwargs["source_sop_sequence_sha256"],
                    expected_sop_digest,
                )
                sop_uids = [
                    value
                    for keyword, value, slice_index in series.attributes
                    if keyword == "SOPInstanceUID" and slice_index is not None
                ]
                expected_prefix = "2.25" if prefix is None else prefix
                self.assertTrue(
                    series.seriesInstanceUID.startswith(expected_prefix + ".")
                )
                self.assertEqual(series.studyID, "SOURCE-STUDY")
                self.assertEqual(series.studyInstanceUID, HEX_STUDY_UID)
                self.assertEqual(series.frameOfReferenceUID, HEX_FRAME_UID)
                self.assertEqual(len(sop_uids), 2)
                self.assertEqual(len(set(sop_uids)), 2)
                self.assertTrue(all(UID(value).is_valid for value in sop_uids))
                self.assertEqual(
                    series.written,
                    ("/output", {"keep_uid": True}, ["dicom"]),
                )
                finalize.assert_called_once_with("/output")

    def test_missing_or_duplicate_source_sop_uids_are_rejected(self):
        series = SimpleNamespace(slices=2, SOPInstanceUIDs=None)
        with self.assertRaisesRegex(RuntimeError, "per-slice SOP"):
            dicom._source_sop_uids(series)

        series.SOPInstanceUIDs = {
            (0, 0): SOURCE_SOP_UIDS[0],
            (0, 1): SOURCE_SOP_UIDS[0],
        }
        with self.assertRaisesRegex(RuntimeError, "duplicate SOP"):
            dicom._source_sop_uids(series)

    def test_pixel_digest_includes_shape_and_uint16_values(self):
        first = np.array([[1, 2]], dtype=np.uint16)
        same_bytes_different_shape = first.reshape(2, 1)
        changed = np.array([[1, 3]], dtype=np.uint16)
        self.assertNotEqual(
            dicom._pixel_array_sha256(first),
            dicom._pixel_array_sha256(same_bytes_different_shape),
        )
        self.assertNotEqual(
            dicom._pixel_array_sha256(first),
            dicom._pixel_array_sha256(changed),
        )

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
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                dataset.StudyInstanceUID = HEX_STUDY_UID
                dataset.FrameOfReferenceUID = HEX_FRAME_UID
                dicom._save_dataset(dataset, path, enforce_file_format=True)

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                dicom._finalize_written_dicom(directory)
            self.assertEqual(caught, [])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                written = dcmread(str(path), stop_before_pixels=True)
            self.assertEqual(str(written.file_meta.MediaStorageSOPInstanceUID), new_uid)
            self.assertEqual(str(written.StudyInstanceUID), HEX_STUDY_UID)
            self.assertEqual(str(written.FrameOfReferenceUID), HEX_FRAME_UID)

    def test_metadata_has_software_and_model_provenance(self):
        class FakeSeries:
            def __init__(self):
                self.attributes = {}

            def getDicomAttribute(self, keyword):
                return ["ORIGINAL", "PRIMARY", "M", "ND"]

            def setDicomAttribute(self, keyword, value, **kwargs):
                self.attributes[keyword] = value

        for representation, marker in (
            (dicom.SEGMENTATION_MASK, "MASK"),
            (dicom.PROBABILITY_MAP, "PROBABILITY"),
        ):
            with self.subTest(representation=representation):
                series = FakeSeries()
                dicom._set_derived_metadata(
                    series, self.deployment(), representation, True
                )
                self.assertEqual(
                    series.attributes["SoftwareVersions"],
                    [f"fastMONAI {dicom.fastMONAI.__version__}"],
                )
                self.assertEqual(
                    series.attributes["ImageType"][:2], ["DERIVED", "SECONDARY"]
                )
                self.assertEqual(series.attributes["ImageType"][-1], marker)
                self.assertIn(
                    "UNet 5-model ensemble", series.attributes["SeriesDescription"]
                )
                derivation = series.attributes["DerivationDescription"]
                self.assertIn("model=unet", derivation)
                self.assertIn("members=5", derivation)
                self.assertIn("bundle_sha256=" + "a" * 64, derivation)
                self.assertIn("tta=on", derivation)
                if representation == dicom.PROBABILITY_MAP:
                    self.assertIn("stored uint16 value / 65535", derivation)
                else:
                    self.assertNotIn("stored uint16 value / 65535", derivation)

    def test_probability_map_is_stored_as_scaled_uint16(self):
        class FakePrediction:
            def numpy(self):
                return np.array(
                    [[[[0.0, 0.5], [1.0, 0.25]], [[0.75, 0.1], [0.9, 0.2]]]],
                    dtype=np.float32,
                )

        class FakeSeries:
            def __init__(self, *args, **kwargs):
                self.attributes = {}
                self.data = None

            def getDicomAttribute(self, keyword):
                return None

            def setDicomAttribute(self, keyword, value, **kwargs):
                self.attributes[keyword] = value

            def __setitem__(self, key, value):
                self.data = value

        fake_series = FakeSeries()
        with (
            patch.object(dicom, "Series", return_value=fake_series),
            patch.object(dicom, "save_series_pred") as save,
            tempfile.TemporaryDirectory() as directory,
        ):
            dicom.create_dicom_probability_map(
                FakePrediction(),
                "/input",
                directory,
                self.deployment(),
                use_tta=False,
            )
        expected = np.rint(
            np.transpose(FakePrediction().numpy().squeeze(), (-1, 1, 0)) * 65535
        ).astype("<u2")
        np.testing.assert_array_equal(fake_series.data, expected)
        self.assertEqual(fake_series.data.dtype, np.dtype("<u2"))
        save.assert_called_once()


if __name__ == "__main__":
    unittest.main()
