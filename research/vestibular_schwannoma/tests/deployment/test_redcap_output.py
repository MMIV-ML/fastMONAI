import base64
import csv
import gzip
import io
import json
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile
from pathlib import Path

import numpy as np
from pydicom import dcmread

PACS_DIR = Path(__file__).resolve().parents[2] / "deployment" / "pacs"
sys.path.insert(0, str(PACS_DIR))
from redcap_output import (  # noqa: E402
    encode_mask,
    decode_mask,
    build_mask_payload,
    write_redcap_mask,
    FIELDS,
    model_repeat_instance,
)
import redcap_output  # noqa: E402
from vestibular_schwannoma.tests.deployment.test_dicom_output import _write_test_image  # noqa: E402

DEPLOYMENT = {
    "model_type": "unet",
    "bundle_sha256": "4d8991eff16c90ad0eb185757df9cfccf5615cb7a5d7a9d3304c61e65ad9d172",
    "members": [{"member_id": "all_data"}],
}


def fixture(root, empty=False):
    source, work = root / "source", root / "work"
    source.mkdir()
    (work / "mask").mkdir(parents=True)
    mask = np.zeros((3, 2, 3), dtype=np.uint16)
    if not empty:
        mask[0, 0, 2] = mask[2, 1, 0] = 1
    # Oblique to the usual axial grid, anisotropic and irregular slice positions.
    for i, offset in enumerate([0.0, 2.0, 5.0]):
        kwargs = dict(
            orientation=(0, 1, 0, 0, 0, 1),
            pixel_spacing=(0.7, 0.9),
            position=(offset, 12, 31),
        )
        p = _write_test_image(source / f"{2 - i}", i, **kwargs)
        d = dcmread(p)
        d.PatientID = "SYNTHETIC"
        d.ReferringPhysicianName = "EventName:test_arm_1"
        d.save_as(p)
        p = _write_test_image(
            work / "mask" / f"{2 - i}",
            i,
            series_uid="1.2.8",
            sop_uid=f"1.2.8.{i + 1}",
            **kwargs,
        )
        d = dcmread(p)
        d.PixelData = mask[i].astype("<u2").tobytes()
        d.save_as(p)
    report = work / "redcap" / "1.2.9"
    report.mkdir(parents=True)
    rows = [
        dict(
            record_id="SYNTHETIC",
            redcap_event_name="test_arm_1",
            redcap_repeat_instrument="pr2mask",
            redcap_repeat_instance=str(i),
            field_name="physical_size",
            value="1",
        )
        for i in [1, 2]
    ]
    (report / "output.json").write_text(json.dumps([] if empty else rows))
    with zipfile.ZipFile(report / "output_data_dictionary.zip", "w") as z:
        z.writestr("OriginID.txt", "PR2MASK")
        z.writestr(
            "instrument.csv",
            "Variable / Field Name,Form Name,Field Type,Field Label\nphysical_size,pr2mask,text,Physical size\n",
        )
    return source, work, mask, report, ([] if empty else rows)


class CodecTests(unittest.TestCase):
    def test_empty_full_and_random_masks_with_padding(self):
        rng = np.random.default_rng(10)
        for mask in [
            np.zeros((1, 1, 1)),
            np.ones((3, 5, 7)),
            rng.integers(0, 2, (3, 5, 7)),
        ]:
            payload = json.loads(json.dumps(encode_mask(mask)))
            np.testing.assert_array_equal(decode_mask(payload), mask)
            self.assertNotIn("spatial_units", payload)

    def test_corruption_and_invalid_input(self):
        payload = encode_mask(np.ones((3, 5, 7)))
        for key, value in [
            ("mask_sha256", "0" * 64),
            ("data", "bad!"),
            ("shape", [1, 1, 1]),
            ("foreground_voxels", 0),
            ("encoding", "unknown"),
            ("shape", [999999999, 1, 1]),
        ]:
            with self.subTest(key=key):
                changed = dict(payload, **{key: value})
                with self.assertRaises(ValueError):
                    decode_mask(changed)
        with self.assertRaises(ValueError):
            encode_mask(np.full((2, 2, 2), 2))
        payload = encode_mask(np.ones((1, 1, 1)))
        payload["data"] = base64.b64encode(gzip.compress(bytes([255]))).decode()
        with self.assertRaisesRegex(ValueError, "padding"):
            decode_mask(payload)


class DicomExportTests(unittest.TestCase):
    def test_dicom_pixels_geometry_rows_and_dictionary_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, work, mask, report, original = fixture(Path(tmp))
            write_redcap_mask(work, source, DEPLOYMENT, version="test", use_tta=True)
            rows = json.loads((report / "output.json").read_text())
            values = {r["field_name"]: r["value"] for r in rows}
            self.assertEqual(
                json.loads(values["vs_measurements_json"]),
                [
                    {k: r[k] for k in ("redcap_repeat_instance", "field_name", "value")}
                    for r in original
                ],
            )
            self.assertEqual(set(values), set(FIELDS))
            payload = json.loads(values["vs_mask_json"])
            np.testing.assert_array_equal(decode_mask(payload), mask)
            self.assertEqual(
                payload["geometry"]["image_positions_patient"],
                [[0, 12, 31], [2, 12, 31], [5, 12, 31]],
            )
            self.assertEqual(payload["geometry"]["pixel_spacing"], [0.7, 0.9])
            self.assertEqual(
                payload["source"]["sop_instance_uids"],
                ["1.2.3.4.1", "1.2.3.4.2", "1.2.3.4.3"],
            )
            self.assertEqual(
                payload["model"]["bundle_sha256"], DEPLOYMENT["bundle_sha256"]
            )
            self.assertEqual(values["vs_prediction_id"], payload["prediction_id"])
            self.assertTrue(all(r["redcap_repeat_instance"] == "1" for r in rows))
            with zipfile.ZipFile(report / "output_data_dictionary.zip") as z:
                self.assertEqual(z.read("OriginID.txt"), b"PR2MASK")
                fields = {
                    r["Variable / Field Name"]: r
                    for r in csv.DictReader(
                        io.StringIO(z.read("instrument.csv").decode())
                    )
                }
                self.assertEqual(fields["vs_mask_json"]["Field Type"], "notes")
                self.assertIn("physical_size", fields)
            again, _ = build_mask_payload(
                work / "mask", source, DEPLOYMENT, version="test", use_tta=True
            )
            self.assertEqual(again["prediction_id"], payload["prediction_id"])
            changed, _ = build_mask_payload(
                work / "mask", source, DEPLOYMENT, version="test", use_tta=False
            )
            self.assertNotEqual(changed["prediction_id"], payload["prediction_id"])

    def test_empty_mask_and_no_measurements(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, work, mask, report, _ = fixture(Path(tmp), empty=True)
            write_redcap_mask(work, source, DEPLOYMENT, version="test", use_tta=False)
            rows = json.loads((report / "output.json").read_text())
            self.assertEqual(rows[0]["record_id"], "SYNTHETIC")
            self.assertEqual(rows[0]["redcap_event_name"], "test_arm_1")
            np.testing.assert_array_equal(
                decode_mask(
                    json.loads(
                        next(
                            r["value"]
                            for r in rows
                            if r["field_name"] == "vs_mask_json"
                        )
                    )
                ),
                mask,
            )

    def test_models_have_distinct_destinations_and_reruns_replace_complete_result(self):
        stored = {}
        dynunet = dict(
            DEPLOYMENT,
            model_type="dynunet",
            bundle_sha256="c5642793b0f57b49ea8f04a08d2b8c9deff1523813d87d89eec204f85846d6ec",
        )
        for deployment, tta, version, empty in [
            (DEPLOYMENT, True, "v1", False),
            (dynunet, True, "v1", False),
            (DEPLOYMENT, False, "v2", True),
        ]:
            with tempfile.TemporaryDirectory() as tmp:
                source, work, _, report, _ = fixture(Path(tmp), empty=empty)
                write_redcap_mask(
                    work, source, deployment, version=version, use_tta=tta
                )
                for row in json.loads((report / "output.json").read_text()):
                    key = tuple(
                        row[k]
                        for k in (
                            "record_id",
                            "redcap_event_name",
                            "redcap_repeat_instrument",
                            "redcap_repeat_instance",
                            "field_name",
                        )
                    )
                    stored[key] = row["value"]
        by_instance = {}
        for key, value in stored.items():
            by_instance.setdefault(key[3], {})[key[4]] = value
        self.assertEqual(set(by_instance), {"1", "2"})
        self.assertEqual(by_instance["1"]["vs_tta"], "0")
        self.assertEqual(by_instance["1"]["vs_deployment_version"], "v2")
        self.assertEqual(json.loads(by_instance["1"]["vs_measurements_json"]), [])
        self.assertEqual(
            json.loads(by_instance["1"]["vs_mask_json"])["foreground_voxels"], 0
        )
        self.assertEqual(by_instance["2"]["vs_tta"], "1")
        self.assertTrue(json.loads(by_instance["2"]["vs_measurements_json"]))

    def test_unknown_and_colliding_registry_entries_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unregistered"):
            model_repeat_instance("f" * 64)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "registry.json"
            path.write_text(json.dumps({"a" * 64: 1, "b" * 64: 1}))
            with patch.object(redcap_output, "MODEL_INSTANCES_PATH", path):
                with self.assertRaisesRegex(ValueError, "Invalid"):
                    model_repeat_instance("a" * 64)

    def test_geometry_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, work, *_ = fixture(Path(tmp))
            p = work / "mask" / "0"
            d = dcmread(p)
            d.ImagePositionPatient = [6, 12, 31]
            d.save_as(p)
            with self.assertRaisesRegex(ValueError, "geometry differ"):
                build_mask_payload(
                    work / "mask", source, DEPLOYMENT, version="test", use_tta=True
                )

    def test_duplicate_slice_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, work, *_ = fixture(Path(tmp))
            p = work / "mask" / "0"
            d = dcmread(p)
            d.ImagePositionPatient = [2, 12, 31]
            d.save_as(p)
            with self.assertRaisesRegex(ValueError, "Duplicate slice"):
                build_mask_payload(
                    work / "mask", source, DEPLOYMENT, version="test", use_tta=True
                )


if __name__ == "__main__":
    unittest.main()
