import csv
import io
import json
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile
from pathlib import Path

from pydicom import dcmread

PACS_DIR = Path(__file__).resolve().parents[2] / "deployment" / "pacs"
sys.path.insert(0, str(PACS_DIR))
from redcap_output import write_redcap_output, model_repeat_instance  # noqa: E402
import redcap_output  # noqa: E402
from vestibular_schwannoma.tests.deployment.test_dicom_output import _write_test_image  # noqa: E402

DICTIONARY_HEADER = "Variable / Field Name,Form Name,Field Type,Field Label\n"
DICTIONARY_ROWS = (
    "mask_base64_gzip_compressed,pr2mask,text,mask_base64_gzip_compressed\n"
    "number_of_pixel,pr2mask,text,Number of pixel\n"
    "physical_size,pr2mask,text,Physical size\n"
)
PR2MASK_COLUMNS = ["mask_base64_gzip_compressed", "number_of_pixel", "physical_size"]
ROUTING = ("record_id", "redcap_event_name", "redcap_repeat_instrument", "redcap_repeat_instance")
DEPLOYMENT = {
    "model_type": "dynunet",
    "bundle_sha256": "c5642793b0f57b49ea8f04a08d2b8c9deff1523813d87d89eec204f85846d6ec",
    "members": [{"member_id": "all_data"}],
}


def fixture(root, empty=False):
    source, work = root / "source", root / "work"
    source.mkdir()
    for i in range(3):
        p = _write_test_image(source / f"{i}", i, position=(0, 0, i))
        d = dcmread(p)
        d.PatientID = "SYNTHETIC"
        d.ReferringPhysicianName = "EventName:test_arm_1"
        d.save_as(p)
    report = work / "redcap" / "1.2.9"
    report.mkdir(parents=True)
    rows = [
        dict(
            record_id="SYNTHETIC",
            redcap_event_name="test_arm_1",
            redcap_repeat_instrument="pr2mask",
            redcap_repeat_instance=str(i),
            field_name=name,
            value=value,
        )
        for i, size in [(1, "20"), (2, "50"), (3, "50")]
        for name, value in [
            ("mask_base64_gzip_compressed", f"m{i}"),
            ("number_of_pixel", size),
            ("physical_size", f"{size}.{i}"),
        ]
    ]
    (report / "output.json").write_text(json.dumps([] if empty else rows))
    with zipfile.ZipFile(report / "output_data_dictionary.zip", "w") as z:
        z.writestr("OriginID.txt", "PR2MASK")
        # Like pr2mask, list the columns only when there is a region.
        z.writestr("instrument.csv", DICTIONARY_HEADER + ("" if empty else DICTIONARY_ROWS))
    return source, work, report


def dictionary_fields(report):
    with zipfile.ZipFile(report / "output_data_dictionary.zip") as z:
        rows = csv.DictReader(io.StringIO(z.read("instrument.csv").decode()))
        return z.read("OriginID.txt"), [r["Variable / Field Name"] for r in rows]


class Pr2maskDictionaryTests(unittest.TestCase):
    def test_shipped_dictionary_lists_pr2mask_columns(self):
        names = [row["Variable / Field Name"] for row in redcap_output.pr2mask_dictionary_rows()]
        self.assertIn("number_of_pixel", names)
        self.assertIn("physical_size", names)
        self.assertEqual(len(names), len(set(names)))
        self.assertFalse([name for name in names if name.startswith("vs_")])


class RedcapOutputTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        reference = Path(tmp.name) / "pr2mask_dictionary.csv"
        reference.write_text(DICTIONARY_HEADER + DICTIONARY_ROWS)
        patcher = patch.object(redcap_output, "PR2MASK_DICTIONARY_PATH", reference)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_largest_region_fills_pr2mask_columns_in_model_instance(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, work, report = fixture(Path(tmp))
            original = (report / "output_data_dictionary.zip").read_bytes()
            write_redcap_output(work, source, DEPLOYMENT)
            rows = json.loads((report / "output.json").read_text())
            self.assertEqual([r["field_name"] for r in rows], PR2MASK_COLUMNS)
            self.assertEqual(
                {tuple(r[k] for k in ROUTING) for r in rows},
                {("SYNTHETIC", "test_arm_1", "pr2mask", "11")},
            )
            values = {r["field_name"]: r["value"] for r in rows}
            # Regions 2 and 3 tie on size: the lower region number fills the columns.
            self.assertEqual(
                [values[name] for name in PR2MASK_COLUMNS], ["m2", "50", "50.2"]
            )
            # pr2mask's dictionary already lists its columns and is left as written.
            self.assertEqual((report / "output_data_dictionary.zip").read_bytes(), original)

    def test_empty_mask_sends_blank_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, work, report = fixture(Path(tmp), empty=True)
            write_redcap_output(work, source, DEPLOYMENT)
            rows = json.loads((report / "output.json").read_text())
            self.assertEqual([r["field_name"] for r in rows], PR2MASK_COLUMNS)
            self.assertEqual(
                {tuple(r[k] for k in ROUTING) for r in rows},
                {("SYNTHETIC", "test_arm_1", "pr2mask", "11")},
            )
            self.assertEqual({r["value"] for r in rows}, {""})
            self.assertEqual(dictionary_fields(report), (b"PR2MASK", PR2MASK_COLUMNS))

    def test_mixed_destinations_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, work, report = fixture(Path(tmp))
            rows = json.loads((report / "output.json").read_text())
            rows[-1]["record_id"] = "OTHER"
            (report / "output.json").write_text(json.dumps(rows))
            with self.assertRaisesRegex(ValueError, "Mixed REDCap destinations"):
                write_redcap_output(work, source, DEPLOYMENT)

    def test_live_registry_holds_only_the_dynunet_bundle(self):
        self.assertEqual(
            json.loads(redcap_output.MODEL_INSTANCES_PATH.read_text()),
            {DEPLOYMENT["bundle_sha256"]: 11},
        )

    def test_models_have_distinct_destinations_and_reruns_replace_complete_result(self):
        stored = {}
        other = dict(DEPLOYMENT, bundle_sha256="d" * 64)
        with tempfile.TemporaryDirectory() as tmp:
            registry = Path(tmp) / "registry.json"
            registry.write_text(json.dumps({DEPLOYMENT["bundle_sha256"]: 10, "d" * 64: 11}))
            with patch.object(redcap_output, "MODEL_INSTANCES_PATH", registry):
                for deployment, empty in [(DEPLOYMENT, False), (other, False), (DEPLOYMENT, True)]:
                    with tempfile.TemporaryDirectory() as tmp:
                        source, work, report = fixture(Path(tmp), empty=empty)
                        write_redcap_output(work, source, deployment)
                        for row in json.loads((report / "output.json").read_text()):
                            key = tuple(row[k] for k in ROUTING) + (row["field_name"],)
                            stored[key] = row["value"]
        by_instance = {}
        for key, value in stored.items():
            by_instance.setdefault(key[3], {})[key[4]] = value
        self.assertEqual(set(by_instance), {"10", "11"})
        self.assertEqual(by_instance["10"], dict.fromkeys(PR2MASK_COLUMNS, ""))
        self.assertEqual(by_instance["11"]["physical_size"], "50.2")

    def test_unknown_and_colliding_registry_entries_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unregistered"):
            model_repeat_instance("f" * 64)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "registry.json"
            for registry in ({"a" * 64: 10, "b" * 64: 10}, {"a" * 64: 9}):
                with self.subTest(registry=registry):
                    path.write_text(json.dumps(registry))
                    with patch.object(redcap_output, "MODEL_INSTANCES_PATH", path):
                        with self.assertRaisesRegex(ValueError, "Invalid"):
                            model_repeat_instance("a" * 64)


if __name__ == "__main__":
    unittest.main()
