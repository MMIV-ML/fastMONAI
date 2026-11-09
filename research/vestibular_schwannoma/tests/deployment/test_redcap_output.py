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
from redcap_output import write_redcap_output, model_instance_base  # noqa: E402
import redcap_output  # noqa: E402
from vestibular_schwannoma.tests.deployment.test_dicom_output import _write_test_image  # noqa: E402

DICTIONARY_HEADER = "Variable / Field Name,Form Name,Field Type,Field Label\n"
DICTIONARY_ROWS = (
    "mask_base64_gzip_compressed,pr2mask,text,mask_base64_gzip_compressed\n"
    "number_of_pixel,pr2mask,text,Number of pixel\n"
    "physical_size,pr2mask,text,Physical size\n"
)
PR2MASK_COLUMNS = ["mask_base64_gzip_compressed", "number_of_pixel", "physical_size"]
COLUMNS = PR2MASK_COLUMNS + ["model_version"]
MODEL_VERSION = "dynunet 5-fold, fastmonai-vsseg:20261110T000000Z"
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


def export(work, source, deployment=DEPLOYMENT):
    write_redcap_output(work, source, deployment, model_version=MODEL_VERSION)


def instances(report):
    """The written rows, and their values per repeat instance."""
    rows = json.loads((report / "output.json").read_text())
    values = {}
    for row in rows:
        values.setdefault(row["redcap_repeat_instance"], {})[row["field_name"]] = row["value"]
    return rows, values


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

    def test_each_region_fills_its_own_instance(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, work, report = fixture(Path(tmp))
            export(work, source)
            rows, values = instances(report)
            self.assertEqual([r["field_name"] for r in rows], COLUMNS * 3)
            self.assertEqual(
                {tuple(r[k] for k in ROUTING) for r in rows},
                {("SYNTHETIC", "test_arm_1", "pr2mask", str(n)) for n in (201, 202, 203)},
            )
            for region, size in [(1, "20"), (2, "50"), (3, "50")]:
                self.assertEqual(
                    values[str(200 + region)],
                    {
                        "mask_base64_gzip_compressed": f"m{region}",
                        "number_of_pixel": size,
                        "physical_size": f"{size}.{region}",
                        "model_version": MODEL_VERSION,
                    },
                )
            self.assertEqual(dictionary_fields(report), (b"PR2MASK", COLUMNS))

    def test_empty_mask_fills_first_instance_with_blank_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, work, report = fixture(Path(tmp), empty=True)
            export(work, source)
            rows, values = instances(report)
            self.assertEqual([r["field_name"] for r in rows], COLUMNS)
            self.assertEqual(
                {tuple(r[k] for k in ROUTING) for r in rows},
                {("SYNTHETIC", "test_arm_1", "pr2mask", "201")},
            )
            self.assertEqual(
                values["201"],
                dict.fromkeys(PR2MASK_COLUMNS, "") | {"model_version": MODEL_VERSION},
            )
            self.assertEqual(dictionary_fields(report), (b"PR2MASK", COLUMNS))

    def test_model_version_from_pr2mask_is_replaced_not_duplicated(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, work, report = fixture(Path(tmp))
            rows = json.loads((report / "output.json").read_text())
            rows.append(dict(rows[0], field_name="model_version", value="pr2mask"))
            (report / "output.json").write_text(json.dumps(rows))
            with zipfile.ZipFile(report / "output_data_dictionary.zip", "w") as z:
                z.writestr("OriginID.txt", "PR2MASK")
                z.writestr(
                    "instrument.csv",
                    DICTIONARY_HEADER + DICTIONARY_ROWS + "model_version,pr2mask,text,Model\n",
                )
            export(work, source)
            rows, values = instances(report)
            self.assertEqual([r["field_name"] for r in rows], COLUMNS * 3)
            self.assertEqual({v["model_version"] for v in values.values()}, {MODEL_VERSION})
            self.assertEqual(dictionary_fields(report), (b"PR2MASK", COLUMNS))

    def test_region_outside_the_block_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, work, report = fixture(Path(tmp))
            rows = json.loads((report / "output.json").read_text())
            rows[-1]["redcap_repeat_instance"] = "100"
            (report / "output.json").write_text(json.dumps(rows))
            with self.assertRaisesRegex(ValueError, "outside"):
                export(work, source)

    def test_model_version_is_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, work, _ = fixture(Path(tmp))
            with self.assertRaisesRegex(ValueError, "model_version"):
                write_redcap_output(work, source, DEPLOYMENT, model_version="")

    def test_mixed_destinations_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, work, report = fixture(Path(tmp))
            rows = json.loads((report / "output.json").read_text())
            rows[-1]["record_id"] = "OTHER"
            (report / "output.json").write_text(json.dumps(rows))
            with self.assertRaisesRegex(ValueError, "Mixed REDCap destinations"):
                export(work, source)

    def test_live_registry_holds_only_the_dynunet_bundle(self):
        self.assertEqual(
            json.loads(redcap_output.MODEL_INSTANCES_PATH.read_text()),
            {DEPLOYMENT["bundle_sha256"]: 200},
        )

    def test_models_have_separate_blocks_and_reruns_replace_their_instances(self):
        stored = {}
        other = dict(DEPLOYMENT, bundle_sha256="d" * 64)
        with tempfile.TemporaryDirectory() as tmp:
            registry = Path(tmp) / "registry.json"
            registry.write_text(json.dumps({DEPLOYMENT["bundle_sha256"]: 100, "d" * 64: 200}))
            with patch.object(redcap_output, "MODEL_INSTANCES_PATH", registry):
                for deployment, empty in [(DEPLOYMENT, False), (other, False), (DEPLOYMENT, True)]:
                    with tempfile.TemporaryDirectory() as tmp:
                        source, work, report = fixture(Path(tmp), empty=empty)
                        export(work, source, deployment)
                        for row in json.loads((report / "output.json").read_text()):
                            key = tuple(row[k] for k in ROUTING) + (row["field_name"],)
                            stored[key] = row["value"]
        by_instance = {}
        for key, value in stored.items():
            by_instance.setdefault(key[3], {})[key[4]] = value
        # A rerun with fewer regions leaves the higher instances as they were.
        self.assertEqual(set(by_instance), {"101", "102", "103", "201", "202", "203"})
        self.assertEqual(
            by_instance["101"],
            dict.fromkeys(PR2MASK_COLUMNS, "") | {"model_version": MODEL_VERSION},
        )
        self.assertEqual(by_instance["202"]["physical_size"], "50.2")

    def test_unknown_and_invalid_registry_entries_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unregistered"):
            model_instance_base("f" * 64)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "registry.json"
            for registry in (
                {"a" * 64: 200, "b" * 64: 200},
                {"a" * 64: 11},
                {"a" * 64: 250},
                {"a" * 64: True},
            ):
                with self.subTest(registry=registry):
                    path.write_text(json.dumps(registry))
                    with patch.object(redcap_output, "MODEL_INSTANCES_PATH", path):
                        with self.assertRaisesRegex(ValueError, "Invalid"):
                            model_instance_base("a" * 64)


if __name__ == "__main__":
    unittest.main()
