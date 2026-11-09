"""pr2mask REDCap EAV rows, one repeat instance of the pr2mask instrument per region."""

from __future__ import annotations

import csv
import io
import json
import zipfile
from pathlib import Path

from pydicom import dcmread
from pydicom.misc import is_dicom

MODEL_INSTANCES_PATH = Path(__file__).with_name("redcap_model_instances.json")
PR2MASK_DICTIONARY_PATH = Path(__file__).with_name("pr2mask_dictionary.csv")
DICTIONARY_REQUIRED = ["Variable / Field Name", "Form Name", "Field Type", "Field Label"]
INSTRUMENT = "pr2mask"
MODEL_VERSION_FIELD = "model_version"
# Each bundle owns a block of 100 instances: pr2mask's region k goes to base + k.
# Instances below 100 are left to the original (manual) mask, pr2mask's own region
# numbering and earlier single-instance models.
REGION_BLOCK = 100
MAX_INSTANCE = 2147483647


def model_instance_base(bundle_sha256):
    """Explicit immutable allocation: never truncate a hash into a REDCap integer."""
    registry = json.loads(MODEL_INSTANCES_PATH.read_text())
    if (
        not isinstance(registry, dict)
        or any(
            len(key) != 64 or any(c not in "0123456789abcdef" for c in key)
            for key in registry
        )
        or any(
            type(n) is not int
            or n < REGION_BLOCK
            or n % REGION_BLOCK
            or n + REGION_BLOCK - 1 > MAX_INSTANCE
            for n in registry.values()
        )
        or len(set(registry.values())) != len(registry)
    ):
        raise ValueError("Invalid REDCap model-instance registry")
    if bundle_sha256 not in registry:
        raise ValueError(
            "Unregistered model bundle: allocate a permanent REDCap instance block in "
            "redcap_model_instances.json"
        )
    return registry[bundle_sha256]


def _json(value):
    return json.dumps(value, separators=(",", ":"), sort_keys=True, allow_nan=False)


def _read_instrument(text):
    reader = csv.DictReader(io.StringIO(text))
    names = reader.fieldnames
    if not names or any(k not in names for k in DICTIONARY_REQUIRED):
        raise ValueError("Unsupported pr2mask data dictionary")
    return names, list(reader)


def pr2mask_dictionary_rows():
    """pr2mask's dictionary rows, taken from a non-empty run of the packaged pr2mask."""
    return _read_instrument(PR2MASK_DICTIONARY_PATH.read_text(encoding="utf-8"))[1]


def _dictionary_columns(path):
    """pr2mask's column names plus model_version, also written to its dictionary."""
    with zipfile.ZipFile(path) as archive:
        contents = {name: archive.read(name) for name in archive.namelist()}
    names, rows = _read_instrument(contents["instrument.csv"].decode("utf-8-sig"))
    if not rows:
        # pr2mask lists its columns from the first region, so an empty mask has none.
        rows = [{k: row.get(k, "") for k in names} for row in pr2mask_dictionary_rows()]
    if all(row["Variable / Field Name"] != MODEL_VERSION_FIELD for row in rows):
        rows.append(
            dict.fromkeys(names, "")
            | {
                "Variable / Field Name": MODEL_VERSION_FIELD,
                "Form Name": INSTRUMENT,
                "Field Type": "text",
                "Field Label": MODEL_VERSION_FIELD,
            }
        )
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=names)
    writer.writeheader()
    writer.writerows(rows)
    contents["instrument.csv"] = buffer.getvalue().encode()
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in contents.items():
            archive.writestr(name, data)
    return [row["Variable / Field Name"] for row in rows]


def _regions(rows):
    """pr2mask's values per region; it numbers regions from 1 as repeat instances."""
    regions = {}
    for row in rows:
        number = int(row["redcap_repeat_instance"])
        if not 1 <= number < REGION_BLOCK:
            raise ValueError(
                f"pr2mask region {number} is outside the {REGION_BLOCK - 1} model instances"
            )
        regions.setdefault(number, {})[row["field_name"]] = row["value"]
    return regions


def _source_dataset(input_dir):
    for path in sorted(Path(input_dir).rglob("*")):
        if path.is_file() and is_dicom(path):
            return dcmread(path, stop_before_pixels=True)
    raise ValueError("No DICOM source images found")


def write_redcap_output(work_dir, input_dir, deployment, *, model_version):
    """Store each region's pr2mask columns in its own instance of the model's block."""
    if not model_version:
        raise ValueError("model_version is required")
    base = model_instance_base(deployment["bundle_sha256"])
    paths = list((Path(work_dir) / "redcap").glob("*/output.json"))
    if len(paths) != 1:
        raise ValueError("Expected exactly one pr2mask REDCap output.json")
    path = paths[0]
    rows = json.loads(path.read_text())
    if not isinstance(rows, list):
        raise ValueError("Expected pr2mask EAV rows")
    keys = ("record_id", "redcap_event_name", "redcap_repeat_instrument")
    if rows:
        destination = {k: rows[0][k] for k in keys}
        if destination["redcap_repeat_instrument"] != INSTRUMENT:
            raise ValueError("Unexpected REDCap instrument")
        if any(row[k] != destination[k] for row in rows for k in keys):
            raise ValueError("Mixed REDCap destinations")
    else:
        # An empty mask has no rows, so route it from the source images like pr2mask.
        source = _source_dataset(input_dir)
        destination = dict(
            record_id=str(source.get("PatientID", "")).rstrip(),
            redcap_event_name=str(source.get("ReferringPhysicianName", "")).removeprefix(
                "EventName:"
            ),
            redcap_repeat_instrument=INSTRUMENT,
        )
    columns = _dictionary_columns(path.with_name("output_data_dictionary.zip"))
    # An empty mask still fills the first instance, with blank columns, to record the run.
    regions = _regions(rows) or {1: {}}
    rows = []
    for number, region in sorted(regions.items()):
        values = region | {MODEL_VERSION_FIELD: model_version}
        instance = str(base + number)
        rows += [
            dict(
                destination,
                redcap_repeat_instance=instance,
                field_name=name,
                value=values.get(name, ""),
            )
            for name in columns
        ]
    path.write_text(_json(rows) + "\n")
