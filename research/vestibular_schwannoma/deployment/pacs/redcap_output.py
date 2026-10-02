"""Lossless DICOM mask transport in pr2mask REDCap EAV JSON.

Schema v1: distances are millimetres, coordinates are DICOM LPS, and the
array axes are (slice, row, column). Per-slice positions are authoritative.
"""

from __future__ import annotations

import base64
import csv
import gzip
import hashlib
import io
import json
import math
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from pydicom import dcmread
from pydicom.misc import is_dicom

ENCODING = "packbits-gzip-base64-v1"
MAX_VOXELS = 256 * 1024 * 1024
MODEL_INSTANCES_PATH = Path(__file__).with_name("redcap_model_instances.json")


def model_repeat_instance(bundle_sha256):
    """Explicit immutable allocation: never truncate a hash into a REDCap integer."""
    registry = json.loads(MODEL_INSTANCES_PATH.read_text())
    if (
        not isinstance(registry, dict)
        or any(
            len(key) != 64 or any(c not in "0123456789abcdef" for c in key)
            for key in registry
        )
        or any(
            type(n) is not int or not 1 <= n <= 2147483647 for n in registry.values()
        )
        or len(set(registry.values())) != len(registry)
    ):
        raise ValueError("Invalid REDCap model-instance registry")
    if bundle_sha256 not in registry:
        raise ValueError(
            "Unregistered model bundle: allocate a permanent REDCap instance in redcap_model_instances.json"
        )
    return str(registry[bundle_sha256])


FIELDS = {
    "vs_measurements_json": ("notes", "Complete pr2mask region measurements (JSON)"),
    "vs_mask_json": (
        "notes",
        "Lossless binary mask and geometry (schema v1; distances in mm)",
    ),
    "vs_model_type": ("text", "Model type"),
    "vs_bundle_sha256": ("text", "Model bundle SHA-256"),
    "vs_prediction_id": ("text", "Prediction identity SHA-256"),
    "vs_deployment_version": ("text", "Deployment version"),
    "vs_tta": ("text", "Test-time augmentation enabled"),
    "vs_created_at": ("text", "Prediction export time (UTC)"),
}


def _json(value):
    return json.dumps(value, separators=(",", ":"), sort_keys=True, allow_nan=False)


def encode_mask(mask):
    array = np.asarray(mask)
    if array.ndim != 3 or not 0 < array.size <= MAX_VOXELS:
        raise ValueError("Invalid mask dimensions or size")
    if not np.all((array == 0) | (array == 1)):
        raise ValueError("Expected binary DICOM mask")
    array = np.ascontiguousarray(array, dtype=np.uint8)
    packed = np.packbits(array.ravel(), bitorder="little").tobytes()
    return dict(
        schema_version=1,
        encoding=ENCODING,
        shape=list(array.shape),
        axis_order=["slice", "row", "column"],
        flatten_order="C",
        bit_order="little",
        foreground_value=1,
        foreground_voxels=int(array.sum()),
        mask_sha256=hashlib.sha256(array.tobytes()).hexdigest(),
        data=base64.b64encode(gzip.compress(packed, mtime=0)).decode("ascii"),
    )


def decode_mask(payload):
    """Recover uint8 voxels; geometry is in payload['geometry'] (all lengths mm)."""
    expected = dict(
        schema_version=1,
        encoding=ENCODING,
        axis_order=["slice", "row", "column"],
        flatten_order="C",
        bit_order="little",
        foreground_value=1,
    )
    if any(payload.get(k) != v for k, v in expected.items()):
        raise ValueError("Unsupported mask schema")
    shape = payload["shape"]
    if len(shape) != 3 or any(type(n) is not int or n <= 0 for n in shape):
        raise ValueError("Invalid mask shape")
    count = math.prod(shape)
    if count > MAX_VOXELS:
        raise ValueError("Mask exceeds size limit")
    size = (count + 7) // 8
    # Bound both the compressed input and decompressed output.
    if len(payload["data"]) > 4 * (size + 65536):
        raise ValueError("Encoded mask exceeds size limit")
    compressed = base64.b64decode(payload["data"], validate=True)
    with gzip.GzipFile(fileobj=io.BytesIO(compressed)) as stream:
        raw = stream.read(size + 1)
    if len(raw) != size:
        raise ValueError("Mask length mismatch")
    bits = np.unpackbits(np.frombuffer(raw, dtype=np.uint8), bitorder="little")
    if bits[count:].any():
        raise ValueError("Nonzero mask padding")
    array = bits[:count].reshape(shape)
    if hashlib.sha256(array.tobytes()).hexdigest() != payload["mask_sha256"]:
        raise ValueError("Mask checksum mismatch")
    if int(array.sum()) != payload["foreground_voxels"]:
        raise ValueError("Foreground count mismatch")
    return array


def _series(directory, *, pixels=False):
    datasets = [
        dcmread(p, stop_before_pixels=not pixels)
        for p in Path(directory).rglob("*")
        if p.is_file() and is_dicom(p)
    ]
    datasets = [
        d for d in datasets if hasattr(d, "ImagePositionPatient") and hasattr(d, "Rows")
    ]
    if not datasets:
        raise ValueError("No DICOM image slices found")
    first = datasets[0]
    orientation = np.asarray(first.ImageOrientationPatient, dtype=float)
    spacing = np.asarray(first.PixelSpacing, dtype=float)
    if (
        orientation.shape != (6,)
        or spacing.shape != (2,)
        or not np.isfinite(orientation).all()
        or not np.isfinite(spacing).all()
        or (spacing <= 0).any()
    ):
        raise ValueError("Invalid DICOM geometry")
    x, y = orientation[:3], orientation[3:]
    if not (
        np.isclose(np.linalg.norm(x), 1)
        and np.isclose(np.linalg.norm(y), 1)
        and np.isclose(np.dot(x, y), 0, atol=1e-5)
    ):
        raise ValueError("Invalid DICOM orientation")
    for d in datasets:
        if any(
            str(d.get(k, "")) != str(first.get(k, ""))
            for k in (
                "StudyInstanceUID",
                "SeriesInstanceUID",
                "FrameOfReferenceUID",
                "Rows",
                "Columns",
            )
        ):
            raise ValueError("Mixed DICOM series or dimensions")
        if not (
            np.allclose(d.ImageOrientationPatient, orientation)
            and np.allclose(d.PixelSpacing, spacing)
        ):
            raise ValueError("Inconsistent DICOM geometry")
        if not np.isfinite(np.asarray(d.ImagePositionPatient, dtype=float)).all():
            raise ValueError("Invalid slice position")
    normal = np.cross(x, y)
    datasets.sort(key=lambda d: np.dot(d.ImagePositionPatient, normal))
    positions = np.asarray([d.ImagePositionPatient for d in datasets], dtype=float)
    if len(datasets) > 1 and (np.diff(positions @ normal) <= 1e-6).any():
        raise ValueError("Duplicate slice positions")
    return datasets


def build_mask_payload(mask_dir, input_dir, deployment, *, version, use_tta):
    """Read final DICOM pixels and match them to source slices by patient geometry."""
    masks, source = _series(mask_dir, pixels=True), _series(input_dir)
    if len(masks) != len(source):
        raise ValueError("Source and mask slice counts differ")
    for m, s in zip(masks, source):
        if (m.Rows, m.Columns) != (s.Rows, s.Columns):
            raise ValueError("Source and mask dimensions differ")
        for key in ("ImagePositionPatient", "ImageOrientationPatient", "PixelSpacing"):
            if not np.allclose(m.get(key), s.get(key), rtol=0, atol=1e-4):
                raise ValueError(f"Source and mask geometry differ: {key}")
        for key in ("StudyInstanceUID", "FrameOfReferenceUID"):
            if str(m.get(key, "")) != str(s.get(key, "")):
                raise ValueError(f"Source and mask identifiers differ: {key}")
    payload = encode_mask(np.stack([d.pixel_array for d in masks]))
    payload["geometry"] = {
        "coordinate_system": "LPS",
        "pixel_spacing": [float(x) for x in masks[0].PixelSpacing],
        "image_orientation_patient": [
            float(x) for x in masks[0].ImageOrientationPatient
        ],
        "image_positions_patient": [
            [float(x) for x in d.ImagePositionPatient] for d in masks
        ],
    }
    payload["source"] = {
        "study_uid": str(source[0].StudyInstanceUID),
        "series_uid": str(source[0].SeriesInstanceUID),
        "frame_of_reference_uid": str(source[0].get("FrameOfReferenceUID", "")),
        "sop_instance_uids": [str(d.SOPInstanceUID) for d in source],
    }
    payload["mask_series_uid"] = str(masks[0].SeriesInstanceUID)
    payload["model"] = dict(
        model_type=deployment["model_type"],
        bundle_sha256=deployment["bundle_sha256"],
        deployment_version=version,
        tta=bool(use_tta),
        member_ids=[m["member_id"] for m in deployment["members"]],
    )
    payload["prediction_id"] = hashlib.sha256(_json(payload).encode()).hexdigest()
    payload["created_at"] = datetime.now(timezone.utc).isoformat()
    return payload, source[0]


def _extend_dictionary(path):
    with zipfile.ZipFile(path) as archive:
        contents = {name: archive.read(name) for name in archive.namelist()}
    reader = csv.DictReader(io.StringIO(contents["instrument.csv"].decode("utf-8-sig")))
    names = reader.fieldnames
    required = ["Variable / Field Name", "Form Name", "Field Type", "Field Label"]
    if not names or any(k not in names for k in required):
        raise ValueError("Unsupported pr2mask data dictionary")
    rows = list(reader)
    if any(row["Variable / Field Name"] in FIELDS for row in rows):
        raise ValueError("Mask fields already present in data dictionary")
    for field, (kind, label) in FIELDS.items():
        rows.append(dict(zip(required, [field, "pr2mask", kind, label])))
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=names)
    writer.writeheader()
    writer.writerows(rows)
    contents["instrument.csv"] = buffer.getvalue().encode()
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in contents.items():
            archive.writestr(name, data)


def write_redcap_mask(work_dir, input_dir, deployment, *, version, use_tta):
    """Store the complete prediction in its permanently allocated model instance."""
    instance = model_repeat_instance(deployment["bundle_sha256"])
    work_dir = Path(work_dir)
    paths = list((work_dir / "redcap").glob("*/output.json"))
    if len(paths) != 1:
        raise ValueError("Expected exactly one pr2mask REDCap output.json")
    path = paths[0]
    rows = json.loads(path.read_text())
    if not isinstance(rows, list):
        raise ValueError("Expected pr2mask EAV rows")
    payload, source = build_mask_payload(
        work_dir / "mask", input_dir, deployment, version=version, use_tta=use_tta
    )
    destination = dict(
        record_id=str(source.get("PatientID", "")).rstrip(),
        redcap_event_name=str(source.get("ReferringPhysicianName", "")).removeprefix(
            "EventName:"
        ),
        redcap_repeat_instrument="pr2mask",
        redcap_repeat_instance="1",
    )
    if rows:
        destination = {k: rows[0][k] for k in destination}
        destination["redcap_repeat_instance"] = "1"
        if destination["redcap_repeat_instrument"] != "pr2mask":
            raise ValueError("Unexpected REDCap instrument")
        if any(
            any(
                row[k] != destination[k]
                for k in ("record_id", "redcap_event_name", "redcap_repeat_instrument")
            )
            for row in rows
        ):
            raise ValueError("Mixed REDCap destinations")
    if any(row.get("field_name") in FIELDS for row in rows):
        raise ValueError("Mask fields already present in output.json")
    # Region instances belong inside the measurement list, not in model routing.
    measurements = [
        {k: row[k] for k in ("redcap_repeat_instance", "field_name", "value")}
        for row in rows
    ]
    destination["redcap_repeat_instance"] = instance
    values = dict(
        vs_measurements_json=_json(measurements),
        vs_mask_json=_json(payload),
        vs_model_type=deployment["model_type"],
        vs_bundle_sha256=deployment["bundle_sha256"],
        vs_prediction_id=payload["prediction_id"],
        vs_deployment_version=version,
        vs_tta=str(int(use_tta)),
        vs_created_at=payload["created_at"],
    )
    rows = [
        dict(destination, field_name=key, value=value) for key, value in values.items()
    ]
    _extend_dictionary(path.with_name("output_data_dictionary.zip"))
    path.write_text(_json(rows) + "\n")
