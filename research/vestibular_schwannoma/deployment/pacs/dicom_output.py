"""Create deterministic derived DICOM mask and probability series."""

from __future__ import annotations

import hashlib
import uuid
import warnings
from contextlib import contextmanager
from pathlib import Path

import fastMONAI
import numpy as np
from imagedata.series import Series
from pydicom import dcmread
from pydicom.misc import is_dicom
from pydicom.uid import UID, generate_uid

from deployment_hashing import canonical_json, sha256_bytes
from deployment_models import (
    DEPLOYMENT_SCHEMA,
    DICOM_APPLICATION_ID,
    DICOM_OUTPUT_CODES,
    DICOM_UID_DEFAULT_PREFIX,
    DICOM_UID_NAMESPACE,
    MODEL_CONFIGS,
)


SEGMENTATION_MASK = "segmentation_mask"
PROBABILITY_MAP = "probability_map"

_REPRESENTATION_LABELS = {
    SEGMENTATION_MASK: "segmentation mask",
    PROBABILITY_MAP: "probability map",
}
_IMAGE_TYPE_MARKERS = {
    SEGMENTATION_MASK: "MASK",
    PROBABILITY_MAP: "PROBABILITY",
}


@contextmanager
def _suppress_invalid_ui_warnings():
    """Suppress only pydicom's expected warning for nonstandard source UIDs."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"^Invalid value for VR UI:",
            category=UserWarning,
            module=r"^pydicom\.valuerep$",
        )
        yield


def _validate_sha256(value: str, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 hexadecimal digest")
    return value


def _require_source_identifier(value, label: str) -> str:
    text = "" if value is None else str(value)
    if not text.strip():
        raise RuntimeError(f"source {label} is missing")
    return text


def _require_generated_dicom_uid(value, label: str) -> str:
    text = str(value)
    if not UID(text).is_valid:
        raise RuntimeError(f"generated {label} is not a valid DICOM UID: {text!r}")
    return text


_REQUIRED_GEOMETRY = (
    ("PixelSpacing", 2),
    ("ImageOrientationPatient", 6),
    ("ImagePositionPatient", 3),
)


def _is_standard_dicom_uid(value: str) -> bool:
    with _suppress_invalid_ui_warnings():
        return UID(value).is_valid


def _looks_like_image(dataset) -> bool:
    sop_class = str(dataset.get("SOPClassUID", "")).strip()
    if sop_class:
        try:
            uid = UID(sop_class)
            if uid.keyword.endswith("ImageStorage") or uid.name.endswith(
                "Image Storage"
            ):
                return True
        except (TypeError, ValueError):
            pass
    return any(
        keyword in dataset
        for keyword in ("Rows", "Columns", "PixelSpacing", "ImagePositionPatient")
    )


def _required_text(dataset, keyword: str, image_number: int) -> str:
    value = dataset.get(keyword)
    text = "" if value is None else str(value).strip()
    if not text:
        raise RuntimeError(f"DICOM image {image_number} is missing required {keyword}")
    return text


def _required_numbers(
    dataset,
    keyword: str,
    length: int,
    image_number: int,
) -> tuple[float, ...]:
    value = dataset.get(keyword)
    if value is None:
        raise RuntimeError(f"DICOM image {image_number} is missing required {keyword}")
    try:
        values = tuple(float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"DICOM image {image_number} has invalid {keyword}") from exc
    if len(values) != length or not all(np.isfinite(values)):
        raise RuntimeError(f"DICOM image {image_number} has invalid {keyword}")
    return values


def _validate_geometry(datasets) -> None:
    rows = []
    columns = []
    geometry = {keyword: [] for keyword, _ in _REQUIRED_GEOMETRY}
    for image_number, dataset in enumerate(datasets, start=1):
        for keyword, target in (("Rows", rows), ("Columns", columns)):
            value = dataset.get(keyword)
            try:
                number = int(value)
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"DICOM image {image_number} is missing or has invalid {keyword}"
                ) from exc
            if number < 1:
                raise RuntimeError(
                    f"DICOM image {image_number} is missing or has invalid {keyword}"
                )
            target.append(number)
        for keyword, length in _REQUIRED_GEOMETRY:
            geometry[keyword].append(
                _required_numbers(dataset, keyword, length, image_number)
            )

    if len(set(rows)) != 1 or len(set(columns)) != 1:
        raise RuntimeError("DICOM input has inconsistent Rows or Columns")
    for keyword in ("PixelSpacing", "ImageOrientationPatient"):
        reference = geometry[keyword][0]
        if any(
            not np.allclose(value, reference, rtol=1e-5, atol=1e-6)
            for value in geometry[keyword][1:]
        ):
            raise RuntimeError(f"DICOM input has inconsistent {keyword}")

    positions = geometry["ImagePositionPatient"]
    if len(set(positions)) != len(positions):
        raise RuntimeError("DICOM input has duplicate ImagePositionPatient values")

    orientation = np.asarray(geometry["ImageOrientationPatient"][0], dtype=float)
    row_direction = orientation[:3]
    column_direction = orientation[3:]
    normal = np.cross(row_direction, column_direction)
    normal_length = float(np.linalg.norm(normal))
    if (
        not np.isclose(np.linalg.norm(row_direction), 1.0, rtol=1e-4, atol=1e-4)
        or not np.isclose(np.linalg.norm(column_direction), 1.0, rtol=1e-4, atol=1e-4)
        or not np.isclose(np.dot(row_direction, column_direction), 0.0, atol=1e-4)
        or normal_length < 1e-8
    ):
        raise RuntimeError("DICOM input has invalid ImageOrientationPatient")
    normal /= normal_length

    if len(positions) > 1:
        origin = np.asarray(positions[0], dtype=float)
        projected = sorted(
            float(np.dot(np.asarray(position, dtype=float) - origin, normal))
            for position in positions
        )
        if any(
            np.isclose(first, second, rtol=0.0, atol=1e-6)
            for first, second in zip(projected, projected[1:])
        ):
            raise RuntimeError("DICOM input has inconsistent ImagePositionPatient")


def validate_dicom_input(input_dir) -> None:
    """Validate one readable, structurally consistent MR image series."""
    root = Path(input_dir)
    if not root.is_dir():
        raise RuntimeError(f"DICOM input directory does not exist: {root}")

    datasets = []
    with _suppress_invalid_ui_warnings():
        for path in sorted(
            candidate for candidate in root.rglob("*") if candidate.is_file()
        ):
            if not is_dicom(str(path)):
                continue
            try:
                dataset = dcmread(
                    str(path),
                    stop_before_pixels=True,
                    force=False,
                )
            except Exception as exc:
                raise RuntimeError(f"Unable to read DICOM file: {path}") from exc
            if not _looks_like_image(dataset):
                continue
            datasets.append(dataset)

        if not datasets:
            raise RuntimeError("DICOM input contains no readable image DICOM files")

        studies = []
        series = []
        sop_instances = []
        modalities = []
        frames = []
        file_meta_issue = False
        nonstandard_uid = False
        for image_number, dataset in enumerate(datasets, start=1):
            study_uid = _required_text(dataset, "StudyInstanceUID", image_number)
            series_uid = _required_text(dataset, "SeriesInstanceUID", image_number)
            sop_uid = _required_text(dataset, "SOPInstanceUID", image_number)
            modality = _required_text(dataset, "Modality", image_number).upper()
            studies.append(study_uid)
            series.append(series_uid)
            sop_instances.append(sop_uid)
            modalities.append(modality)

            for value in (study_uid, series_uid, sop_uid):
                nonstandard_uid = nonstandard_uid or not _is_standard_dicom_uid(value)

            frame_value = dataset.get("FrameOfReferenceUID")
            frame_uid = "" if frame_value is None else str(frame_value).strip()
            frames.append(frame_uid)
            if frame_uid and not _is_standard_dicom_uid(frame_uid):
                nonstandard_uid = True

            file_meta = getattr(dataset, "file_meta", None)
            meta_value = (
                None
                if file_meta is None
                else file_meta.get("MediaStorageSOPInstanceUID")
            )
            meta_uid = "" if meta_value is None else str(meta_value).strip()
            if (
                not meta_uid
                or meta_uid != sop_uid
                or not _is_standard_dicom_uid(meta_uid)
            ):
                file_meta_issue = True

        _validate_geometry(datasets)

    if len(set(studies)) != 1:
        raise RuntimeError("DICOM input contains multiple StudyInstanceUID values")
    if len(set(series)) != 1:
        raise RuntimeError("DICOM input contains multiple SeriesInstanceUID values")
    if len(set(sop_instances)) != len(sop_instances):
        raise RuntimeError("DICOM input contains duplicate SOPInstanceUID values")
    if len(set(modalities)) != 1 or modalities[0] != "MR":
        raise RuntimeError(
            "DICOM input must contain one MR modality series; "
            f"found {sorted(set(modalities))}"
        )

    frame_values = {value for value in frames if value}
    frame_issue = (
        len(frame_values) != 1
        or any(not value for value in frames)
        or any(not _is_standard_dicom_uid(value) for value in frame_values)
    )
    warning_categories = []
    if nonstandard_uid:
        warning_categories.append("nonstandard source UID syntax")
    if file_meta_issue:
        warning_categories.append(
            "file-meta SOP identity missing, invalid, or mismatched"
        )
    if frame_issue:
        warning_categories.append(
            "FrameOfReferenceUID missing, invalid, or inconsistent"
        )
    if warning_categories:
        warnings.warn(
            "DICOM input warnings: " + "; ".join(warning_categories),
            RuntimeWarning,
            stacklevel=2,
        )


def _require_prediction_representation(value: str) -> str:
    if value not in _REPRESENTATION_LABELS:
        raise ValueError(f"unknown prediction representation: {value!r}")
    return value


def _generate_uid(deployment: dict, identity: dict) -> str:
    payload = canonical_json(identity)
    registered_prefix = deployment.get("registered_prefix")
    if registered_prefix is None:
        prefix = DICOM_UID_DEFAULT_PREFIX
        generated = uuid.uuid5(uuid.UUID(DICOM_UID_NAMESPACE), payload)
        uid = f"{prefix}.{generated.int}"
    else:
        prefix = registered_prefix
        uid = str(generate_uid(prefix=f"{prefix}.", entropy_srcs=[payload]))
    if len(uid) > 64 or not UID(uid).is_valid or not uid.startswith(f"{prefix}."):
        raise RuntimeError(f"generated invalid DICOM UID: {uid!r}")
    return uid


def make_derived_series_uid(
    deployment: dict,
    source_series_uid: str,
    prediction_representation: str,
    *,
    source_sop_sequence_sha256: str,
    output_pixels_sha256: str,
    use_tta: bool,
) -> str:
    """Create the deterministic Series Instance UID for a prediction."""
    prediction_representation = _require_prediction_representation(
        prediction_representation
    )
    if not isinstance(use_tta, bool):
        raise ValueError("use_tta must be a Boolean")
    model_type = deployment["model_type"]
    try:
        model_code = MODEL_CONFIGS[model_type]["dicom_model_code"]
        output_code = DICOM_OUTPUT_CODES[prediction_representation]
    except KeyError as exc:
        raise ValueError(
            f"unknown deployed model or prediction representation: {exc}"
        ) from exc
    identity = {
        "schema_version": DEPLOYMENT_SCHEMA,
        "application_id": DICOM_APPLICATION_ID,
        "model_code": model_code,
        "output_code": output_code,
        "bundle_sha256": _validate_sha256(deployment["bundle_sha256"], "bundle digest"),
        "fastmonai_version": str(fastMONAI.__version__),
        "tta": use_tta,
        "source_series_uid": _require_source_identifier(
            source_series_uid, "Series Instance UID"
        ),
        "source_sop_sequence_sha256": _validate_sha256(
            source_sop_sequence_sha256, "source SOP UID sequence digest"
        ),
        "output_pixels_sha256": _validate_sha256(
            output_pixels_sha256, "output pixel digest"
        ),
    }
    return _generate_uid(deployment, identity)


def _make_derived_instance_uid(
    deployment: dict,
    derived_series_uid: str,
    source_sop_uid: str,
) -> str:
    """Derive one SOP Instance UID from its derived series and source SOP."""
    return _generate_uid(
        deployment,
        {
            "schema_version": DEPLOYMENT_SCHEMA,
            "derived_series_uid": _require_generated_dicom_uid(
                derived_series_uid, "Series Instance UID"
            ),
            "source_sop_uid": _require_source_identifier(
                source_sop_uid, "SOP Instance UID"
            ),
        },
    )


def _source_sop_uids(series_obj) -> list[str]:
    values = getattr(series_obj, "SOPInstanceUIDs", None)
    if not isinstance(values, dict):
        raise RuntimeError("source series does not expose per-slice SOP Instance UIDs")
    ordered = []
    for slice_idx in range(series_obj.slices):
        value = values.get((0, slice_idx), values.get(slice_idx))
        if value is None:
            raise RuntimeError(
                f"source series has no SOP Instance UID for slice {slice_idx}"
            )
        ordered.append(_require_source_identifier(value, "SOP Instance UID"))
    if len(set(ordered)) != len(ordered):
        raise RuntimeError("source series contains duplicate SOP Instance UIDs")
    return ordered


def _pixel_array_sha256(data: np.ndarray) -> str:
    array = np.ascontiguousarray(data, dtype=np.dtype("<u2"))
    digest = hashlib.sha256()
    digest.update(
        canonical_json({"dtype": "uint16-le", "shape": list(array.shape)}).encode(
            "utf-8"
        )
    )
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _finalize_written_dicom(save_dir):
    """Synchronize the file-meta SOP UID after imagedata writes."""
    paths = sorted(path for path in Path(save_dir).iterdir() if path.is_file())
    if not paths:
        raise RuntimeError(f"DICOM writer produced no files in {save_dir}")
    with _suppress_invalid_ui_warnings():
        for path in paths:
            dataset = dcmread(str(path))
            series_uid = str(dataset.SeriesInstanceUID)
            sop_uid = str(dataset.SOPInstanceUID)
            if not UID(series_uid).is_valid or not UID(sop_uid).is_valid:
                raise RuntimeError(f"writer produced an invalid DICOM UID in {path}")
            dataset.file_meta.MediaStorageSOPInstanceUID = sop_uid
            dataset.save_as(str(path), write_like_original=False)


def save_series_pred(
    series_obj,
    save_dir,
    deployment,
    prediction_representation,
    *,
    output_pixels_sha256,
    use_tta,
):
    """Save a prediction series with deterministic numeric DICOM UIDs."""
    source_series_uid = _require_source_identifier(
        series_obj.seriesInstanceUID, "Series Instance UID"
    )
    source_sop_uids = _source_sop_uids(series_obj)
    source_sop_sequence_sha256 = sha256_bytes(
        canonical_json(source_sop_uids).encode("utf-8")
    )
    series_uid = make_derived_series_uid(
        deployment,
        source_series_uid,
        prediction_representation,
        source_sop_sequence_sha256=source_sop_sequence_sha256,
        output_pixels_sha256=output_pixels_sha256,
        use_tta=use_tta,
    )
    series_obj.seriesInstanceUID = series_uid
    series_obj.setDicomAttribute("SeriesInstanceUID", series_uid)
    for slice_idx, source_sop_uid in enumerate(source_sop_uids):
        new_uid = _make_derived_instance_uid(deployment, series_uid, source_sop_uid)
        series_obj.setDicomAttribute("SOPInstanceUID", new_uid, slice=slice_idx)
    with _suppress_invalid_ui_warnings():
        series_obj.write(save_dir, opts={"keep_uid": True}, formats=["dicom"])
    _finalize_written_dicom(save_dir)


def _series_description(deployment: dict, prediction_representation: str) -> str:
    representation_label = _REPRESENTATION_LABELS[
        _require_prediction_representation(prediction_representation)
    ]
    count = len(deployment["members"])
    mode = "single model" if count == 1 else f"{count}-model ensemble"
    return (
        f"fastMONAI {MODEL_CONFIGS[deployment['model_type']]['display_name']} "
        f"{mode} {representation_label}"
    )


def _set_derived_metadata(
    series_obj,
    deployment,
    prediction_representation,
    use_tta,
):
    representation_label = _REPRESENTATION_LABELS[
        _require_prediction_representation(prediction_representation)
    ]
    series_obj.setDicomAttribute("SoftwareVersions", build_software_versions())
    image_type = series_obj.getDicomAttribute("ImageType")
    image_type = (
        []
        if image_type is None
        else ([image_type] if isinstance(image_type, str) else list(image_type))
    )
    series_obj.setDicomAttribute(
        "ImageType",
        ["DERIVED", "SECONDARY"]
        + image_type[2:]
        + [_IMAGE_TYPE_MARKERS[prediction_representation]],
    )
    series_obj.setDicomAttribute(
        "SeriesDescription",
        _series_description(deployment, prediction_representation),
    )
    derivation = (
        f"fastMONAI {representation_label}; model={deployment['model_type']}; "
        f"members={len(deployment['members'])}; "
        f"bundle_sha256={deployment['bundle_sha256']}; "
        f"tta={'on' if use_tta else 'off'}"
    )
    if prediction_representation == PROBABILITY_MAP:
        derivation += "; foreground probability = stored uint16 value / 65535"
    series_obj.setDicomAttribute("DerivationDescription", derivation)


def _create_prediction_series(
    pred,
    dicom_input_path,
    output_dir,
    deployment,
    prediction_representation,
    *,
    use_tta,
):
    with _suppress_invalid_ui_warnings():
        series_obj = Series(str(dicom_input_path), opts={"slice_tolerance": 1e-2})
    _set_derived_metadata(series_obj, deployment, prediction_representation, use_tta)
    data = np.transpose(pred.numpy().squeeze(), (-1, 1, 0)).copy()
    if prediction_representation == PROBABILITY_MAP:
        data = np.rint(data * 65535)
    stored_data = data.astype("<u2")
    series_obj[:] = stored_data
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    save_series_pred(
        series_obj,
        str(output_path),
        deployment,
        prediction_representation,
        output_pixels_sha256=_pixel_array_sha256(stored_data),
        use_tta=use_tta,
    )
    return output_path


def create_dicom_mask(
    pred,
    dicom_input_path,
    output_dir,
    deployment,
    *,
    use_tta,
):
    return _create_prediction_series(
        pred,
        dicom_input_path,
        output_dir,
        deployment,
        SEGMENTATION_MASK,
        use_tta=use_tta,
    )


def create_dicom_probability_map(
    pred,
    dicom_input_path,
    output_dir,
    deployment,
    *,
    use_tta,
):
    return _create_prediction_series(
        pred,
        dicom_input_path,
        output_dir,
        deployment,
        PROBABILITY_MAP,
        use_tta=use_tta,
    )


def build_software_versions() -> list[str]:
    return [f"fastMONAI {fastMONAI.__version__}"]


def write_prediction_outputs(
    segmentation,
    foreground_probability,
    dicom_input_path,
    output_dir,
    deployment,
    *,
    use_tta,
):
    """Write the paired segmentation and probability DICOM series."""
    output_path = Path(output_dir)
    mask_path = create_dicom_mask(
        segmentation,
        dicom_input_path,
        output_path / "mask",
        deployment,
        use_tta=use_tta,
    )
    probability_path = create_dicom_probability_map(
        foreground_probability,
        dicom_input_path,
        output_path / "vote_map",
        deployment,
        use_tta=use_tta,
    )
    return mask_path, probability_path
