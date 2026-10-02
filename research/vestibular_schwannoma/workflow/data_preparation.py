"""Prepare and verify the exact study cohort against its public fingerprints."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import os
import re
import shutil
import tempfile
import zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import Callable

import nibabel as nib
import numpy as np
from scipy import ndimage

from .data_download import file_checksum, safe_extract_zip

SCHEMA = "vs-study-reference-v1"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
HUMAN_CORRECTED_CASES = tuple(
    f"vs_gk_{i}"
    for i in [4, 5, 29, 35, 43, 44, 45, 47, 55, 56, 61, 63, 71, 74, 76, 96, 119]
)
LABEL_CONTRIBUTORS = ("Njål Lura", "Satheshkumar Kaliyugarassan")
LABEL_RELEASE_LICENSE = "CC BY 4.0"
LABEL_RELEASE_LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/"
COMPONENT_SIGNATURES = {
    "vs_gk_12": (6685, 79),
    "vs_gk_13": (2927, 10),
    "vs_gk_53": (2756, 52),
    "vs_gk_86": (3336, 55),
    "vs_gk_87": (4144, 17),
    "vs_gk_250": (24946, 1),
    "crossmoda2022_etz_38": (3616, 1),
    "crossmoda2022_etz_67": (3568, 1),
}
EXCLUSIONS = {
    "vs_gk_73": "Meningioma; excluded in VS_Seg/00c_combined_data.ipynb",
    "vs_gk_8": "Excluded after radiologist review; fastMONAI commit d8bd2de",
    "vs_gk_131": "Excluded after radiologist review; fastMONAI commit d8bd2de",
}
_CASE = re.compile(r"^(vs_gk_[1-9][0-9]*|crossmoda2022_etz_(0|[1-9][0-9]*))$")


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def case_paths(case_id: str) -> tuple[Path, Path]:
    if not _CASE.fullmatch(case_id):
        raise ValueError(f"Invalid study case ID: {case_id!r}")
    if case_id.startswith("vs_gk_"):
        base = Path("queen_square_data") / case_id
        return base / f"{case_id}_t1_refT1.nii.gz", base / f"{case_id}_seg_refT1.nii.gz"
    base = Path("tilburg_data") / case_id
    return base / f"{case_id}_ceT1.nii.gz", base / f"{case_id}_Label.nii.gz"


def load_index(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "case_id",
            "t1_img_path",
            "t1_seg_path",
            "fold",
            "volume_mm3",
            "quartile_label",
        }
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"{path}: missing study CSV columns")
        rows = list(reader)
    if not rows or len({r["case_id"] for r in rows}) != len(rows):
        raise ValueError("Study CSV is empty or has duplicate case IDs")
    for row in rows:
        paths = case_paths(row["case_id"])
        if row["case_id"] in EXCLUSIONS:
            raise ValueError(f"Excluded case in study CSV: {row['case_id']}")
        for field, relative in zip(("t1_img_path", "t1_seg_path"), paths):
            if row[field] != str(PurePosixPath("../nii_data") / relative.as_posix()):
                raise ValueError(f"{row['case_id']}: unexpected canonical CSV path")
        if row["fold"] not in {"1", "2", "3", "4", "5"}:
            raise ValueError(f"{row['case_id']}: invalid fold")
        if row["quartile_label"] not in {"0", "1", "2", "3"}:
            raise ValueError(f"{row['case_id']}: invalid quartile label")
        if not np.isfinite(float(row["volume_mm3"])) or float(row["volume_mm3"]) <= 0:
            raise ValueError(f"{row['case_id']}: invalid tumour volume")
    return rows


def _values(image: nib.Nifti1Image, role: str) -> np.ndarray:
    array = np.asanyarray(image.dataobj)
    if array.ndim != 3:
        raise ValueError(f"Expected 3D {role}, got {array.shape}")
    if role == "mask":
        array = integer_labels(array)
        if not np.all((array == 0) | (array == 1)) or not np.any(array == 1):
            raise ValueError("Tumour mask must contain non-empty binary labels {0, 1}")
        dtype = np.dtype("u1")
    else:
        if not np.isfinite(array).all() or array.min() < 0 or array.max() > 65535:
            raise ValueError("Study image values must be finite and fit uint16")
        if array.dtype.kind == "f" and not np.equal(array, np.floor(array)).all():
            raise ValueError("Study images must have integral intensity values")
        dtype = np.dtype("<u2")
    return np.ascontiguousarray(array, dtype=dtype)


def integer_labels(values: np.ndarray) -> np.ndarray:
    """Recover integer labels from harmless NIfTI scaling quantization.

    103 reference Tilburg masks decode label 1 as 0.9999999997671694.
    Float32 label loading recovers 1. Reject actual fractional annotations.
    """

    if values.dtype.kind == "f":
        rounded = np.rint(values)
        if not np.isfinite(values).all() or not np.allclose(
            values, rounded, rtol=0, atol=1e-6
        ):
            raise ValueError("Mask contains fractional or non-finite labels")
        return rounded
    return values


def describe_nifti(path: Path, role: str) -> dict:
    image = nib.load(path)
    values = _values(image, role)
    result = {
        "shape": list(values.shape),
        "affine": image.affine.tolist(),
        "spacing": [float(v) for v in image.header.get_zooms()[:3]],
        "voxel_sha256": hashlib.sha256(memoryview(values).cast("B")).hexdigest(),
    }
    if role == "mask":
        result["tumour_voxels"] = int(np.count_nonzero(values))
        result["volume_mm3"] = result["tumour_voxels"] * float(
            np.prod(np.asarray(result["spacing"], dtype=np.float64))
        )
    return result


def check_fingerprint(actual: dict, expected: dict, *, case_id: str, role: str) -> None:
    if (
        actual["shape"] != expected["shape"]
        or actual["voxel_sha256"] != expected["voxel_sha256"]
    ):
        raise ValueError(f"{case_id}: {role} voxels differ from the study reference")
    if not np.allclose(actual["affine"], expected["affine"], rtol=0, atol=1e-5):
        raise ValueError(f"{case_id}: {role} affine differs from the study reference")
    if not np.allclose(actual["spacing"], expected["spacing"], rtol=0, atol=1e-6):
        raise ValueError(f"{case_id}: {role} spacing differs from the study reference")


def _describe_pair(row: dict, root: Path) -> dict:
    case_id = row["case_id"]
    image_path, mask_path = case_paths(case_id)
    try:
        image = describe_nifti(root / image_path, "image")
        mask = describe_nifti(root / mask_path, "mask")
    except ValueError as error:
        raise ValueError(f"{case_id}: {error}") from error
    if image["shape"] != mask["shape"] or not np.allclose(
        image["affine"], mask["affine"], rtol=0, atol=5e-4
    ):
        raise ValueError(f"{case_id}: image/mask geometry mismatch")
    if not np.isclose(
        mask["volume_mm3"], float(row["volume_mm3"]), rtol=1e-6, atol=1e-5
    ):
        raise ValueError(f"{case_id}: mask volume disagrees with the committed CSV")
    return {"case_id": case_id, "image": image, "mask": mask}


def _parallel(
    rows: list[dict], function: Callable, workers: int, progress: Callable | None
) -> list:
    if workers < 1:
        raise ValueError("workers must be positive")
    results = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = {pool.submit(function, row): row["case_id"] for row in rows}
        try:
            for number, future in enumerate(as_completed(pending), 1):
                results[pending[future]] = future.result()
                if progress is not None:
                    progress(number, len(rows), pending[future])
        except BaseException:
            for future in pending:
                future.cancel()
            raise
    return [results[row["case_id"]] for row in rows]


def freeze_reference(
    index: Path, root: Path, *, workers: int = 2, progress: Callable | None = None
) -> dict:
    """Read only indexed files; no source pixels, headers, or local paths are published."""

    rows = load_index(index)
    cases = _parallel(rows, lambda row: _describe_pair(row, root), workers, progress)
    ids = {row["case_id"] for row in rows}
    return {
        "schema": SCHEMA,
        "index_sha256": file_checksum(index, "sha256"),
        "canonical_voxels": {
            "image": "little-endian uint16, C order",
            "mask": "integer labels (quantization tolerance 1e-6), uint8, C order",
        },
        "counts": {
            "cases": len(rows),
            "images": len(rows),
            "masks": len(rows),
            "queen_square": sum(c.startswith("vs_gk_") for c in ids),
            "tilburg": sum(c.startswith("crossmoda2022_etz_") for c in ids),
        },
        "fold_counts": dict(sorted(Counter(row["fold"] for row in rows).items())),
        "human_corrected_cases": [
            case for case in HUMAN_CORRECTED_CASES if case in ids
        ],
        "component_cleanup": {
            case: list(sizes)
            for case, sizes in COMPONENT_SIGNATURES.items()
            if case in ids
        },
        "exclusions": EXCLUSIONS,
        "sources": {
            "queen_square": "10.7937/TCIA.9YTJ-5Q73 (version 2)",
            "tilburg": "10.5281/zenodo.6504722 (training_source, center=Tilburg)",
        },
        "conversion": "VS_Seg historical SimpleITK T1 / rounded JSON polygon rasterization v1",
        "cases": cases,
    }


def load_reference(path: Path, index: Path) -> tuple[list[dict], dict]:
    rows = load_index(index)
    reference = json.loads(path.read_text())
    if reference["schema"] != SCHEMA or reference["index_sha256"] != file_checksum(
        index, "sha256"
    ):
        raise ValueError(
            "Study CSV does not match the frozen reference; preserve the shared index"
        )
    if [r["case_id"] for r in rows] != [c["case_id"] for c in reference["cases"]]:
        raise ValueError("Reference case IDs/order do not match the shared CSV")
    return rows, reference


def verify_dataset(
    index: Path,
    reference_path: Path,
    root: Path,
    *,
    workers: int = 2,
    progress: Callable | None = None,
    allow_extra_files: bool = False,
) -> dict:
    rows, reference = load_reference(reference_path, index)
    expected = {case["case_id"]: case for case in reference["cases"]}
    if not allow_extra_files:
        wanted = {p.as_posix() for row in rows for p in case_paths(row["case_id"])}
        present = {
            p.relative_to(root).as_posix() for p in root.rglob("*.nii*") if p.is_file()
        }
        if present != wanted:
            raise ValueError(
                f"Prepared inventory mismatch: missing={sorted(wanted - present)}, extra={sorted(present - wanted)}"
            )

    def check(row: dict) -> dict:
        actual = _describe_pair(row, root)
        for role in ("image", "mask"):
            check_fingerprint(
                actual[role],
                expected[row["case_id"]][role],
                case_id=row["case_id"],
                role=role,
            )
        return actual

    _parallel(rows, check, workers, progress)
    generated = root / "ml_dataset.csv"
    if generated.exists():
        with generated.open(newline="") as handle:
            prepared_rows = list(csv.DictReader(handle))
        if len(prepared_rows) != len(rows):
            raise ValueError("Prepared CSV case count differs from the shared CSV")
        for shared, prepared in zip(rows, prepared_rows):
            for field in shared:
                if field not in {"t1_img_path", "t1_seg_path"} and shared[
                    field
                ] != prepared.get(field):
                    raise ValueError(
                        f"Prepared CSV changed {shared['case_id']}:{field}"
                    )
            for field, relative in zip(
                ("t1_img_path", "t1_seg_path"), case_paths(shared["case_id"])
            ):
                target = Path(prepared[field])
                if not target.is_absolute():
                    target = PROJECT_ROOT / target
                if target.resolve() != (root / relative).resolve():
                    raise ValueError(
                        f"Prepared CSV points outside the verified dataset: {shared['case_id']}"
                    )
    return {
        "verified": True,
        **reference["counts"],
        "fold_counts": reference["fold_counts"],
    }


def clean_components(case_id: str, values: np.ndarray) -> tuple[np.ndarray, int]:
    """Remove only the eight verified blobs, accepting already-cleaned masks."""

    if case_id not in COMPONENT_SIGNATURES:
        return values, 0
    components, _ = ndimage.label(
        values == 1, structure=ndimage.generate_binary_structure(3, 1)
    )
    sizes = np.bincount(components.ravel())[1:]
    actual = tuple(sorted(sizes.tolist(), reverse=True))
    expected = COMPONENT_SIGNATURES[case_id]
    if actual == (expected[0],):
        return values, 0
    if actual != expected:
        raise ValueError(
            f"{case_id}: unexpected component signature {actual}, expected {expected}"
        )
    return (components == int(sizes.argmax() + 1)).astype(np.uint8), expected[1]


def save_binary(
    source: Path, destination: Path, case_id: str, *, crossmoda: bool = False
) -> int:
    image = nib.load(source)
    values = integer_labels(np.asanyarray(image.dataobj))
    if not np.all(np.isin(values, [0, 1, 2] if crossmoda else [0, 1])):
        raise ValueError(f"{case_id}: unsupported source mask labels")
    values, removed = clean_components(case_id, (values == 1).astype(np.uint8))
    output = nib.Nifti1Image(values, image.affine, image.header.copy())
    output.set_data_dtype(np.uint8)
    # Preserve source spatial metadata exactly; nibabel can otherwise recalculate qform.
    for key in (
        "pixdim",
        "qform_code",
        "sform_code",
        "quatern_b",
        "quatern_c",
        "quatern_d",
        "qoffset_x",
        "qoffset_y",
        "qoffset_z",
        "srow_x",
        "srow_y",
        "srow_z",
    ):
        output.header[key] = image.header[key]
    nib.save(output, destination)
    return removed


@contextmanager
def correction_directory(source: Path):
    if source.is_file():
        with tempfile.TemporaryDirectory(prefix="vs-labels-") as temporary:
            yield safe_extract_zip(source, Path(temporary) / "labels")
    elif source.is_dir():
        yield source
    else:
        raise FileNotFoundError(
            f"Corrected labels not found: {source}; see data/README.md"
        )


def corrected_paths(root: Path, reference: dict) -> dict[str, Path]:
    found = {}
    for case_id in reference["human_corrected_cases"]:
        matches = [
            p
            for p in root.rglob(f"{case_id}_*.nii*")
            if p.is_file() and p.name.endswith((".nii", ".nii.gz"))
        ]
        if len(matches) != 1:
            raise ValueError(
                f"{case_id}: expected exactly one corrected mask under {root}, found {len(matches)}"
            )
        expected = next(
            c["mask"] for c in reference["cases"] if c["case_id"] == case_id
        )
        check_fingerprint(
            describe_nifti(matches[0], "mask"),
            expected,
            case_id=case_id,
            role="corrected mask",
        )
        found[case_id] = matches[0]
    return found


def _tilburg_ids(root: Path) -> set[str]:
    with (root / "infos_source_training.csv").open(
        newline="", encoding="utf-8-sig"
    ) as handle:
        reader = csv.DictReader(handle)
        if not {"crossmoda_name", "center", "group"}.issubset(reader.fieldnames or []):
            raise ValueError("CrossMoDA metadata is missing name, center, or group")
        selected = [
            r["crossmoda_name"]
            for r in reader
            if r["center"] == "Tilburg" and r["group"] == "training_source"
        ]
    if len(selected) != len(set(selected)):
        raise ValueError("Duplicate Tilburg cases in metadata")
    return set(selected)


def prepare_dataset(
    index: Path,
    reference_path: Path,
    output: Path,
    *,
    queen_square: Path,
    crossmoda: Path,
    corrections: Path,
    contours: Path | None = None,
    queen_square_nifti: bool = False,
    workers: int = 2,
    progress: Callable | None = None,
) -> dict:
    """Stage only indexed T1 pairs; publish after all 344 pairs match reference."""

    rows, reference = load_reference(reference_path, index)
    output = output.resolve()
    if output.exists():
        raise FileExistsError(
            f"Output already exists: {output}; choose a new directory"
        )
    for source in (queen_square, crossmoda, corrections):
        if output == source.resolve() or output.is_relative_to(source.resolve()):
            raise ValueError("Prepared output must be outside source directories")
    expected_tilburg = {
        r["case_id"] for r in rows if r["case_id"].startswith("crossmoda")
    }
    if _tilburg_ids(crossmoda) != expected_tilburg:
        raise ValueError("Tilburg metadata does not match the shared study case list")
    if not queen_square_nifti and (contours is None or not contours.exists()):
        raise FileNotFoundError(
            "TCIA JSON contours are required for native DICOM preparation"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    expected = {c["case_id"]: c for c in reference["cases"]}
    with correction_directory(corrections) as label_root:
        overlays = corrected_paths(label_root, reference)
        staging = Path(
            tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent)
        )
        try:

            def prepare(row: dict) -> dict:
                case_id = row["case_id"]
                image_relative, mask_relative = case_paths(case_id)
                image_out, mask_out = staging / image_relative, staging / mask_relative
                image_out.parent.mkdir(parents=True)
                human = case_id in overlays
                if case_id.startswith("crossmoda"):
                    source = crossmoda / "training_source"
                    shutil.copy2(source / image_out.name, image_out)
                    mask_source = source / mask_out.name
                elif queen_square_nifti:
                    source = queen_square / case_id
                    shutil.copy2(source / image_out.name, image_out)
                    mask_source = overlays[case_id] if human else source / mask_out.name
                else:
                    from .data_conversion import convert_t1, rasterize_t1, read_contours

                    convert_t1(queen_square, case_id, image_out)
                    if human:
                        mask_source = overlays[case_id]
                    else:
                        rasterize_t1(
                            read_contours(contours, case_id), image_out, mask_out
                        )
                        mask_source = mask_out
                removed = save_binary(
                    mask_source,
                    mask_out,
                    case_id,
                    crossmoda=case_id.startswith("crossmoda"),
                )
                actual = _describe_pair(row, staging)
                for role in ("image", "mask"):
                    check_fingerprint(
                        actual[role],
                        expected[case_id][role],
                        case_id=case_id,
                        role=role,
                    )
                return {
                    "case_id": case_id,
                    "human_corrected": human,
                    "removed_voxels": removed,
                }

            records = _parallel(rows, prepare, workers, progress)
            manifest = {
                "schema": "vs-prepared-dataset-v1",
                "reference_sha256": file_checksum(reference_path, "sha256"),
                "index_sha256": reference["index_sha256"],
                "counts": reference["counts"],
                "sources": reference["sources"],
                "exclusions": reference["exclusions"],
                "verified_against_reference": True,
                "cases": records,
            }
            write_json(staging / "manifest.json", manifest)
            with (staging / "ml_dataset.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                for row in rows:
                    prepared_row = row.copy()
                    for key, relative in zip(
                        ("t1_img_path", "t1_seg_path"), case_paths(row["case_id"])
                    ):
                        prepared_row[key] = os.path.relpath(
                            output / relative, PROJECT_ROOT
                        )
                    writer.writerow(prepared_row)
            if output.exists():
                raise FileExistsError(f"Output appeared during preparation: {output}")
            staging.rename(output)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
    return manifest


def build_label_bundle(
    index: Path, reference_path: Path, root: Path, destination: Path
) -> dict:
    """Package the 17 used labels with fresh headers, deterministic bytes, and provenance."""

    _, reference = load_reference(reference_path, index)
    if destination.exists():
        raise FileExistsError(f"Label bundle already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    entries = {}
    records = []
    for case_id in reference["human_corrected_cases"]:
        _, relative = case_paths(case_id)
        source = root / relative
        actual = describe_nifti(source, "mask")
        expected = next(
            c["mask"] for c in reference["cases"] if c["case_id"] == case_id
        )
        check_fingerprint(actual, expected, case_id=case_id, role="corrected mask")
        loaded = nib.load(source)
        # Fresh headers exclude free-text fields and extensions from local annotations.
        image = nib.Nifti1Image(_values(loaded, "mask"), loaded.affine)
        image.header.set_xyzt_units("mm")
        image.set_qform(loaded.affine, code=1)
        image.set_sform(loaded.affine, code=1)
        payload = gzip.compress(image.to_bytes(), compresslevel=6, mtime=0)
        name = f"labels/{case_id}_seg_refT1.nii.gz"
        entries[name] = payload
        records.append(
            {
                "case_id": case_id,
                "tcia_patient_id": f"VS-SEG-{int(case_id.removeprefix('vs_gk_')):03d}",
                "file": name,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "reference_mask": actual,
            }
        )
    title = (
        f"Replacement tumour segmentation masks for {len(records)} cases "
        "from Vestibular-Schwannoma-SEG"
    )
    provenance = {
        "schema": "vs-corrected-labels-v1",
        "version": 1,
        "title": title,
        "source_doi": "10.7937/TCIA.9YTJ-5Q73",
        "source_license": "CC BY 4.0",
        "release_license": LABEL_RELEASE_LICENSE,
        "release_license_url": LABEL_RELEASE_LICENSE_URL,
        "annotation_set": "NL masks newly drawn for scans with errors in the original annotations",
        "source_role": "MRI images",
        "contributors": list(LABEL_CONTRIBUTORS),
        "associated_paper": {
            "status": "in preparation",
            "doi": None,
            "citation_location": "Dataset record and reproduction repository",
        },
        "index_sha256": reference["index_sha256"],
        "cases": records,
    }
    entries["manifest.json"] = (
        json.dumps(provenance, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode()
    entries["case_mapping.csv"] = (
        "case_id,tcia_patient_id,mask_file\n"
        + "".join(
            f"{r['case_id']},{r['tcia_patient_id']},{r['file']}\n" for r in records
        )
    ).encode()
    entries["LICENSE.txt"] = (
        "Replacement masks and release documentation: CC BY 4.0.\n"
        f"License: {LABEL_RELEASE_LICENSE_URL}\n\n"
        "This license permits redistribution and adaptation, including commercial use,\n"
        "with appropriate credit, a license link, and identification of changes.\n"
        "Consult the linked license for its full terms.\n\n"
        f"Contributors: {', '.join(LABEL_CONTRIBUTORS)}.\n"
        "Image source: Shapey et al. (2021), Vestibular-Schwannoma-SEG version 2,\n"
        "The Cancer Imaging Archive, DOI: 10.7937/TCIA.9YTJ-5Q73 (CC BY 4.0).\n"
        "These masks are new annotations for scans with errors in the original annotations.\n"
    ).encode()
    entries["README.txt"] = (
        f"{title}\nVersion 1.0\n\n"
        f"{len(records)} newly drawn tumour masks for scans whose original annotations contained errors.\n"
        "These are the NL masks used in our study.\n"
        f"Contributors: {', '.join(LABEL_CONTRIBUTORS)}.\n"
        "Labels: 0 background, 1 tumour. Native T1 geometry; no MRI images.\n"
        "case_mapping.csv maps each mask to its original TCIA PatientID.\n"
        "manifest.json records provenance and reference fingerprints.\n\n"
        "Image source: TCIA Vestibular-Schwannoma-SEG version 2, DOI 10.7937/TCIA.9YTJ-5Q73.\n"
        "Shapey et al. (2021), Scientific Data 8, 286.\n"
        "Source paper DOI: 10.1038/s41597-021-01064-w.\n"
        f"License: {LABEL_RELEASE_LICENSE}. See LICENSE.txt.\n\n"
        "Preparation: research/vestibular_schwannoma/data/README.md in\n"
        "https://github.com/MMIV-ML/fastMONAI\n\n"
        "Please cite our accompanying study paper when available (in preparation).\n"
    ).encode()
    descriptor, temporary = tempfile.mkstemp(suffix=".zip", dir=destination.parent)
    os.close(descriptor)
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED) as archive:
            for name, payload in sorted(entries.items()):
                info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                info.external_attr = 0o100644 << 16
                archive.writestr(info, payload)
        os.rename(temporary, destination)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
    return {
        "schema": "vs-corrected-label-asset-v1",
        "version": 1,
        "filename": destination.name,
        "sha256": file_checksum(destination, "sha256"),
        "size_bytes": destination.stat().st_size,
        "url": None,
        "doi": None,
        "cases": reference["human_corrected_cases"],
        "contributors": list(LABEL_CONTRIBUTORS),
        "release_license": LABEL_RELEASE_LICENSE,
        "release_license_url": LABEL_RELEASE_LICENSE_URL,
    }
