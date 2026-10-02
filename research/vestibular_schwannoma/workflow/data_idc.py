"""Download only the study's Queen Square T1 series from the public IDC mirror."""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from importlib.metadata import version
from pathlib import Path

from .data_download import DownloadError, file_checksum

IDC_COLLECTION = "vestibular_schwannoma_seg"
IDC_DIRECTORY_TEMPLATE = "%PatientID/%StudyInstanceUID/%Modality_%SeriesInstanceUID"


def create_idc_client():
    """Import IDC only when downloading DICOM; help and other routes stay usable."""

    try:
        from idc_index import IDCClient
    except ImportError as error:
        raise DownloadError(
            "Install the Python IDC downloader with "
            "'pip install -r data/requirements.txt' (includes idc-index). "
            "Data Retriever is not required."
        ) from error
    return IDCClient.client()


def select_t1_series(client, index: Path) -> list[dict]:
    """Require exactly one T1 MR series for each Queen Square case in the CSV."""

    with index.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    case_ids = [r["case_id"] for r in rows if r["case_id"].startswith("vs_gk_")]
    if not case_ids or len(case_ids) != len(set(case_ids)):
        raise ValueError("Study CSV has no Queen Square cases or repeats a case")
    if any(not re.fullmatch(r"vs_gk_[1-9][0-9]*", c) for c in case_ids):
        raise ValueError("Study CSV contains an invalid Queen Square case ID")
    patients = {f"VS-SEG-{int(c.removeprefix('vs_gk_')):03d}": c for c in case_ids}
    metadata = client.sql_query(
        "SELECT PatientID, StudyInstanceUID, SeriesInstanceUID, SeriesDescription, "
        "instanceCount, series_size_MB FROM index "
        f"WHERE collection_id = '{IDC_COLLECTION}' AND Modality = 'MR'"
    )
    series = [
        {"case_id": patients[s["PatientID"]], **s}
        for s in metadata.to_dict("records")
        if s["PatientID"] in patients
        and str(s["SeriesDescription"]).lower().startswith("t1_")
    ]
    counts = Counter(s["case_id"] for s in series)
    mismatches = [f"{c}: {counts[c]} series" for c in case_ids if counts[c] != 1]
    if mismatches:
        raise DownloadError(
            "IDC must provide one T1 series per case: " + ", ".join(mismatches)
        )
    if len({s["SeriesInstanceUID"] for s in series}) != len(series):
        raise DownloadError("IDC returned a repeated T1 SeriesInstanceUID")
    for item in series:
        if (
            any(
                not re.fullmatch(r"[0-9]+(?:\.[0-9]+)+", str(item[key]))
                for key in ("StudyInstanceUID", "SeriesInstanceUID")
            )
            or int(item["instanceCount"]) <= 0
        ):
            raise DownloadError(f"{item['case_id']}: invalid IDC series metadata")
    by_case = {s["case_id"]: s for s in series}
    return [by_case[c] for c in case_ids]


def download_t1_series(
    client, series: list[dict], destination: Path, index: Path
) -> Path:
    """Resume IDC downloads, check instance counts, and record the exact selection."""

    if not series:
        raise ValueError("Cannot download an empty IDC selection")
    data_dir = destination / "dicom"
    data_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema": "vs-idc-download-v1",
        "collection_id": IDC_COLLECTION,
        "idc_version": client.get_idc_version(),
        "idc_index_version": version("idc-index"),
        "index_sha256": file_checksum(index, "sha256"),
        "directory_template": IDC_DIRECTORY_TEMPLATE,
        "source_bucket_location": "aws",
        "download_complete": False,
        "series": series,
    }
    manifest_path = destination / "idc_manifest.json"

    def write_manifest():
        temporary = manifest_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        temporary.replace(manifest_path)

    write_manifest()
    try:
        client.download_dicom_series(
            seriesInstanceUID=[s["SeriesInstanceUID"] for s in series],
            downloadDir=str(data_dir),
            dirTemplate=IDC_DIRECTORY_TEMPLATE,
            use_s5cmd_sync=True,
            show_progress_bar=True,
            quiet=True,
        )
    except (RuntimeError, OSError, ValueError) as error:
        raise DownloadError(f"IDC download failed; rerun to resume: {error}") from error
    for item in series:
        folder = (
            data_dir
            / item["PatientID"]
            / item["StudyInstanceUID"]
            / f"MR_{item['SeriesInstanceUID']}"
        )
        count = sum(1 for _ in folder.glob("*.dcm"))
        if count != int(item["instanceCount"]):
            raise DownloadError(
                f"{item['case_id']}: downloaded {count}/{item['instanceCount']} "
                "DICOM instances; rerun to resume"
            )
    manifest["download_complete"] = True
    write_manifest()
    return data_dir
