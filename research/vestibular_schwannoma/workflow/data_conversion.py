"""Reproduce the study's native T1 conversion from TCIA DICOM and JSON.

The contour algorithm is intentionally the historical VS_Seg
``preprocessing/simple_data_conversion.py`` algorithm. Changing its rounding,
polygon filling, or coordinate calculation would change the study labels.
Every output is checked against the frozen study reference by data_preparation.
"""

from __future__ import annotations

import json
import zipfile
from functools import lru_cache
from pathlib import Path

import numpy as np
import pydicom
import SimpleITK as sitk
from matplotlib.path import Path as PolygonPath


TUMOR_NAMES = frozenset(
    [
        "AN",
        "an",
        "Acoustic Neuroma",
        "TV",
        "tv",
        "Tumor",
        "tumour",
        "GTV",
        "Vol 2y",
        "Vol2016",
        "Vol2y",
        "Vol2015",
        "Vol1y",
        "Vol 3y",
        "Vol 1y",
        "Vol2.5y",
        "Vol2017",
        "Vol T2 2y",
        "Vol3y",
        "Vol3m",
        "vol 2y",
        "Vol 2.5y",
        "Vol2.5",
        "Vol 2.5",
        "Vol20mo",
        "Vol2018_CISS",
        "Vol2018",
        "Vol2016CISS",
        "Vol2014",
        "vol 1y",
        "Vol16",
        "Vol1",
        "TvT1",
        "T1+ 13",
        "Rt AN",
        "6m",
        "1y",
    ]
)


def find_t1_series(root: Path, case_id: str) -> list[str]:
    """Find one T1 MR series in classic/descriptive TCIA or organized input."""

    number = int(case_id.removeprefix("vs_gk_"))
    candidates = [
        root / f"VS-SEG-{number:03d}",
        root / "Vestibular-Schwannoma-SEG" / f"VS-SEG-{number:03d}",
        root / f"{case_id}_t1",
    ]
    cases = [p for p in candidates if p.is_dir()]
    if len(cases) != 1:
        raise ValueError(f"{case_id}: expected one TCIA case folder under {root}")
    folders = sorted({p.parent for p in cases[0].rglob("*.dcm")})
    series = {}
    for folder in folders:
        # TCIA stores each series in its own directory. Skip RT/T2 directories
        # before GDCM enumeration, which warns on non-image radiation files.
        first = sorted(folder.glob("*.dcm"))[0]
        header = pydicom.dcmread(
            str(first),
            stop_before_pixels=True,
            specific_tags=["Modality", "SeriesDescription"],
        )
        if header.get("Modality") != "MR" or not str(
            header.get("SeriesDescription", "")
        ).lower().startswith("t1_"):
            continue
        for uid in sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(folder)) or ():
            files = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(folder), uid)
            if not files:
                continue
            header = pydicom.dcmread(
                files[0],
                stop_before_pixels=True,
                specific_tags=["Modality", "SeriesDescription", "PatientID"],
            )
            if header.get("Modality") != "MR":
                continue
            description = str(header.get("SeriesDescription", "")).lower()
            if not description.startswith("t1_"):
                continue
            if str(header.get("PatientID", "")) != f"VS-SEG-{number:03d}":
                raise ValueError(f"{case_id}: DICOM PatientID does not match")
            if uid in series:
                raise ValueError(f"{case_id}: duplicated T1 series {uid}")
            series[uid] = list(files)
    if len(series) != 1:
        raise ValueError(f"{case_id}: expected one T1 MR series, found {len(series)}")
    return next(iter(series.values()))


def convert_t1(root: Path, case_id: str, destination: Path) -> None:
    """Convert the selected native MR series, without registration/resampling."""

    reader = sitk.ImageSeriesReader()
    reader.SetNumberOfThreads(1)
    reader.SetFileNames(find_t1_series(root, case_id))
    image = reader.Execute()
    sitk.WriteImage(image, str(destination))


def read_contours(source: Path, case_id: str) -> list[dict]:
    relative = f"{case_id}_t1/contours.json"
    if source.is_file():
        with zipfile.ZipFile(source) as archive:
            names = [n for n in archive.namelist() if n.endswith(relative)]
            if len(names) != 1:
                raise ValueError(f"{case_id}: expected one contours.json in {source}")
            return json.loads(archive.read(names[0]))
    candidates = [source / relative, source / "contours" / relative]
    matches = [p for p in candidates if p.is_file()]
    if len(matches) != 1:
        raise ValueError(f"{case_id}: expected one contours.json under {source}")
    return json.loads(matches[0].read_text())


@lru_cache(maxsize=4)
def _slice_grid(width: int, height: int) -> np.ndarray:
    x, y = np.meshgrid(np.arange(width), np.arange(height), indexing="xy")
    return np.column_stack([x.ravel(), y.ravel()])


def rasterize_t1(contours: list[dict], reference: Path, destination: Path) -> None:
    """Apply the frozen historical rounding and polygon fill algorithm.

    The historical physical-to-index calculation uses origin and spacing. It
    must not be substituted for a general direction-aware contour converter.
    Its suitability here is established by voxel parity with this study only.
    """

    image = sitk.ReadImage(str(reference))
    size = image.GetSize()
    origin = np.asarray(image.GetOrigin())
    spacing = np.asarray(image.GetSpacing())
    mask = np.zeros((size[2], size[1], size[0]), dtype=np.uint8)
    for structure in contours:
        if structure.get("structure_name") not in TUMOR_NAMES:
            continue
        for points in structure.get("LPS_contour_points", []):
            if len(points) < 3:
                continue
            indices = np.round((np.asarray(points) - origin) / spacing).astype(int)
            z = int(np.median(indices[:, 2]))
            if 0 <= z < size[2]:
                inside = (
                    PolygonPath(indices[:, :2])
                    .contains_points(_slice_grid(size[0], size[1]))
                    .reshape(size[1], size[0])
                )
                mask[z] = np.logical_or(mask[z], inside)
    output = sitk.GetImageFromArray(mask)
    output.CopyInformation(image)
    sitk.WriteImage(output, str(destination))
