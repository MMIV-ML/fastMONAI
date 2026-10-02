"""Native series selection and historical contour conversion regression checks."""

import json
import zipfile
from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk

from workflow.data_conversion import find_t1_series, rasterize_t1, read_contours


def test_t1_selection_uses_dicom_metadata_not_folder_names(tmp_path, monkeypatch):
    case = tmp_path / "Vestibular-Schwannoma-SEG/VS-SEG-001"
    headers = {}
    for folder, modality, description in [
        ("a", "RTDOSE", "t1_rt"),
        ("b", "MR", "t2_ciss"),
        ("c", "MR", "t1_mpr"),
    ]:
        file = case / folder / "one.dcm"
        file.parent.mkdir(parents=True)
        file.touch()
        headers[str(file)] = {
            "Modality": modality,
            "SeriesDescription": description,
            "PatientID": "VS-SEG-001",
        }
    monkeypatch.setattr(
        "workflow.data_conversion.sitk.ImageSeriesReader.GetGDCMSeriesIDs",
        lambda _: ["uid"],
    )
    monkeypatch.setattr(
        "workflow.data_conversion.sitk.ImageSeriesReader.GetGDCMSeriesFileNames",
        lambda p, _: [str(next(Path(p).glob("*.dcm")))],
    )
    monkeypatch.setattr(
        "workflow.data_conversion.pydicom.dcmread", lambda p, **_: headers[p]
    )
    assert find_t1_series(tmp_path, "vs_gk_1") == [str(case / "c/one.dcm")]
    headers[str(case / "b/one.dcm")]["SeriesDescription"] = "t1_other"
    with pytest.raises(ValueError, match="duplicated T1|expected one T1"):
        find_t1_series(tmp_path, "vs_gk_1")


def test_wrong_patient_is_rejected(tmp_path, monkeypatch):
    folder = tmp_path / "VS-SEG-001"
    folder.mkdir()
    file = folder / "one.dcm"
    file.touch()
    monkeypatch.setattr(
        "workflow.data_conversion.sitk.ImageSeriesReader.GetGDCMSeriesIDs",
        lambda _: ["uid"],
    )
    monkeypatch.setattr(
        "workflow.data_conversion.sitk.ImageSeriesReader.GetGDCMSeriesFileNames",
        lambda *_: [str(file)],
    )
    monkeypatch.setattr(
        "workflow.data_conversion.pydicom.dcmread",
        lambda *a, **k: {
            "Modality": "MR",
            "SeriesDescription": "t1_mpr",
            "PatientID": "VS-SEG-002",
        },
    )
    with pytest.raises(ValueError, match="PatientID"):
        find_t1_series(tmp_path, "vs_gk_1")


def test_historical_polygon_fill_and_geometry(tmp_path):
    image = sitk.GetImageFromArray(np.zeros((3, 6, 6), dtype=np.uint16))
    image.SetOrigin((10, 20, 30))
    image.SetSpacing((1, 1, 1.5))
    reference, output = tmp_path / "image.nii.gz", tmp_path / "mask.nii.gz"
    sitk.WriteImage(image, str(reference))
    square = [[11, 21, 31.5], [14, 21, 31.5], [14, 24, 31.5], [11, 24, 31.5]]
    contours = [
        {"structure_name": "TV", "LPS_contour_points": [square]},
        {
            "structure_name": "Cochlea",
            "LPS_contour_points": [[[10, 20, 30], [15, 20, 30], [15, 25, 30]]],
        },
    ]
    rasterize_t1(contours, reference, output)
    result = sitk.ReadImage(str(output))
    values = sitk.GetArrayFromImage(result)
    assert values[1, 2, 2] == 1
    assert values[0].sum() == values[2].sum() == 0
    assert values[1, 0, 0] == 0
    assert result.GetOrigin() == image.GetOrigin()
    assert result.GetSpacing() == image.GetSpacing()
    assert result.GetDirection() == image.GetDirection()


def test_contours_zip_and_directory_are_equivalent_and_duplicates_rejected(tmp_path):
    contours = [{"structure_name": "TV", "LPS_contour_points": []}]
    path = tmp_path / "contours/vs_gk_1_t1/contours.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(contours))
    archive = tmp_path / "contours.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.write(path, "contours/vs_gk_1_t1/contours.json")
    assert (
        read_contours(archive, "vs_gk_1")
        == read_contours(tmp_path, "vs_gk_1")
        == contours
    )
    with zipfile.ZipFile(archive, "a") as z:
        z.write(path, "other/vs_gk_1_t1/contours.json")
    with pytest.raises(ValueError, match="expected one contours"):
        read_contours(archive, "vs_gk_1")
