"""Data-integrity checks using synthetic public-ID fixtures, without real scans."""

import csv
import json
import zipfile

import nibabel as nib
import numpy as np
import pytest

from scripts.prepare_data import main
from scripts import prepare_data as cli
from workflow.data_preparation import (
    COMPONENT_SIGNATURES,
    build_label_bundle,
    case_paths,
    clean_components,
    describe_nifti,
    freeze_reference,
    integer_labels,
    prepare_dataset,
    verify_dataset,
    write_json,
)


def write_nifti(path, data, *, private_text=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    image = nib.Nifti1Image(data, np.diag([-1.0, -1.0, 1.5, 1.0]))
    image.header.set_xyzt_units("mm")
    if private_text:
        image.header["descrip"] = b"private annotation editor information"
        image.header.extensions.append(
            nib.nifti1.Nifti1Extension(6, b"private extension")
        )
    nib.save(image, path)


@pytest.fixture
def study(tmp_path):
    reference_root = tmp_path / "reference"
    index = tmp_path / "project/data/ml_dataset.csv"
    index.parent.mkdir(parents=True)
    rows = []
    for case, fold in [("vs_gk_4", "1"), ("crossmoda2022_etz_0", "2")]:
        image_path, mask_path = case_paths(case)
        image = np.arange(60, dtype=np.uint16).reshape(3, 4, 5)
        mask = np.zeros_like(image, dtype=np.uint8)
        mask[1, 1, 1:4] = 1
        write_nifti(reference_root / image_path, image)
        write_nifti(reference_root / mask_path, mask, private_text=True)
        rows.append(
            {
                "case_id": case,
                "t1_img_path": "../nii_data/" + image_path.as_posix(),
                "t1_seg_path": "../nii_data/" + mask_path.as_posix(),
                "fold": fold,
                "quartile_label": "0",
                "volume_mm3": "4.5",
            }
        )
    with index.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    frozen = index.parent / "reference_dataset.json"
    write_json(frozen, freeze_reference(index, reference_root))
    crossmoda = tmp_path / "crossmoda"
    crossmoda.mkdir()
    with (crossmoda / "infos_source_training.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["crossmoda_name", "center", "group"])
        writer.writerow(["crossmoda2022_etz_0", "Tilburg", "training_source"])
        writer.writerow(["crossmoda2021_ldn_1", "London", "training_source"])
    image_path, mask_path = case_paths("crossmoda2022_etz_0")
    raw_image = nib.load(reference_root / image_path)
    raw_mask = np.asanyarray(nib.load(reference_root / mask_path).dataobj).copy()
    raw_mask[0, 0, 0] = 2  # Cochlea must never become tumour foreground.
    write_nifti(
        crossmoda / "training_source" / image_path.name,
        np.asanyarray(raw_image.dataobj),
    )
    write_nifti(crossmoda / "training_source" / mask_path.name, raw_mask)
    corrections = tmp_path / "corrections"
    _, relative = case_paths("vs_gk_4")
    write_nifti(
        corrections / relative.name,
        np.asanyarray(nib.load(reference_root / relative).dataobj),
    )
    return index, frozen, reference_root, crossmoda, corrections


def prepare(study, output):
    index, frozen, root, crossmoda, corrections = study
    return prepare_dataset(
        index,
        frozen,
        output,
        queen_square=root / "queen_square_data",
        queen_square_nifti=True,
        crossmoda=crossmoda,
        corrections=corrections,
    )


def test_prepare_applies_overlay_removes_cochlea_preserves_sources_and_folds(
    study, tmp_path
):
    index, frozen, root, crossmoda, _ = study
    source = root / case_paths("vs_gk_4")[1]
    before = source.read_bytes()
    # The prepared Queen Square input can contain an incorrect original mask.
    bad = nib.load(source)
    values = np.asanyarray(bad.dataobj).copy()
    values[0, 0, 0] = 1
    write_nifti(source, values)
    original = source.read_bytes()
    output = tmp_path / "prepared"
    result = prepare(study, output)
    assert source.read_bytes() == original != before
    assert result["counts"]["cases"] == 2
    assert result["cases"][0]["human_corrected"]
    assert verify_dataset(index, frozen, output)["verified"]
    _, mask = case_paths("crossmoda2022_etz_0")
    assert nib.load(crossmoda / "training_source" / mask.name).get_fdata()[0, 0, 0] == 2
    assert nib.load(output / mask).get_fdata()[0, 0, 0] == 0
    assert not (output / "tilburg_data/crossmoda2021_ldn_1").exists()


def test_reference_checks_voxels_not_compression_or_mask_storage_dtype(study):
    index, frozen, root, _, _ = study
    path = root / case_paths("vs_gk_4")[1]
    image = nib.load(path)
    write_nifti(path, np.asanyarray(image.dataobj).astype(np.int16))
    assert verify_dataset(index, frozen, root)["verified"]
    values = image.get_fdata()
    values[0, 0, 0] = 1
    write_nifti(path, values.astype(np.uint8))
    with pytest.raises(ValueError, match="volume|voxels"):
        verify_dataset(index, frozen, root)


def test_geometry_drift_is_rejected(study):
    index, frozen, root, _, _ = study
    path = root / case_paths("vs_gk_4")[0]
    image = nib.load(path)
    affine = image.affine.copy()
    affine[0, 3] += 2
    nib.save(nib.Nifti1Image(np.asanyarray(image.dataobj), affine), path)
    with pytest.raises(ValueError, match="geometry|affine"):
        verify_dataset(index, frozen, root)


def test_failed_overlay_does_not_publish_partial_dataset(study, tmp_path):
    *_, corrections = study
    path = next(corrections.glob("*.nii.gz"))
    image = nib.load(path)
    values = np.asanyarray(image.dataobj).copy()
    values[0, 0, 0] = 1
    write_nifti(path, values)
    output = tmp_path / "prepared"
    with pytest.raises(ValueError, match="corrected mask voxels"):
        prepare(study, output)
    assert not output.exists()
    assert not list(tmp_path.glob(".prepared.staging-*"))


def test_existing_output_is_refused(study, tmp_path):
    output = tmp_path / "prepared"
    output.mkdir()
    marker = output / "keep"
    marker.write_text("existing data")
    with pytest.raises(FileExistsError):
        prepare(study, output)
    assert marker.read_text() == "existing data"


def test_csv_changes_are_rejected(study):
    index, frozen, root, _, _ = study
    index.write_text(index.read_text().replace(",1,0,4.5", ",3,0,4.5"))
    with pytest.raises(ValueError, match="CSV does not match"):
        verify_dataset(index, frozen, root)


def test_exact_inventory_rejects_unindexed_cases_and_backups(study):
    index, frozen, root, _, _ = study
    extra = root / "queen_square_data/vs_gk_73/backup.nii.gz"
    write_nifti(extra, np.ones((2, 2, 2), dtype=np.uint8))
    with pytest.raises(ValueError, match="inventory mismatch"):
        verify_dataset(index, frozen, root)
    assert verify_dataset(index, frozen, root, allow_extra_files=True)["verified"]


def test_verified_cleanup_is_idempotent_and_rejects_unexpected_blobs():
    largest, extra = COMPONENT_SIGNATURES["vs_gk_13"]
    mask = np.zeros((1, 3, largest), dtype=np.uint8)
    mask[0, 0, :] = 1
    mask[0, 2, :extra] = 1
    cleaned, removed = clean_components("vs_gk_13", mask)
    assert removed == extra and cleaned.sum() == largest
    assert clean_components("vs_gk_13", cleaned)[1] == 0
    mask[0, 2, extra] = 1
    with pytest.raises(ValueError, match="unexpected component signature"):
        clean_components("vs_gk_13", mask)
    assert np.array_equal(clean_components("vs_gk_1", mask)[0], mask)


def test_label_quantization_is_normalized_but_fractional_masks_are_rejected():
    assert np.array_equal(integer_labels(np.array([0.0, 0.9999999997671694])), [0, 1])
    with pytest.raises(ValueError, match="fractional"):
        integer_labels(np.array([0.0, 0.5]))


def test_label_bundle_is_deterministic_and_strips_private_headers(study, tmp_path):
    index, frozen, root, crossmoda, _ = study
    first, second = tmp_path / "first.zip", tmp_path / "second.zip"
    one = build_label_bundle(index, frozen, root, first)
    two = build_label_bundle(index, frozen, root, second)
    assert one["sha256"] == two["sha256"]
    with zipfile.ZipFile(first) as archive:
        assert archive.namelist() == [
            "LICENSE.txt",
            "README.txt",
            "case_mapping.csv",
            "labels/vs_gk_4_seg_refT1.nii.gz",
            "manifest.json",
        ]
        archive.extractall(tmp_path / "labels")
    mask = nib.load(tmp_path / "labels/labels/vs_gk_4_seg_refT1.nii.gz")
    assert not bytes(mask.header["descrip"]).strip(b"\0")
    assert not mask.header.extensions
    source = describe_nifti(root / case_paths("vs_gk_4")[1], "mask")
    bundled = describe_nifti(
        tmp_path / "labels/labels/vs_gk_4_seg_refT1.nii.gz", "mask"
    )
    assert source["voxel_sha256"] == bundled["voxel_sha256"]
    output = tmp_path / "from_bundle"
    assert prepare_dataset(
        index,
        frozen,
        output,
        queen_square=root / "queen_square_data",
        queen_square_nifti=True,
        crossmoda=crossmoda,
        corrections=first,
    )


def test_download_preflights_retriever_before_network(monkeypatch, capsys):
    monkeypatch.setattr("scripts.prepare_data.shutil.which", lambda _: None)
    assert main(["download", "--tcia-backend", "retriever"]) == 1
    assert "Install TCIA_Data_Retriever" in capsys.readouterr().err


def bundle_in_project(study, monkeypatch):
    index, frozen, root, *_ = study
    project = index.parent.parent
    archive = project / "data/corrected_labels/vs_corrected_labels_v1.zip"
    archive.parent.mkdir()
    build_label_bundle(index, frozen, root, archive)
    monkeypatch.setattr(cli, "PROJECT_ROOT", project)
    return archive


def test_labels_command_extracts_bundled_masks_without_network(
    study, monkeypatch, capsys
):
    archive = bundle_in_project(study, monkeypatch)
    monkeypatch.setattr(
        "workflow.data_download.urlopen", lambda *a, **k: pytest.fail("Network used")
    )
    assert main(["download", "--dataset", "labels"]) == 0
    extracted = cli.PROJECT_ROOT / "data/raw/corrected_labels"
    assert json.loads(capsys.readouterr().out)["corrected_labels"] == str(extracted)
    with zipfile.ZipFile(archive) as bundle:
        name = "labels/vs_gk_4_seg_refT1.nii.gz"
        assert (extracted / name).read_bytes() == bundle.read(name)


def test_corrupt_bundled_archive_is_rejected_before_extraction(
    study, monkeypatch, capsys
):
    archive = bundle_in_project(study, monkeypatch)
    archive.write_bytes(archive.read_bytes()[:-22])
    assert main(["download", "--dataset", "labels"]) == 1
    assert "not a zip file" in capsys.readouterr().err
    assert not (cli.PROJECT_ROOT / "data/raw/corrected_labels").exists()


def test_prepare_uses_bundled_masks_without_separate_label_argument(
    study, monkeypatch, tmp_path, capsys
):
    bundle_in_project(study, monkeypatch)
    _, _, root, crossmoda, _ = study
    output = tmp_path / "from_default_bundle"
    assert (
        main(
            [
                "prepare",
                "--queen-square-nifti",
                str(root / "queen_square_data"),
                "--crossmoda",
                str(crossmoda),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["verified"] is True
    assert verify_dataset(*study[:2], output)["cases"] == 2
