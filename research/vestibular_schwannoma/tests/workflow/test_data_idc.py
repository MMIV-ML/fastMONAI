"""Check cohort selection and incomplete transfers without downloading test data."""

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.prepare_data import main
from workflow import data_idc as idc
from workflow.data_download import DownloadError


def write_index(path, cases):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["case_id"])
        writer.writeheader()
        writer.writerows({"case_id": c} for c in cases)
    return path


def metadata(number, description="t1_test", uid=None):
    return {
        "PatientID": f"VS-SEG-{number:03d}",
        "StudyInstanceUID": f"1.2.{number}",
        "SeriesInstanceUID": uid or f"1.3.{number}",
        "SeriesDescription": description,
        "instanceCount": 2,
        "series_size_MB": 1.0,
    }


class FakeClient:
    def __init__(self, records, *, missing_slice=False):
        self.records = records
        self.missing_slice = missing_slice
        self.requested = []

    def sql_query(self, query):
        return SimpleNamespace(to_dict=lambda _: self.records)

    def get_idc_version(self):
        return "v24"

    def download_dicom_series(self, **kwargs):
        self.requested = kwargs["seriesInstanceUID"]
        assert kwargs["use_s5cmd_sync"]
        for s in self.records:
            if s["SeriesInstanceUID"] not in self.requested:
                continue
            folder = (
                Path(kwargs["downloadDir"])
                / s["PatientID"]
                / s["StudyInstanceUID"]
                / f"MR_{s['SeriesInstanceUID']}"
            )
            folder.mkdir(parents=True, exist_ok=True)
            for i in range(s["instanceCount"] - int(self.missing_slice)):
                (folder / f"{i}.dcm").write_bytes(b"synthetic instance")


def test_selection_uses_csv_order_and_omits_t2_and_excluded_cases(tmp_path):
    index = write_index(
        tmp_path / "index.csv", ["vs_gk_12", "vs_gk_4", "crossmoda2022_etz_0"]
    )
    client = FakeClient(
        [
            metadata(4),
            metadata(12),
            metadata(8),
            metadata(73),
            metadata(131),
            metadata(4, description="t2_test", uid="1.3.4.2"),
        ]
    )
    selected = idc.select_t1_series(client, index)
    assert [s["case_id"] for s in selected] == ["vs_gk_12", "vs_gk_4"]
    assert [s["SeriesInstanceUID"] for s in selected] == ["1.3.12", "1.3.4"]


@pytest.mark.parametrize("records", [[], [metadata(4), metadata(4, uid="1.3.4.2")]])
def test_missing_or_ambiguous_t1_series_aborts_before_transfer(tmp_path, records):
    index = write_index(tmp_path / "index.csv", ["vs_gk_4"])
    with pytest.raises(DownloadError, match="one T1 series per case"):
        idc.select_t1_series(FakeClient(records), index)


def test_two_cases_cannot_share_a_series_uid(tmp_path):
    index = write_index(tmp_path / "index.csv", ["vs_gk_4", "vs_gk_12"])
    with pytest.raises(DownloadError, match="repeated T1 SeriesInstanceUID"):
        idc.select_t1_series(
            FakeClient([metadata(4), metadata(12, uid="1.3.4")]), index
        )


def test_partial_download_remains_incomplete_and_can_resume(tmp_path, monkeypatch):
    monkeypatch.setattr(idc, "version", lambda _: "0.12.5")
    index = write_index(tmp_path / "index.csv", ["vs_gk_4"])
    client = FakeClient([metadata(4)], missing_slice=True)
    selected = idc.select_t1_series(client, index)
    destination = tmp_path / "download"
    with pytest.raises(DownloadError, match="downloaded 1/2"):
        idc.download_t1_series(client, selected, destination, index)
    manifest = destination / "idc_manifest.json"
    assert not json.loads(manifest.read_text())["download_complete"]
    client.missing_slice = False
    assert (
        idc.download_t1_series(client, selected, destination, index)
        == destination / "dicom"
    )
    completed = json.loads(manifest.read_text())
    assert completed["download_complete"]
    assert completed["series"] == selected
    assert completed["index_sha256"] == idc.file_checksum(index, "sha256")


def test_default_cli_checks_idc_dependency_before_network(monkeypatch, capsys):
    def unavailable():
        raise DownloadError("Install idc-index")

    monkeypatch.setattr(idc, "create_idc_client", unavailable)
    monkeypatch.setattr(
        "scripts.prepare_data.acquire_tcia",
        lambda *a, **k: pytest.fail("network called"),
    )
    monkeypatch.setattr(
        "scripts.prepare_data.acquire_crossmoda",
        lambda *a, **k: pytest.fail("network called"),
    )
    assert main(["download"]) == 1
    assert "Install idc-index" in capsys.readouterr().err


def test_assets_only_does_not_import_idc(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(idc, "create_idc_client", lambda: pytest.fail("IDC imported"))
    monkeypatch.setattr(
        "scripts.prepare_data.acquire_tcia",
        lambda *a, **k: SimpleNamespace(data_dir=None, assets=[]),
    )
    assert (
        main(
            [
                "download",
                "--dataset",
                "queen-square",
                "--assets-only",
                "--raw-root",
                str(tmp_path),
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["queen_square_backend"] == "assets-only"
