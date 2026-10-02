from __future__ import annotations

import hashlib
import io
import stat
import subprocess
import zipfile
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError

import pytest

from workflow import data_download as download
from workflow.data_sources import CROSSMODA_2022, TCIA_VESTIBULAR_SCHWANNOMA_SEG


class FakeResponse(io.BytesIO):
    def __init__(
        self, body: bytes, *, status: int, headers: dict[str, str] | None = None
    ):
        super().__init__(body)
        self.status = status
        self.headers = headers or {}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False


class RecordingOpener:
    def __init__(self, *responses: FakeResponse):
        self.responses = list(responses)
        self.requests = []

    def __call__(self, request, *, timeout):
        self.requests.append((request, timeout))
        return self.responses.pop(0)


def digest(data: bytes, algorithm: str = "md5") -> str:
    return f"{algorithm}:{hashlib.new(algorithm, data).hexdigest()}"


def make_zip(path: Path, members: dict[str, bytes]) -> Path:
    with zipfile.ZipFile(path, "w") as archive:
        for name, body in members.items():
            archive.writestr(name, body)
    return path


def test_source_constants_match_official_records():
    assert CROSSMODA_2022.doi == "10.5281/zenodo.6504722"
    assert CROSSMODA_2022.license_name == "CC BY-NC-SA 4.0"
    training = CROSSMODA_2022.asset("training")
    assert training.url == (
        "https://zenodo.org/records/6504722/files/crossmoda2022_training.zip?download=1"
    )
    assert training.checksum == "md5:8d68fcd44eaee6b0a4371ce344e775cd"

    tcia = TCIA_VESTIBULAR_SCHWANNOMA_SEG
    assert tcia.doi == "10.7937/TCIA.9YTJ-5Q73"
    assert tcia.license_name == "CC BY 4.0"
    assert {asset.name for asset in tcia.assets} == {
        "manifest",
        "contours",
        "matrices",
        "modality_mapping",
    }
    root = "https://www.cancerimagingarchive.net/wp-content/uploads/"
    assert tcia.asset("manifest").url == root + (
        "Vestibular-Schwannoma-SEG-Feb-2021-manifest.tcia"
    )
    assert tcia.asset("contours").url == root + (
        "Vestibular-Schwannoma-SEG-contours-Mar-2021.zip"
    )
    assert tcia.asset("matrices").url == root + (
        "Vestibular-Schwannoma-SEG_matrices-Mar-2021.zip"
    )
    assert (
        tcia.asset("matrices").filename
        == "Vestibular-Schwannoma-SEG_matrices-Mar-2021.zip"
    )
    assert (
        tcia.asset("modality_mapping").url == root + "DirectoryNamesMappingModality.csv"
    )


def test_download_resumes_partial_file_when_range_is_honored(tmp_path):
    destination = tmp_path / "asset.bin"
    partial = tmp_path / "asset.bin.part"
    partial.write_bytes(b"first-")
    complete = b"first-second"
    opener = RecordingOpener(
        FakeResponse(
            b"second",
            status=206,
            headers={"Content-Range": "bytes 6-11/12"},
        )
    )

    result = download.download_file(
        "https://example.invalid/asset.bin",
        destination,
        checksum=digest(complete),
        opener=opener,
        retries=0,
    )

    assert result == destination
    assert destination.read_bytes() == complete
    assert not partial.exists()
    request, timeout = opener.requests[0]
    assert request.get_header("Range") == "bytes=6-"
    assert timeout == download.DEFAULT_TIMEOUT


def test_download_restarts_partial_file_when_server_ignores_range(tmp_path):
    destination = tmp_path / "asset.bin"
    partial = tmp_path / "asset.bin.part"
    partial.write_bytes(b"stale-prefix")
    complete = b"complete-response"
    opener = RecordingOpener(FakeResponse(complete, status=200))

    download.download_file(
        "https://example.invalid/asset.bin",
        destination,
        checksum=digest(complete, "sha256"),
        opener=opener,
        retries=0,
    )

    assert destination.read_bytes() == complete
    assert opener.requests[0][0].get_header("Range") == "bytes=12-"


def test_download_restarts_after_server_rejects_resume_range(tmp_path):
    destination = tmp_path / "asset.bin"
    partial = tmp_path / "asset.bin.part"
    partial.write_bytes(b"uncertain-partial")
    requests = []

    def opener(request, *, timeout):
        requests.append(request)
        if len(requests) == 1:
            raise HTTPError(request.full_url, 416, "Range Not Satisfiable", {}, None)
        return FakeResponse(b"complete-response", status=200)

    download.download_file(
        "https://example.invalid/asset.bin",
        destination,
        opener=opener,
        retries=1,
    )

    assert destination.read_bytes() == b"complete-response"
    assert [request.get_header("Range") for request in requests] == ["bytes=17-", None]


def test_download_resumes_after_truncated_full_response(tmp_path):
    destination = tmp_path / "asset.bin"
    opener = RecordingOpener(
        FakeResponse(b"abcd", status=200, headers={"Content-Length": "8"}),
        FakeResponse(
            b"efgh",
            status=206,
            headers={"Content-Length": "4", "Content-Range": "bytes 4-7/8"},
        ),
    )

    download.download_file(
        "https://example.invalid/asset.bin",
        destination,
        opener=opener,
        retries=1,
    )

    assert destination.read_bytes() == b"abcdefgh"
    assert [request.get_header("Range") for request, _ in opener.requests] == [
        None,
        "bytes=4-",
    ]


def test_download_resumes_after_truncated_partial_response(tmp_path):
    destination = tmp_path / "asset.bin"
    destination.with_name("asset.bin.part").write_bytes(b"ab")
    opener = RecordingOpener(
        FakeResponse(
            b"cd",
            status=206,
            headers={"Content-Length": "6", "Content-Range": "bytes 2-7/8"},
        ),
        FakeResponse(
            b"efgh",
            status=206,
            headers={"Content-Length": "4", "Content-Range": "bytes 4-7/8"},
        ),
    )

    download.download_file(
        "https://example.invalid/asset.bin",
        destination,
        opener=opener,
        retries=1,
    )

    assert destination.read_bytes() == b"abcdefgh"
    assert [request.get_header("Range") for request, _ in opener.requests] == [
        "bytes=2-",
        "bytes=4-",
    ]


@pytest.mark.parametrize(
    "response",
    [
        FakeResponse(
            b"ignored", status=200, headers={"Content-Length": "not-a-number"}
        ),
        FakeResponse(
            b"ignored",
            status=206,
            headers={"Content-Length": "5", "Content-Range": "bytes 2-7/8"},
        ),
    ],
)
def test_invalid_length_headers_preserve_existing_partial(tmp_path, response):
    destination = tmp_path / "asset.bin"
    partial = destination.with_name("asset.bin.part")
    partial.write_bytes(b"ab")

    with pytest.raises(download.DownloadError, match="Content-Length"):
        download.download_file(
            "https://example.invalid/asset.bin",
            destination,
            opener=RecordingOpener(response),
            retries=0,
        )

    assert partial.read_bytes() == b"ab"
    assert not destination.exists()


def test_checksum_mismatch_does_not_publish_or_keep_invalid_partial(tmp_path):
    destination = tmp_path / "asset.bin"
    opener = RecordingOpener(FakeResponse(b"wrong", status=200))

    with pytest.raises(download.ChecksumMismatchError, match="Checksum mismatch"):
        download.download_file(
            "https://example.invalid/asset.bin",
            destination,
            checksum=digest(b"expected"),
            opener=opener,
            retries=0,
        )

    assert not destination.exists()
    assert not destination.with_name("asset.bin.part").exists()


def test_verified_cache_is_reused_and_force_redownloads(tmp_path):
    destination = tmp_path / "asset.bin"
    destination.write_bytes(b"cached")

    def should_not_open(*args, **kwargs):
        raise AssertionError("verified cache should not make a request")

    assert (
        download.download_file(
            "https://example.invalid/asset.bin",
            destination,
            checksum=digest(b"cached"),
            opener=should_not_open,
        )
        == destination
    )

    opener = RecordingOpener(FakeResponse(b"fresh", status=200))
    download.download_file(
        "https://example.invalid/asset.bin",
        destination,
        checksum=digest(b"fresh"),
        opener=opener,
        force=True,
        retries=0,
    )
    assert destination.read_bytes() == b"fresh"
    assert opener.requests[0][0].get_header("Range") is None


@pytest.mark.parametrize("member", ["../escape.txt", "/absolute.txt", "C:\\escape.txt"])
def test_safe_zip_extraction_rejects_traversal_and_absolute_paths(tmp_path, member):
    archive = make_zip(tmp_path / "unsafe.zip", {member: b"unsafe"})
    destination = tmp_path / "output"

    with pytest.raises(download.UnsafeArchiveError, match="Unsafe archive path"):
        download.safe_extract_zip(archive, destination)

    assert not destination.exists()
    assert not (tmp_path / "escape.txt").exists()


def test_safe_zip_extraction_rejects_symbolic_links(tmp_path):
    archive_path = tmp_path / "symlink.zip"
    link = zipfile.ZipInfo("link")
    link.create_system = 3
    link.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(link, "target")

    with pytest.raises(download.UnsafeArchiveError, match="symbolic link"):
        download.safe_extract_zip(archive_path, tmp_path / "output")


def test_safe_zip_extraction_publishes_complete_tree_and_keeps_siblings(tmp_path):
    archive = make_zip(
        tmp_path / "safe.zip",
        {"training_source/case.nii.gz": b"image", "metadata.csv": b"case,center\n"},
    )
    sibling = tmp_path / "raw-archive.zip"
    sibling.write_bytes(b"do not delete")

    destination = download.safe_extract_zip(archive, tmp_path / "output")

    assert (destination / "training_source" / "case.nii.gz").read_bytes() == b"image"
    assert (destination / "metadata.csv").read_bytes() == b"case,center\n"
    assert sibling.read_bytes() == b"do not delete"
    assert not list(tmp_path.glob(".output.staging-*"))


def test_safe_zip_extraction_rejects_file_destination_even_with_force(tmp_path):
    archive = make_zip(tmp_path / "safe.zip", {"case.nii.gz": b"image"})
    destination = tmp_path / "output"
    destination.write_bytes(b"keep me")

    with pytest.raises(FileExistsError, match="not a directory"):
        download.safe_extract_zip(archive, destination, force=True)

    assert destination.read_bytes() == b"keep me"


def test_safe_zip_extraction_rejects_dangling_destination_symlink(tmp_path):
    archive = make_zip(tmp_path / "safe.zip", {"case.nii.gz": b"image"})
    destination = tmp_path / "output"
    destination.symlink_to(tmp_path / "missing", target_is_directory=True)

    with pytest.raises(FileExistsError, match="not a directory"):
        download.safe_extract_zip(archive, destination, force=True)

    assert destination.is_symlink()


def test_safe_zip_keeps_backup_if_publish_and_rollback_both_fail(tmp_path, monkeypatch):
    archive = make_zip(tmp_path / "safe.zip", {"case.nii.gz": b"new"})
    destination = tmp_path / "output"
    destination.mkdir()
    (destination / "sentinel.txt").write_bytes(b"original")
    real_replace = download.os.replace

    def fail_publish_and_rollback(source, target):
        source = Path(source)
        target = Path(target)
        if source.name.startswith(".output.staging-") and target == destination:
            raise OSError("publish failed")
        if source.name.startswith(".output.backup-") and target == destination:
            raise OSError("rollback failed")
        return real_replace(source, target)

    monkeypatch.setattr(download.os, "replace", fail_publish_and_rollback)
    with pytest.raises(OSError, match="rollback failed"):
        download.safe_extract_zip(archive, destination, force=True)

    assert not destination.exists()
    backups = list(tmp_path.glob(".output.backup-*"))
    assert len(backups) == 1
    assert (backups[0] / "sentinel.txt").read_bytes() == b"original"


def test_crossmoda_acquisition_extracts_downloaded_archive(tmp_path, monkeypatch):
    archive = make_zip(
        tmp_path / "fixture.zip", {"training_source/case.nii.gz": b"data"}
    )

    def fake_download(url, destination, **kwargs):
        destination = Path(destination)
        destination.write_bytes(archive.read_bytes())
        return destination

    monkeypatch.setattr(download, "download_file", fake_download)
    result = download.acquire_crossmoda(tmp_path / "crossmoda")

    assert result.source is CROSSMODA_2022
    assert result.assets[0].name == "crossmoda2022_training.zip"
    assert (result.data_dir / "training_source" / "case.nii.gz").read_bytes() == b"data"


def test_tcia_retriever_uses_required_argument_list(tmp_path):
    executable = tmp_path / "TCIA_Data_Retriever"
    executable.write_text("#!/bin/sh\n")
    executable.chmod(0o755)
    manifest = tmp_path / "manifest.tcia"
    manifest.write_text("manifest")
    calls = []

    def runner(command, *, check):
        calls.append((command, check))
        return subprocess.CompletedProcess(command, 0)

    command = download.run_tcia_data_retriever(
        manifest,
        tmp_path / "dicom",
        executable=executable,
        runner=runner,
    )

    assert list(command) == [
        str(executable),
        "--cli",
        "-i",
        str(manifest),
        "-o",
        str(tmp_path / "dicom"),
        "--skip-existing",
        "--directory-mode",
        "descriptive",
        "--accept-data-policy",
    ]
    assert calls == [(list(command), True)]


def test_tcia_retriever_absence_has_actionable_error(tmp_path):
    with patch("workflow.data_download.shutil.which", return_value=None):
        with pytest.raises(
            download.RetrieverNotFoundError,
            match="manifest/assets-only mode",
        ):
            download.run_tcia_data_retriever(
                tmp_path / "manifest.tcia",
                tmp_path / "dicom",
            )


def test_tcia_assets_only_does_not_resolve_or_run_retriever(tmp_path, monkeypatch):
    expected_assets = tuple(
        tmp_path / asset.filename for asset in TCIA_VESTIBULAR_SCHWANNOMA_SEG.assets
    )

    def fake_download_assets(*args, **kwargs):
        return expected_assets

    monkeypatch.setattr(download, "download_assets", fake_download_assets)
    with patch("workflow.data_download.shutil.which") as which:
        result = download.acquire_tcia(tmp_path, assets_only=True)

    assert result.assets == expected_assets
    assert result.data_dir is None
    which.assert_not_called()
