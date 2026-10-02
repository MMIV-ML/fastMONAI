"""Reliable, restartable acquisition of the upstream datasets."""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import stat
import subprocess
import tempfile
import time
import uuid
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import BinaryIO
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .data_sources import (
    CROSSMODA_2022,
    TCIA_VESTIBULAR_SCHWANNOMA_SEG,
    DatasetSource,
)

DEFAULT_CHUNK_SIZE = 1024 * 1024
DEFAULT_TIMEOUT = 60.0
DEFAULT_RETRIES = 2
TCIA_RETRIEVER_NAME = "TCIA_Data_Retriever"
_CONTENT_RANGE_RE = re.compile(r"bytes ([0-9]+)-([0-9]+)/([0-9]+)", re.IGNORECASE)


class DownloadError(RuntimeError):
    """An upstream asset could not be downloaded or acquired."""


class ChecksumMismatchError(DownloadError):
    """Downloaded bytes do not match the checksum published upstream."""


class UnsafeArchiveError(DownloadError):
    """An archive contains an entry that cannot be extracted safely."""


class RetrieverNotFoundError(DownloadError):
    """The TCIA command-line data retriever is not installed."""


@dataclass(frozen=True, slots=True)
class AcquisitionResult:
    """Paths produced while acquiring one upstream dataset."""

    source: DatasetSource
    assets: tuple[Path, ...]
    data_dir: Path | None


UrlOpener = Callable[..., BinaryIO]
ProcessRunner = Callable[..., subprocess.CompletedProcess[object]]


def _checksum_spec(expected: str) -> tuple[str, str]:
    value = expected.strip().lower()
    if ":" in value:
        algorithm, digest = value.split(":", 1)
    elif len(value) == 32:
        algorithm, digest = "md5", value
    elif len(value) == 64:
        algorithm, digest = "sha256", value
    else:
        raise ValueError(
            "Checksum must be prefixed with 'md5:' or 'sha256:', or have "
            "the corresponding hexadecimal digest length"
        )

    if algorithm not in {"md5", "sha256"}:
        raise ValueError(f"Unsupported checksum algorithm: {algorithm!r}")
    expected_length = 32 if algorithm == "md5" else 64
    if len(digest) != expected_length or any(
        c not in "0123456789abcdef" for c in digest
    ):
        raise ValueError(f"Invalid {algorithm} checksum: {digest!r}")
    return algorithm, digest


def file_checksum(
    path: str | Path, algorithm: str, *, chunk_size: int = DEFAULT_CHUNK_SIZE
) -> str:
    """Calculate an MD5 or SHA-256 digest without loading the file into memory."""

    algorithm = algorithm.lower()
    if algorithm not in {"md5", "sha256"}:
        raise ValueError(f"Unsupported checksum algorithm: {algorithm!r}")
    try:
        digest = hashlib.new(algorithm, usedforsecurity=False)
    except TypeError:  # pragma: no cover - compatibility with older Python builds
        digest = hashlib.new(algorithm)
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_checksum(path: str | Path, expected: str) -> bool:
    """Return whether *path* matches an ``md5:`` or ``sha256:`` digest."""

    algorithm, expected_digest = _checksum_spec(expected)
    return file_checksum(path, algorithm) == expected_digest


def _response_status(response: object) -> int | None:
    status = getattr(response, "status", None)
    if status is not None:
        return int(status)
    getcode = getattr(response, "getcode", None)
    return int(getcode()) if getcode is not None and getcode() is not None else None


def _response_header(response: object, name: str) -> str | None:
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    value = headers.get(name)
    return str(value) if value is not None else None


def _content_length(response: object, url: str) -> int | None:
    raw_value = _response_header(response, "Content-Length")
    if raw_value is None:
        return None
    value = raw_value.strip()
    if re.fullmatch(r"[0-9]+", value) is None:
        raise DownloadError(
            f"Server returned an invalid Content-Length for {url!r}: {raw_value!r}"
        )
    return int(value)


def _resume_range(
    response: object,
    offset: int,
    content_length: int | None,
    url: str,
) -> tuple[int, int] | None:
    if _response_status(response) != 206:
        return None
    if offset <= 0:
        raise DownloadError(
            f"Server returned an unsolicited partial response for {url!r}"
        )

    raw_value = _response_header(response, "Content-Range")
    match = (
        _CONTENT_RANGE_RE.fullmatch(raw_value.strip())
        if raw_value is not None
        else None
    )
    if match is None:
        raise DownloadError(
            f"Server returned an invalid Content-Range for {url!r}: {raw_value!r}"
        )
    start, end, total = (int(value) for value in match.groups())
    if start != offset or end < start or total <= end:
        raise DownloadError(
            f"Server returned an inconsistent Content-Range for {url!r}: {raw_value!r}"
        )

    range_length = end - start + 1
    if content_length is not None and content_length != range_length:
        raise DownloadError(
            f"Content-Length does not match Content-Range for {url!r}: "
            f"{content_length} != {range_length}"
        )
    return range_length, total


def _download_once(
    url: str,
    partial_path: Path,
    *,
    timeout: float,
    chunk_size: int,
    opener: UrlOpener,
) -> None:
    offset = partial_path.stat().st_size if partial_path.exists() else 0
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    request = Request(url, headers=headers)

    try:
        response_context = opener(request, timeout=timeout)
    except HTTPError as error:
        if error.code == 416 and offset:
            # Without a trustworthy remote length, a 416 response does not
            # prove that the partial file is complete. Discard it so the next
            # retry requests the full asset instead of publishing uncertain
            # bytes (especially for assets without a published checksum).
            partial_path.unlink(missing_ok=True)
            raise DownloadError(
                f"Server rejected the resume offset for {url!r}; restarting"
            ) from error
        raise

    with response_context as response:
        content_length = _content_length(response, url)
        resume_range = _resume_range(response, offset, content_length, url)
        expected_response_bytes = (
            resume_range[0] if resume_range is not None else content_length
        )

        # A server is allowed to ignore Range and return 200. Restarting the
        # partial file avoids silently appending a full response to it.
        mode = "ab" if resume_range is not None else "wb"
        response_bytes = 0
        try:
            with partial_path.open(mode) as output:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    response_bytes += len(chunk)
                    if (
                        expected_response_bytes is not None
                        and response_bytes > expected_response_bytes
                    ):
                        raise DownloadError(
                            f"Server sent more bytes than declared for {url!r}: "
                            f"expected {expected_response_bytes}, "
                            f"received at least {response_bytes}"
                        )
                    output.write(chunk)
        except DownloadError:
            if mode == "ab":
                with partial_path.open("r+b") as output:
                    output.truncate(offset)
            else:
                partial_path.unlink(missing_ok=True)
            raise

        if (
            expected_response_bytes is not None
            and response_bytes != expected_response_bytes
        ):
            raise DownloadError(
                f"Server response ended early for {url!r}: "
                f"expected {expected_response_bytes} bytes, received {response_bytes}"
            )
        if resume_range is not None:
            _, total = resume_range
            accumulated = partial_path.stat().st_size
            if accumulated != total:
                raise DownloadError(
                    f"Partial response for {url!r} ended at {accumulated} of {total} bytes"
                )


def download_file(
    url: str,
    destination: str | Path,
    *,
    checksum: str | None = None,
    timeout: float = DEFAULT_TIMEOUT,
    retries: int = DEFAULT_RETRIES,
    force: bool = False,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    opener: UrlOpener | None = None,
) -> Path:
    """Stream *url* to *destination* with restart and integrity support.

    In-progress bytes are stored at ``<destination>.part``. If the server
    honors a Range request they are resumed; if it returns a full 200 response,
    the partial file is safely overwritten. The final name appears only after
    checksum verification and an atomic rename.
    """

    if retries < 0:
        raise ValueError("retries must be non-negative")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial_path = destination.with_name(destination.name + ".part")

    if destination.exists() and not force:
        if checksum is None or verify_checksum(destination, checksum):
            return destination

    if force and partial_path.exists():
        partial_path.unlink()

    open_url = opener or urlopen
    last_error: BaseException | None = None
    for attempt in range(retries + 1):
        try:
            _download_once(
                url,
                partial_path,
                timeout=timeout,
                chunk_size=chunk_size,
                opener=open_url,
            )
            last_error = None
            break
        except (DownloadError, OSError, URLError) as error:
            last_error = error
            if attempt == retries:
                break
            time.sleep(min(2**attempt, 5))

    if last_error is not None:
        raise DownloadError(
            f"Failed to download {url!r} after {retries + 1} attempt(s): {last_error}"
        ) from last_error
    if not partial_path.exists():
        raise DownloadError(f"Download of {url!r} did not create {partial_path}")

    if checksum is not None and not verify_checksum(partial_path, checksum):
        try:
            algorithm, expected_digest = _checksum_spec(checksum)
            actual_digest = file_checksum(partial_path, algorithm)
        finally:
            # A completed but invalid partial cannot be resumed meaningfully.
            partial_path.unlink(missing_ok=True)
        raise ChecksumMismatchError(
            f"Checksum mismatch for {url!r}: expected {algorithm}:"
            f"{expected_digest}, got {algorithm}:{actual_digest}"
        )

    os.replace(partial_path, destination)
    return destination


def download_assets(
    source: DatasetSource,
    destination: str | Path,
    *,
    timeout: float = DEFAULT_TIMEOUT,
    retries: int = DEFAULT_RETRIES,
    force: bool = False,
    opener: UrlOpener | None = None,
) -> tuple[Path, ...]:
    """Download every published asset in *source* into one directory."""

    destination = Path(destination)
    return tuple(
        download_file(
            asset.url,
            destination / asset.filename,
            checksum=asset.checksum,
            timeout=timeout,
            retries=retries,
            force=force,
            opener=opener,
        )
        for asset in source.assets
    )


def _safe_member_path(info: zipfile.ZipInfo) -> PurePosixPath:
    raw_name = info.filename
    if "\x00" in raw_name:
        raise UnsafeArchiveError(f"Archive entry contains a NUL byte: {raw_name!r}")

    # ZIP uses POSIX separators, but treating backslashes as separators also
    # protects extraction on Windows and archives produced by Windows tools.
    normalized_name = raw_name.replace("\\", "/")
    posix_path = PurePosixPath(normalized_name)
    windows_path = PureWindowsPath(raw_name)
    if (
        not normalized_name
        or posix_path.is_absolute()
        or windows_path.is_absolute()
        or windows_path.drive
        or ".." in posix_path.parts
    ):
        raise UnsafeArchiveError(f"Unsafe archive path: {raw_name!r}")

    mode = (info.external_attr >> 16) & 0xFFFF
    file_type = stat.S_IFMT(mode)
    if stat.S_ISLNK(mode):
        raise UnsafeArchiveError(f"Archive entry is a symbolic link: {raw_name!r}")
    if file_type not in {0, stat.S_IFREG, stat.S_IFDIR}:
        raise UnsafeArchiveError(f"Archive entry is not a regular file: {raw_name!r}")
    return posix_path


def safe_extract_zip(
    archive: str | Path,
    destination: str | Path,
    *,
    force: bool = False,
) -> Path:
    """Safely extract a ZIP into an atomically published directory."""

    archive = Path(archive)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.is_symlink() or (destination.exists() and not destination.is_dir()):
        raise FileExistsError(
            f"Extraction destination is not a directory: {destination}"
        )
    if destination.exists() and not force:
        return destination

    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=destination.parent)
    )
    backup: Path | None = None
    try:
        with zipfile.ZipFile(archive) as zip_file:
            members = [(info, _safe_member_path(info)) for info in zip_file.infolist()]
            staging_root = staging.resolve()
            for info, relative_path in members:
                target = staging.joinpath(*relative_path.parts)
                resolved_target = target.resolve()
                if os.path.commonpath((staging_root, resolved_target)) != str(
                    staging_root
                ):
                    raise UnsafeArchiveError(f"Unsafe archive path: {info.filename!r}")
                if info.is_dir() or stat.S_ISDIR((info.external_attr >> 16) & 0xFFFF):
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with (
                    zip_file.open(info) as source_stream,
                    target.open("wb") as output_stream,
                ):
                    shutil.copyfileobj(source_stream, output_stream)

        if destination.exists():
            backup = destination.with_name(
                f".{destination.name}.backup-{uuid.uuid4().hex}"
            )
            os.replace(destination, backup)
        try:
            os.replace(staging, destination)
        except BaseException:
            if backup is not None:
                os.replace(backup, destination)
                backup = None
            raise
        if backup is not None:
            shutil.rmtree(backup)
            backup = None
        return destination
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def acquire_crossmoda(
    destination: str | Path,
    *,
    force: bool = False,
    timeout: float = DEFAULT_TIMEOUT,
    retries: int = DEFAULT_RETRIES,
    opener: UrlOpener | None = None,
) -> AcquisitionResult:
    """Download and safely extract the CrossMoDA 2022 training archive."""

    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    training = CROSSMODA_2022.asset("training")
    archive = download_file(
        training.url,
        destination / training.filename,
        checksum=training.checksum,
        timeout=timeout,
        retries=retries,
        force=force,
        opener=opener,
    )
    data_dir = safe_extract_zip(
        archive,
        destination / "crossmoda2022_training",
        force=force,
    )
    return AcquisitionResult(CROSSMODA_2022, (archive,), data_dir)


def _resolve_retriever(executable: str | Path | None) -> str:
    if executable is None:
        resolved = shutil.which(TCIA_RETRIEVER_NAME)
    else:
        candidate = Path(executable).expanduser()
        resolved = (
            str(candidate)
            if candidate.is_file() and os.access(candidate, os.X_OK)
            else None
        )

    if resolved is None:
        raise RetrieverNotFoundError(
            f"Could not find the {TCIA_RETRIEVER_NAME!r} executable. Install the "
            "current TCIA Data Retriever from "
            "https://github.com/TCIA/data-retriever, pass its path explicitly, or run "
            "in manifest/assets-only mode."
        )
    return resolved


def run_tcia_data_retriever(
    manifest: str | Path,
    destination: str | Path,
    *,
    executable: str | Path | None = None,
    runner: ProcessRunner | None = None,
) -> tuple[str, ...]:
    """Invoke TCIA Data Retriever without using a shell."""

    retriever = _resolve_retriever(executable)
    manifest = Path(manifest)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    command = (
        retriever,
        "--cli",
        "-i",
        str(manifest),
        "-o",
        str(destination),
        "--skip-existing",
        "--directory-mode",
        "descriptive",
        "--accept-data-policy",
    )
    run_process = runner or subprocess.run
    try:
        run_process(list(command), check=True)
    except subprocess.CalledProcessError as error:
        raise DownloadError(
            f"TCIA Data Retriever failed with exit status {error.returncode}"
        ) from error
    return command


def acquire_tcia(
    destination: str | Path,
    *,
    retriever: str | Path | None = None,
    assets_only: bool = False,
    force: bool = False,
    timeout: float = DEFAULT_TIMEOUT,
    retries: int = DEFAULT_RETRIES,
    opener: UrlOpener | None = None,
    runner: ProcessRunner | None = None,
) -> AcquisitionResult:
    """Acquire TCIA's manifest/supporting assets and, optionally, DICOM data."""

    destination = Path(destination)
    assets_dir = destination / "assets"
    assets = download_assets(
        TCIA_VESTIBULAR_SCHWANNOMA_SEG,
        assets_dir,
        timeout=timeout,
        retries=retries,
        force=force,
        opener=opener,
    )
    if assets_only:
        return AcquisitionResult(TCIA_VESTIBULAR_SCHWANNOMA_SEG, assets, None)

    manifest_asset = TCIA_VESTIBULAR_SCHWANNOMA_SEG.asset("manifest")
    manifest_path = assets_dir / manifest_asset.filename
    data_dir = destination / "dicom"
    run_tcia_data_retriever(
        manifest_path,
        data_dir,
        executable=retriever,
        runner=runner,
    )
    return AcquisitionResult(TCIA_VESTIBULAR_SCHWANNOMA_SEG, assets, data_dir)


__all__ = [
    "AcquisitionResult",
    "ChecksumMismatchError",
    "DEFAULT_CHUNK_SIZE",
    "DEFAULT_RETRIES",
    "DEFAULT_TIMEOUT",
    "DownloadError",
    "RetrieverNotFoundError",
    "TCIA_RETRIEVER_NAME",
    "UnsafeArchiveError",
    "acquire_crossmoda",
    "acquire_tcia",
    "download_assets",
    "download_file",
    "file_checksum",
    "run_tcia_data_retriever",
    "safe_extract_zip",
    "verify_checksum",
]
