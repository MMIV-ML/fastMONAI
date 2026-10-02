#!/usr/bin/env python3
"""Download, prepare, and verify the published 344-case VS study dataset."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from workflow.data_download import (  # noqa: E402
    DownloadError,
    acquire_crossmoda,
    acquire_tcia,
    safe_extract_zip,
)
from workflow.data_sources import TCIA_VESTIBULAR_SCHWANNOMA_SEG  # noqa: E402


def project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    download = commands.add_parser("download", help="download original public sources")
    download.add_argument(
        "--dataset", choices=["all", "queen-square", "tilburg", "labels"], default="all"
    )
    download.add_argument("--raw-root", default="data/raw")
    download.add_argument("--index", default="data/ml_dataset.csv")
    download.add_argument(
        "--tcia-backend",
        choices=["idc", "retriever"],
        help="DICOM downloader (default: idc; --retriever selects retriever)",
    )
    download.add_argument(
        "--retriever", type=Path, help="current TCIA_Data_Retriever executable"
    )
    download.add_argument(
        "--assets-only", action="store_true", help="skip TCIA DICOM retrieval"
    )

    for command, help_text in [
        ("prepare", "prepare and verify only the shared study cases"),
        ("verify", "verify prepared data against the frozen study reference"),
        ("freeze-reference", "maintainer: fingerprint the current reference dataset"),
        ("bundle-labels", "maintainer: package the 17 corrected reference masks"),
    ]:
        sub = commands.add_parser(command, help=help_text)
        sub.add_argument("--index", default="data/ml_dataset.csv")
        sub.add_argument("--reference", default="data/reference_dataset.json")
        if command in {"prepare", "verify", "freeze-reference"}:
            sub.add_argument("--workers", type=positive_int, default=2)
        if command != "prepare":
            sub.add_argument("--data-root", default="../nii_data")
        if command == "prepare":
            sub.add_argument("--raw-root", default="data/raw")
            queen = sub.add_mutually_exclusive_group()
            queen.add_argument("--queen-square", help="TCIA DICOM collection root")
            queen.add_argument(
                "--queen-square-nifti", help="existing native Queen Square NIfTI root"
            )
            sub.add_argument("--contours", help="TCIA JSON contour ZIP or directory")
            sub.add_argument(
                "--crossmoda", help="extracted crossMoDA training archive root"
            )
            sub.add_argument(
                "--corrected-labels", help="17-mask ZIP or local NL mask directory"
            )
            sub.add_argument(
                "--from-existing",
                help="reuse a local VS_Seg checkout's data and NL masks",
            )
            sub.add_argument(
                "--output",
                default="../nii_data",
                help="new dataset root; existing outputs are refused",
            )
        elif command == "verify":
            sub.add_argument(
                "--allow-extra-files",
                action="store_true",
                help="verify indexed pairs in a legacy mixed-data root",
            )
        elif command == "freeze-reference":
            sub.add_argument("--output", default="data/reference_dataset.json")
        else:
            sub.add_argument(
                "--output", default="data/corrected_labels/vs_corrected_labels_v1.zip"
            )
    return root


def progress(number: int, total: int, case_id: str) -> None:
    if number == 1 or number % 10 == 0 or number == total:
        print(f"{number}/{total}: {case_id}", file=sys.stderr, flush=True)


def bundled_labels() -> Path:
    """Check the ZIP's integrity; preparation verifies masks against the reference."""
    archive = PROJECT_ROOT / "data/corrected_labels/vs_corrected_labels_v1.zip"
    with zipfile.ZipFile(archive) as bundle:
        if bad_member := bundle.testzip():
            raise zipfile.BadZipFile(f"Corrupt label archive member: {bad_member}")
    return archive


def _download(args: argparse.Namespace) -> dict:
    raw = project_path(args.raw_root)
    results = {}
    backend = args.tcia_backend or ("retriever" if args.retriever else "idc")
    if args.retriever and backend != "retriever":
        raise ValueError("--retriever requires --tcia-backend retriever")
    retriever = (
        project_path(args.retriever)
        if args.retriever is not None
        else shutil.which("TCIA_Data_Retriever")
    )
    if (
        args.dataset in {"all", "queen-square"}
        and not args.assets_only
        and backend == "retriever"
        and retriever is None
    ):
        raise FileNotFoundError(
            "Install TCIA_Data_Retriever from https://github.com/TCIA/data-retriever "
            "and pass --retriever /path/to/TCIA_Data_Retriever. Use --assets-only "
            "if downloading DICOM through the TCIA graphical app; see data/README.md."
        )
    client = None
    if (
        args.dataset in {"all", "queen-square"}
        and not args.assets_only
        and backend == "idc"
    ):
        from workflow.data_idc import create_idc_client, select_t1_series

        client = create_idc_client()
        series = select_t1_series(client, project_path(args.index))
        print(
            f"IDC: {len(series)} study T1 series, "
            f"{sum(s['series_size_MB'] for s in series) / 1000:.2f} GB; "
            "T2 and radiation-therapy series are omitted.",
            file=sys.stderr,
            flush=True,
        )
    if args.dataset in {"all", "tilburg"}:
        print(
            "Downloading/checking crossMoDA 2022 training archive (7.8 GB)...",
            file=sys.stderr,
            flush=True,
        )
        result = acquire_crossmoda(raw / "crossmoda2022")
        results["crossmoda_training"] = str(result.data_dir)
    if args.dataset in {"all", "queen-square"}:
        print(
            "Downloading/checking TCIA assets and DICOM...", file=sys.stderr, flush=True
        )
        result = acquire_tcia(
            raw / "queen_square",
            retriever=retriever,
            assets_only=args.assets_only or backend == "idc",
        )
        data_dir = result.data_dir
        if client is not None:
            from workflow.data_idc import download_t1_series

            data_dir = download_t1_series(
                client, series, raw / "queen_square", project_path(args.index)
            )
            results["queen_square_idc_manifest"] = str(
                raw / "queen_square/idc_manifest.json"
            )
        results["queen_square_dicom"] = str(data_dir) if data_dir is not None else None
        results["queen_square_backend"] = "assets-only" if args.assets_only else backend
        results["queen_square_assets"] = [str(p) for p in result.assets]
    if args.dataset in {"all", "labels"}:
        archive = bundled_labels()
        results["corrected_labels"] = str(
            safe_extract_zip(archive, raw / "corrected_labels")
        )
    return results


def run(args: argparse.Namespace) -> dict:
    if args.command == "download":
        return _download(args)
    # CLI help and direct URL downloads do not import imaging dependencies.
    from workflow.data_preparation import (
        build_label_bundle,
        freeze_reference,
        prepare_dataset,
        verify_dataset,
        write_json,
    )

    index = project_path(args.index)
    reference = project_path(args.reference)
    if args.command == "freeze-reference":
        output = project_path(args.output)
        if output.exists():
            raise FileExistsError(f"Frozen reference already exists: {output}")
        value = freeze_reference(
            index, project_path(args.data_root), workers=args.workers, progress=progress
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        write_json(output, value)
        return {"reference": str(output), **value["counts"]}
    if args.command == "verify":
        return verify_dataset(
            index,
            reference,
            project_path(args.data_root),
            workers=args.workers,
            progress=progress,
            allow_extra_files=args.allow_extra_files,
        )
    if args.command == "bundle-labels":
        return build_label_bundle(
            index, reference, project_path(args.data_root), project_path(args.output)
        )
    raw = project_path(args.raw_root)
    if args.from_existing:
        if any(
            (
                args.queen_square,
                args.queen_square_nifti,
                args.contours,
                args.crossmoda,
                args.corrected_labels,
            )
        ):
            raise ValueError(
                "Use --from-existing by itself, or pass the individual source options"
            )
        source = project_path(args.from_existing)
        queen = source / "nii_data/queen_square_data"
        crossmoda = source / "crossmoda_data"
        corrections = source / "errors_fixed_nl"
        contours = None
        converted = True
    else:
        queen = (
            project_path(args.queen_square_nifti or args.queen_square)
            if (args.queen_square_nifti or args.queen_square)
            else raw / "queen_square/dicom"
        )
        crossmoda = (
            project_path(args.crossmoda)
            if args.crossmoda
            else raw / "crossmoda2022/crossmoda2022_training"
        )
        corrections = (
            project_path(args.corrected_labels)
            if args.corrected_labels
            else raw / "corrected_labels"
        )
        if not args.corrected_labels and not corrections.exists():
            corrections = bundled_labels()
        contours = (
            project_path(args.contours)
            if args.contours
            else raw
            / "queen_square/assets"
            / TCIA_VESTIBULAR_SCHWANNOMA_SEG.asset("contours").filename
        )
        converted = args.queen_square_nifti is not None
    result = prepare_dataset(
        index,
        reference,
        project_path(args.output),
        queen_square=queen,
        queen_square_nifti=converted,
        crossmoda=crossmoda,
        corrections=corrections,
        contours=contours,
        workers=args.workers,
        progress=progress,
    )
    return {
        "prepared": str(project_path(args.output)),
        "verified": True,
        **result["counts"],
    }


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        result = run(args)
    except (OSError, ValueError, KeyError, DownloadError, zipfile.BadZipFile) as error:
        print(f"Data preparation failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
