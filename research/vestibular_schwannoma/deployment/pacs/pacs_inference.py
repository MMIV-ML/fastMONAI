#!/usr/bin/env python3
"""Run vestibular schwannoma inference and required PACS postprocessing."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path

import torch
from fastMONAI.vision_all import load_safetensors_model
from fastMONAI.vision_patch import PatchInferenceEngine

from deployment_bundle import describe_deployment as load_deployment
from deployment_models import MODEL_CONFIGS
from dicom_output import (
    tag_pr2mask_mask,
    validate_dicom_input,
    write_prediction_outputs,
)
from parallel_inference import ParallelEnsemblePatchInferenceEngine
from redcap_output import write_redcap_mask, model_repeat_instance


SW_BATCH_SIZE = 1
THREADS_PER_ENSEMBLE_MODEL = 3
DEFAULT_WORK_DIR = Path("/output_tmp")
DEFAULT_PR2MASK_DIR = Path("/pr2mask")
FINAL_OUTPUT_DIRS = ("fused", "fused_vote_map", "reports", "labels", "redcap")
LOG_NAME = "pacs_command.log"


def validate_prediction_outputs(mask, probabilities):
    """Validate the paired library output before creating DICOM series."""
    if mask.ndim != 4 or mask.shape[0] != 1:
        raise RuntimeError(
            f"Expected mask shape [1, D, H, W], got {tuple(mask.shape)}"
        )
    if mask.dtype != torch.long:
        raise RuntimeError(f"Expected torch.long mask, got {mask.dtype}")
    if probabilities.ndim != 4 or probabilities.shape[0] != 2:
        raise RuntimeError(
            "Expected two class-probability channels [2, D, H, W], "
            f"got {tuple(probabilities.shape)}"
        )
    if not probabilities.dtype.is_floating_point:
        raise RuntimeError(
            f"Expected floating-point probabilities, got {probabilities.dtype}"
        )
    if tuple(mask.shape[1:]) != tuple(probabilities.shape[1:]):
        raise RuntimeError(
            "Mask and probabilities have different spatial shapes: "
            f"{tuple(mask.shape[1:])} != {tuple(probabilities.shape[1:])}"
        )
    if not bool(torch.isfinite(probabilities).all()):
        raise RuntimeError("Probability output contains non-finite values")

    prob_min = float(probabilities.min())
    prob_max = float(probabilities.max())
    tolerance = 1e-6
    if prob_min < -tolerance or prob_max > 1 + tolerance:
        raise RuntimeError(
            f"Probability output outside [0, 1]: min={prob_min}, max={prob_max}"
        )
    if not set(torch.unique(mask).tolist()).issubset({0, 1}):
        raise RuntimeError("Mask contains labels other than 0 and 1")

    return mask, probabilities.clamp(0, 1)


def _deployment_label(deployment: dict) -> str:
    count = len(deployment["members"])
    return "single model" if count == 1 else f"{count}-model ensemble"


def _create_inference_engine(deployment: dict):
    model_paths = deployment.get("model_paths")
    if model_paths and len(model_paths) > 1:
        return ParallelEnsemblePatchInferenceEngine(
            model_paths,
            [member["member_id"] for member in deployment["members"]],
            deployment["patch_config"],
            output_channels=deployment["output_channels"],
            threads_per_model=THREADS_PER_ENSEMBLE_MODEL,
            sw_batch_size=SW_BATCH_SIZE,
        )
    if model_paths:
        predictor = load_safetensors_model(model_paths[0], device="cpu")
        print(
            "  Loaded {}: {}".format(
                deployment["members"][0]["member_id"], model_paths[0].name
            )
        )
    else:
        # Compatibility path for injected/test deployments using an in-memory model.
        predictor = deployment["predictor"]
    return PatchInferenceEngine(
        predictor,
        deployment["patch_config"],
        sw_batch_size=SW_BATCH_SIZE,
    )


def _required_pr2mask_tools(pr2mask_dir: Path) -> dict[str, Path]:
    tools = {
        name: pr2mask_dir / name
        for name in ("imageAndMask2Report", "imageAndMask2Fused")
    }
    missing = [str(path) for path in tools.values() if not path.is_file()]
    not_executable = [
        str(path)
        for path in tools.values()
        if path.is_file() and not os.access(path, os.X_OK)
    ]
    if missing or not_executable:
        details = []
        if missing:
            details.append(f"missing={missing}")
        if not_executable:
            details.append(f"not executable={not_executable}")
        raise RuntimeError(
            "required pr2mask tools are unavailable: " + ", ".join(details)
        )
    return tools


def _prepare_empty_directory(path: Path, label: str) -> None:
    if path.exists():
        if not path.is_dir():
            raise RuntimeError(f"{label} is not a directory: {path}")
        if any(path.iterdir()):
            raise RuntimeError(f"{label} must be empty: {path}")
        return
    path.mkdir(parents=True)


def _prepare_output_directory(path: Path) -> None:
    if os.path.lexists(path):
        if not path.is_dir():
            raise RuntimeError(f"output path is not a directory: {path}")
    else:
        path.mkdir(parents=True)
    collisions = [
        name for name in FINAL_OUTPUT_DIRS if os.path.lexists(path / name)
    ]
    if collisions:
        raise RuntimeError(
            f"owned output directories already exist in {path}: {collisions}"
        )


def _tag_pr2mask_labels(work_dir: Path, deployment: dict, *, use_tta: bool) -> None:
    """Add fastMONAI provenance to the mask series pr2mask wrote in labels/."""
    labels_dir = work_dir / "labels"
    series = sorted(path for path in labels_dir.glob("*") if path.is_dir())
    if len(series) != 1:
        raise RuntimeError(
            f"pr2mask must create exactly one mask series in {labels_dir}; "
            f"found {len(series)}"
        )
    tag_pr2mask_mask(series[0], deployment, use_tta=use_tta)


def _copy_final_outputs(work_dir: Path, output_dir: Path) -> None:
    missing = [name for name in FINAL_OUTPUT_DIRS if not (work_dir / name).is_dir()]
    if missing:
        raise RuntimeError(
            f"pr2mask did not create required output directories: {missing}"
        )
    for name in FINAL_OUTPUT_DIRS:
        shutil.copytree(work_dir / name, output_dir / name)


def _publish_log(log_path: Path, output_dir: Path) -> None:
    temporary_path = None
    try:
        with (
            log_path.open("rb") as source,
            tempfile.NamedTemporaryFile(
                dir=output_dir,
                prefix=f".{LOG_NAME}.",
                delete=False,
            ) as temporary,
        ):
            temporary_path = Path(temporary.name)
            shutil.copyfileobj(source, temporary)
        shutil.copymode(log_path, temporary_path)
        os.replace(temporary_path, output_dir / LOG_NAME)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _run_postprocessing(
    input_dir: Path,
    work_dir: Path,
    output_dir: Path,
    deployment: dict,
    *,
    use_tta: bool,
    version: str,
    tools: dict[str, Path],
) -> None:
    for name in ("model_mask", "vote_map"):
        if not (work_dir / name).is_dir():
            raise RuntimeError(f"inference did not create required {name!r} output")

    model_type = deployment["model_type"]
    tta = "tta" if use_tta else "no-tta"
    # The image version pins one bundle per model type, so version, model and TTA
    # keep pr2mask's stable series UIDs distinct.
    identity = f"{version}_{model_type}_{tta}"
    # The report title already names the model.
    report_identity = f"{version}_{tta}"
    info = f"{model_type} {_deployment_label(deployment)}, {datetime.now():%b%d%Y}"
    commands = (
        (
            "imageAndMask2Report",
            [
                str(tools["imageAndMask2Report"]),
                str(input_dir),
                str(work_dir / "model_mask"),
                str(work_dir),
                "-u",
                f"{identity}_report",
                "-i",
                report_identity,
                "--reporttype",
                "mosaic",
                "-t",
                f"{info} ",
            ],
        ),
        (
            "imageAndMask2Fused",
            [
                str(tools["imageAndMask2Fused"]),
                str(input_dir),
                str(work_dir / "model_mask"),
                str(work_dir),
                "-u",
                f"{identity}_fused",
                "-t",
                f"{info} ",
            ],
        ),
        (
            "imageAndMask2Fused (vote map)",
            [
                str(tools["imageAndMask2Fused"]),
                str(input_dir),
                str(work_dir / "vote_map"),
                str(work_dir),
                "--votemapmax",
                "65535",
                "--votemapagree",
                "0.5",
                "-u",
                f"{identity}_votemap",
                "-s",
                "peak agreement {peak_agreement}",
                "-t",
                f"{info} ",
            ],
        ),
    )

    log_path = work_dir / LOG_NAME
    with log_path.open("w", encoding="utf-8") as log:
        for label, command in commands:
            print(f"{label}:")
            subprocess.run(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
    _tag_pr2mask_labels(work_dir, deployment, use_tta=use_tta)
    write_redcap_mask(work_dir, input_dir, deployment, version=version, use_tta=use_tta)
    _copy_final_outputs(work_dir, output_dir)
    _publish_log(log_path, output_dir)


def run_inference(
    datafolder,
    output,
    model_type,
    use_tta=True,
    *,
    version,
    work_dir=DEFAULT_WORK_DIR,
    pr2mask_dir=DEFAULT_PR2MASK_DIR,
):
    input_dir = Path(datafolder)
    output_dir = Path(output)
    work_dir = Path(work_dir)
    if not version:
        raise RuntimeError("VERSION is required for PACS output identity")
    tools = _required_pr2mask_tools(Path(pr2mask_dir))
    _prepare_output_directory(output_dir)
    _prepare_empty_directory(work_dir, "work directory")
    validate_dicom_input(input_dir)

    print("=" * 60)
    print("Vestibular Schwannoma PACS Segmentation")
    print("=" * 60)

    deployment = load_deployment(model_type)
    model_repeat_instance(deployment["bundle_sha256"])
    patch_config = deployment["patch_config"]
    label = _deployment_label(deployment)
    print(f"Deployment: {label}")
    print(f"Patch size: {patch_config.patch_size}")

    engine = _create_inference_engine(deployment)
    print(f"Running patch inference (TTA={'on' if use_tta else 'off'})...")
    segmentation, probabilities = engine.predict_mask_and_probabilities(
        str(input_dir), tta=use_tta
    )
    segmentation, probabilities = validate_prediction_outputs(
        segmentation, probabilities
    )
    write_prediction_outputs(
        segmentation,
        probabilities[1],
        input_dir,
        work_dir,
        deployment,
        use_tta=use_tta,
    )
    _run_postprocessing(
        input_dir,
        work_dir,
        output_dir,
        deployment,
        use_tta=use_tta,
        version=version,
        tools=tools,
    )

    tumor_voxels = int((segmentation > 0).sum())
    total_voxels = segmentation.numel()
    print("=" * 60)
    print(f"INFERENCE COMPLETE: {label}")
    print(f"TTA: {'on' if use_tta else 'off'}")
    print(f"Tumor voxels: {tumor_voxels}")
    print(f"Total voxels: {total_voxels}")
    print(f"Tumor percentage: {100 * tumor_voxels / total_voxels:.4f}%")
    print("=" * 60)


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="VestibularSchwannomaSegmentation",
        description="Run declared model inference and PACS postprocessing.",
    )
    parser.add_argument("input_dir", help="Directory containing the input DICOM series")
    parser.add_argument("output_dir", help="Directory for final PACS output")
    parser.add_argument("--model-type", choices=tuple(MODEL_CONFIGS), default="dynunet")
    parser.add_argument(
        "--tta",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable or disable 8-flip TTA (default: enabled)",
    )
    args = parser.parse_args(argv)
    run_inference(
        args.input_dir,
        args.output_dir,
        args.model_type,
        use_tta=args.tta,
        version=os.environ.get("VERSION"),
    )


if __name__ == "__main__":
    main()
