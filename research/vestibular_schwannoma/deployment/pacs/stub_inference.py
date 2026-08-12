#!/usr/bin/env python3
"""Declared single-model or ensemble patch inference for vestibular schwannoma."""

from __future__ import annotations

import argparse
import hashlib
import json
import uuid
from pathlib import Path

import numpy as np
import torch
import fastMONAI
from fastMONAI.vision_all import (
    load_safetensors_model,
    make_output_spec,
    read_safetensors_metadata,
)
from fastMONAI.vision_patch import PatchConfig, PatchInferenceEngine
from imagedata.series import Series
from pydicom import dcmread
from pydicom.uid import UID
from deployment_models import (
    MODEL_ARCH_IDS,
    MODEL_CONFIGS,
    make_dicom_uid_contract,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DEPLOYMENT_CONFIG = "deployment_config.json"
SW_BATCH_SIZE = 2


def _canonical_json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read deployment configuration {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"deployment configuration must be a JSON object: {path}")
    return value


def _validate_new_deployment(models_dir: Path, config: dict, model_type: str) -> dict:
    if config.get("schema_version") != 2:
        raise RuntimeError(f"unsupported deployment schema: {config.get('schema_version')!r}")
    if config.get("model_type") != model_type:
        raise RuntimeError(
            f"bundle declares model_type={config.get('model_type')!r}, requested {model_type!r}"
        )
    model_spec = config.get("model_spec")
    if not isinstance(model_spec, dict):
        raise RuntimeError("deployment model_spec must be an object")
    arch_id = model_spec.get("arch_id")
    if arch_id not in MODEL_ARCH_IDS[model_type]:
        raise RuntimeError(
            f"bundle arch_id={arch_id!r} does not match model_type={model_type!r}"
        )
    mode = config.get("mode")
    members = config.get("members")
    expected = config.get("expected_member_count")
    if mode not in {"single", "ensemble"} or not isinstance(members, list):
        raise RuntimeError("deployment mode/members declaration is invalid")
    if not isinstance(expected, int) or expected != len(members):
        raise RuntimeError(
            f"expected_member_count={expected!r} does not match {len(members)} declared members"
        )
    if mode == "single" and expected != 1:
        raise RuntimeError("single deployment must declare exactly one member")
    if mode == "ensemble" and expected < 2:
        raise RuntimeError("ensemble deployment must declare at least two members")
    expected_uid_contract = make_dicom_uid_contract(model_type, mode, expected)
    if config.get("dicom_uid") != expected_uid_contract:
        raise RuntimeError(
            "deployment DICOM UID contract does not match the model registry: "
            f"expected {expected_uid_contract!r}"
        )

    member_ids = [member.get("member_id") for member in members if isinstance(member, dict)]
    artifacts = [member.get("artifact") for member in members if isinstance(member, dict)]
    run_ids = [member.get("mlflow_run_id") for member in members if isinstance(member, dict)]
    if len(member_ids) != expected or any(not isinstance(value, str) or not value
                                          for value in member_ids + artifacts + run_ids):
        raise RuntimeError("each declared member needs member_id, artifact, and mlflow_run_id")
    for label, values in (("member ids", member_ids), ("artifacts", artifacts),
                          ("MLflow run ids", run_ids)):
        if len(set(values)) != len(values):
            raise RuntimeError(f"deployment contains duplicate {label}")
    if any(Path(name).name != name or not name.endswith(".safetensors") for name in artifacts):
        raise RuntimeError("member artifacts must be local .safetensors filenames")
    if any(member.get("format") != "safetensors" for member in members):
        raise RuntimeError("new deployment schemas accept only Safetensors members")

    # A glob is used only to reject undeclared files; it never determines model count.
    declared_files = set(artifacts)
    packaged_models = {path.name for path in models_dir.glob("*.safetensors")}
    if packaged_models != declared_files:
        missing = sorted(declared_files - packaged_models)
        extra = sorted(packaged_models - declared_files)
        raise RuntimeError(f"bundle model files do not match declaration; missing={missing}, extra={extra}")

    for member in members:
        path = models_dir / member["artifact"]
        if _sha256_file(path) != member.get("sha256"):
            raise RuntimeError(f"SHA-256 mismatch for declared member {member['member_id']!r}")

    inference = config.get("inference_config")
    if not isinstance(inference, dict) or set(inference) != {"canonical_sha256"}:
        raise RuntimeError("deployment must declare exactly one embedded inference-config hash")
    inference_hash = inference["canonical_sha256"]
    if (
        not isinstance(inference_hash, str)
        or len(inference_hash) != 64
        or any(char not in "0123456789abcdef" for char in inference_hash)
    ):
        raise RuntimeError("deployment inference-config hash is invalid")

    bundle_hash = config.get("bundle_sha256")
    payload = dict(config)
    payload.pop("bundle_sha256", None)
    if bundle_hash != _sha256_bytes(_canonical_json(payload).encode("utf-8")):
        raise RuntimeError("deployment declaration has a bundle SHA-256 mismatch")

    config["_models_dir"] = models_dir
    return config


def load_deployment(model_type: str) -> dict:
    model_config = MODEL_CONFIGS[model_type]
    models_dir = SCRIPT_DIR / model_config["models_dir"]
    deployment_path = models_dir / DEPLOYMENT_CONFIG
    if not deployment_path.is_file():
        raise FileNotFoundError(
            f"no declared {model_type!r} bundle found; expected {deployment_path}"
        )
    deployment = _validate_new_deployment(
        models_dir, _read_json(deployment_path), model_type
    )

    models_dir = deployment["_models_dir"]
    reference_inference = None
    models = []
    for member in deployment["members"]:
        path = models_dir / member["artifact"]
        metadata = read_safetensors_metadata(path)
        if metadata.get("mlflow_run") != member["mlflow_run_id"]:
            raise RuntimeError(f"run-id mismatch in member {member['member_id']!r}")
        if metadata.get("artifact_role") != member.get("artifact_role"):
            raise RuntimeError(f"artifact-role mismatch in member {member['member_id']!r}")
        member_spec = {
            "artifact_schema": metadata["artifact_schema"],
            "arch_id": metadata["arch_id"],
            "arch_kwargs": metadata["arch_kwargs"],
            "wrapper_spec": metadata["wrapper_spec"],
        }
        if member_spec != deployment.get("model_spec"):
            raise RuntimeError(f"model-spec mismatch in member {member['member_id']!r}")
        inference_config = metadata["inference_config"]
        expected_output = make_output_spec("multiclass_segmentation", classes=2)
        if inference_config.get("output") != expected_output:
            raise RuntimeError(
                f"output-spec mismatch in member {member['member_id']!r}; "
                f"expected {expected_output!r}"
            )
        inference_hash = _sha256_bytes(_canonical_json(inference_config).encode("utf-8"))
        if inference_hash != deployment["inference_config"].get("canonical_sha256"):
            raise RuntimeError(
                f"inference configuration mismatch in member {member['member_id']!r}"
            )
        if reference_inference is None:
            reference_inference = inference_config
        elif inference_config != reference_inference:
            raise RuntimeError("declared ensemble members have different inference configs")
        model = load_safetensors_model(path, device="cpu")
        model.eval()
        models.append(model)
        print(f"  Loaded {member['member_id']}: {path.name}")

    if reference_inference is None or reference_inference.get("workflow") != "patch":
        raise RuntimeError("model artifact does not declare patch inference")
    patch_config_dict = dict(reference_inference["patch_config"])

    deployment["models"] = models
    deployment["predictor"] = models[0] if deployment["mode"] == "single" else models
    deployment["patch_config"] = PatchConfig(**patch_config_dict)
    return deployment


def make_derived_dicom_uid(
    deployment: dict,
    source_series_uid: str,
    output_kind: str,
    *,
    source_sop_uid: str | None = None,
    slice_index: int | None = None,
) -> str:
    """Create a deterministic, standards-valid ``2.25.<UUID integer>`` UID."""
    contract = deployment["dicom_uid"]
    try:
        output_code = contract["output_codes"][output_kind]
    except KeyError as exc:
        raise ValueError(f"unknown DICOM output kind: {output_kind!r}") from exc
    if (source_sop_uid is None) != (slice_index is None):
        raise ValueError("source_sop_uid and slice_index must be provided together")

    identity = {
        "format_version": contract["format_version"],
        "model_code": contract["model_code"],
        "deployment_code": contract["deployment_code"],
        "member_count": contract["member_count"],
        "output_code": output_code,
        "bundle_sha256": deployment["bundle_sha256"],
        "source_series_uid": str(source_series_uid),
        "scope": "instance" if source_sop_uid is not None else "series",
    }
    if source_sop_uid is not None:
        identity.update({
            "source_sop_uid": str(source_sop_uid),
            "slice_index": slice_index,
        })
    namespace = uuid.UUID(contract["namespace_uuid"])
    generated = uuid.uuid5(namespace, _canonical_json(identity))
    uid = f"{contract['root']}.{generated.int}"
    if len(uid) > 64:
        raise RuntimeError(f"generated DICOM UID exceeds 64 characters: {uid}")
    return uid


def _source_sop_uids(series_obj) -> list[str]:
    values = getattr(series_obj, "SOPInstanceUIDs", None)
    if isinstance(values, dict):
        ordered = []
        for slice_idx in range(series_obj.slices):
            value = values.get((0, slice_idx), values.get(slice_idx))
            ordered.append(str(value) if value is not None else f"slice-{slice_idx}")
        return ordered
    base = series_obj.getDicomAttribute("SOPInstanceUID")
    return [f"{base}:slice-{slice_idx}" for slice_idx in range(series_obj.slices)]


def _finalize_written_dicom(save_dir):
    """Synchronize the file-meta SOP UID after ``imagedata`` writes."""
    paths = sorted(path for path in Path(save_dir).iterdir() if path.is_file())
    if not paths:
        raise RuntimeError(f"DICOM writer produced no files in {save_dir}")
    for path in paths:
        dataset = dcmread(str(path))
        series_uid = str(dataset.SeriesInstanceUID)
        sop_uid = str(dataset.SOPInstanceUID)
        if not UID(series_uid).is_valid or not UID(sop_uid).is_valid:
            raise RuntimeError(f"writer produced an invalid DICOM UID in {path}")
        dataset.file_meta.MediaStorageSOPInstanceUID = sop_uid
        dataset.save_as(str(path), write_like_original=False)


def save_series_pred(series_obj, save_dir, deployment, output_kind):
    """Save a prediction series with deterministic numeric DICOM UIDs."""
    source_series_uid = str(series_obj.seriesInstanceUID)
    source_sop_uids = _source_sop_uids(series_obj)
    series_uid = make_derived_dicom_uid(
        deployment, source_series_uid, output_kind
    )
    series_obj.seriesInstanceUID = series_uid
    series_obj.setDicomAttribute("SeriesInstanceUID", series_uid)
    if hasattr(series_obj, "patientID") and series_obj.patientID:
        series_obj.studyID = (series_obj.patientID[3:] if len(series_obj.patientID) > 3
                              else series_obj.patientID)
    for slice_idx in range(series_obj.slices):
        new_uid = make_derived_dicom_uid(
            deployment,
            source_series_uid,
            output_kind,
            source_sop_uid=source_sop_uids[slice_idx],
            slice_index=slice_idx,
        )
        series_obj.setDicomAttribute("SOPInstanceUID", new_uid, slice=slice_idx)
    series_obj.write(save_dir, opts={"keep_uid": True}, formats=["dicom"])
    # imagedata preserves the source file-meta SOP UID unless it is synchronized here.
    _finalize_written_dicom(save_dir)


def _series_description(deployment: dict, output_kind: str) -> str:
    count = deployment["expected_member_count"]
    mode = "single model" if deployment["mode"] == "single" else f"{count}-model ensemble"
    return (
        f"fastMONAI {MODEL_CONFIGS[deployment['model_type']]['display_name']} "
        f"{mode} {output_kind}"
    )


def _set_derived_metadata(series_obj, deployment, output_kind, software_versions):
    marker = "MASK" if output_kind == "segmentation" else "PROBABILITY"
    series_obj.setDicomAttribute("SoftwareVersions", software_versions)
    image_type = series_obj.getDicomAttribute("ImageType")
    image_type = [] if image_type is None else (
        [image_type] if isinstance(image_type, str) else list(image_type))
    series_obj.setDicomAttribute(
        "ImageType", ["DERIVED", "SECONDARY"] + image_type[2:] + [marker]
    )
    series_obj.setDicomAttribute(
        "SeriesDescription", _series_description(deployment, output_kind)
    )
    derivation = (
        f"fastMONAI {output_kind}; model={deployment['model_type']}; "
        f"mode={deployment['mode']}; members={deployment['expected_member_count']}; "
        f"bundle_sha256={deployment['bundle_sha256']}"
    )
    if output_kind == "probability":
        derivation += "; foreground probability = stored uint16 value / 65535"
    series_obj.setDicomAttribute("DerivationDescription", derivation)


def create_dicom_mask(pred, dicom_input_path, output_dir, deployment, software_versions):
    mask_obj = Series(str(dicom_input_path), opts={"slice_tolerance": 1e-2})
    _set_derived_metadata(
        mask_obj, deployment, "segmentation", software_versions
    )
    data = np.transpose(pred.numpy().squeeze(), (-1, 1, 0)).copy().astype(np.uint16)
    mask_obj[:] = data
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    save_series_pred(mask_obj, str(output_path), deployment, "segmentation")
    return output_path


def create_dicom_prob_mask(
    pred, dicom_input_path, output_dir, deployment, software_versions
):
    mask_obj = Series(str(dicom_input_path), opts={"slice_tolerance": 1e-2})
    _set_derived_metadata(
        mask_obj, deployment, "probability", software_versions
    )
    data = np.transpose(pred.numpy().squeeze(), (-1, 1, 0)).copy()
    mask_obj[:] = np.rint(data * 65535).astype(np.uint16)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    save_series_pred(mask_obj, str(output_path), deployment, "probability")
    return output_path


def build_software_versions(deployment: dict) -> list[str]:
    count = deployment["expected_member_count"]
    model_tag = (f"{deployment['model_type']}-single" if deployment["mode"] == "single"
                 else f"{deployment['model_type']}-{count}model")
    return ([model_tag, f"bundle-{deployment['bundle_sha256'][:8]}"]
            + [member["mlflow_run_id"][:8] for member in deployment["members"]]
            + [f"fastMONAI {fastMONAI.__version__}"])


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


def run_inference(datafolder, output, model_type, use_tta=False):
    print("=" * 60)
    print("Declared model-bundle inference - Vestibular Schwannoma Segmentation")
    print("=" * 60)

    # Validate the complete bundle before opening input DICOM.
    deployment = load_deployment(model_type)
    patch_config = deployment["patch_config"]
    count = deployment["expected_member_count"]
    label = "single model" if deployment["mode"] == "single" else f"{count}-model ensemble"
    print(f"Deployment: {label}")
    print(f"Patch size: {patch_config.patch_size}")

    engine = PatchInferenceEngine(
        deployment["predictor"], patch_config, sw_batch_size=SW_BATCH_SIZE
    )
    print(f"Running patch inference (TTA={'on' if use_tta else 'off'})...")
    segmentation, prob = engine.predict_mask_and_probabilities(
        datafolder, tta=use_tta
    )
    segmentation, prob = validate_prediction_outputs(segmentation, prob)
    tumor_prob = prob[1]

    software_versions = build_software_versions(deployment)
    mask_output_dir = output + "/mask"
    create_dicom_mask(segmentation, datafolder, mask_output_dir,
                      deployment=deployment,
                      software_versions=software_versions)
    prob_output_dir = output + "/vote_map"
    create_dicom_prob_mask(tumor_prob, datafolder, prob_output_dir,
                           deployment=deployment,
                           software_versions=software_versions)

    seg_array = segmentation.cpu().numpy().squeeze()
    tumor_voxels = np.sum(seg_array > 0)
    print("=" * 60)
    print(f"INFERENCE COMPLETE: {label}")
    print(f"TTA: {'on' if use_tta else 'off'}")
    print(f"Tumor voxels: {int(tumor_voxels)}")
    print(f"Total voxels: {seg_array.size}")
    print(f"Tumor percentage: {100 * tumor_voxels / seg_array.size:.4f}%")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        prog="VestibularSchwannomaSegmentation",
        description="Declared single-model or ensemble segmentation.",
    )
    parser.add_argument("fn", help="Directory name of the input folder")
    parser.add_argument("on", help="Directory name for the output")
    parser.add_argument("--model-type", choices=tuple(MODEL_CONFIGS), default="unet")
    parser.add_argument("--tta", action="store_true", help="Enable 8-flip TTA")
    args = parser.parse_args()
    run_inference(
        args.fn + "/input", args.on, args.model_type, use_tta=args.tta,
    )


if __name__ == "__main__":
    main()
