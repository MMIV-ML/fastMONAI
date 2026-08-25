"""Load trusted Safetensors bundles produced by prepare_model_bundle.py."""

from __future__ import annotations

import json
from pathlib import Path

from fastMONAI.vision_all import (
    load_safetensors_model,
    read_safetensors_metadata,
)
from fastMONAI.vision_patch import PatchConfig

from deployment_models import (
    DEPLOYMENT_SCHEMA,
    MODEL_CONFIGS,
    bundle_member_filename,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DEPLOYMENT_CONFIG = "deployment_config.json"


def read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read deployment configuration {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"deployment configuration must be a JSON object: {path}")
    return value


def _read_packaged_deployment(path: Path, model_type: str) -> tuple[dict, list[str]]:
    """Read the trusted builder output needed to locate packaged model members."""
    deployment = read_json(path)
    if deployment.get("schema_version") != DEPLOYMENT_SCHEMA:
        raise RuntimeError(
            f"unsupported deployment schema: {deployment.get('schema_version')!r}"
        )
    if deployment.get("model_type") != model_type:
        raise RuntimeError(
            f"bundle declares model_type={deployment.get('model_type')!r}, "
            f"requested {model_type!r}"
        )
    members = deployment.get("members")
    if not isinstance(members, list) or not members:
        raise RuntimeError("deployment must declare at least one member")
    try:
        member_ids = [member["member_id"] for member in members]
        filenames = [bundle_member_filename(member_id) for member_id in member_ids]
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"deployment contains an invalid member ID: {exc}") from exc
    if len(set(member_ids)) != len(member_ids):
        raise RuntimeError("deployment contains duplicate member ids")
    return deployment, filenames


def load_deployment(model_type: str, *, script_dir: Path = SCRIPT_DIR) -> dict:
    if model_type not in MODEL_CONFIGS:
        raise ValueError(f"unknown model type: {model_type!r}")
    models_dir = script_dir / "model_bundles" / model_type
    deployment_path = models_dir / DEPLOYMENT_CONFIG
    if not deployment_path.is_file():
        raise FileNotFoundError(
            f"no declared {model_type!r} bundle found; expected {deployment_path}"
        )
    deployment, filenames = _read_packaged_deployment(deployment_path, model_type)
    paths = [models_dir / filename for filename in filenames]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"packaged model files not found: {missing}")

    first_metadata = read_safetensors_metadata(paths[0])
    models = [load_safetensors_model(path, device="cpu") for path in paths]
    for member, path in zip(deployment["members"], paths):
        print(f"  Loaded {member['member_id']}: {path.name}")

    deployment["predictor"] = models[0] if len(models) == 1 else models
    deployment["patch_config"] = PatchConfig(
        **dict(first_metadata["inference_config"]["patch_config"])
    )
    return deployment
