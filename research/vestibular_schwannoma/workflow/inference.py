"""Artifact resolution and predictor loading for VS inference notebooks."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from fastMONAI.vision_all import (
    PatchConfig,
    find_model_artifacts,
    load_safetensors_model,
    patch_config_to_dict,
    read_safetensors_metadata,
)
from .config import INFERENCE_RUN_IDS_SCHEMA, VS_OUTPUT_SPEC


@dataclass(frozen=True)
class LoadedPredictorSet:
    """Validated models and their shared inference contract."""

    artifacts: dict[str, Path]
    models: tuple[object, ...]
    patch_config: PatchConfig

    @property
    def predictor(self):
        """Return the single model or the model list expected by patch inference."""

        return self.models[0] if len(self.models) == 1 else list(self.models)


def _validate_member_mapping(name: str, values: Mapping | None) -> dict:
    if values is None:
        return {}
    if not isinstance(values, Mapping):
        raise TypeError(f"{name} must be a mapping of member ID to value")
    resolved = dict(values)
    if any(not isinstance(member, str) or not member for member in resolved):
        raise ValueError(f"Every {name} member ID must be a non-empty string")
    return resolved


def _read_inference_run_ids(
    selection_file: str | Path,
    *,
    model_key: str,
    artifact_role: str,
) -> dict[str, str]:
    path = Path(selection_file).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Inference run selection not found: {path}")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid inference run selection JSON: {path}") from exc
    if not isinstance(manifest, dict):
        raise ValueError("Inference run selection must be a JSON object")
    if manifest.get("schema_version") != INFERENCE_RUN_IDS_SCHEMA:
        raise ValueError(
            f"Unsupported inference run selection schema: "
            f"{manifest.get('schema_version')!r}"
        )
    if not isinstance(manifest.get("run_group"), str) or not manifest["run_group"]:
        raise ValueError("Inference run selection has an invalid run_group")
    models = manifest.get("models")
    if not isinstance(models, dict):
        raise ValueError("Inference run selection has an invalid models mapping")
    if model_key not in models:
        raise ValueError(
            f"Model {model_key!r} is not ready in {path}; "
            f"available models: {sorted(models)}"
        )
    roles = models[model_key]
    if not isinstance(roles, dict) or artifact_role not in roles:
        available = sorted(roles) if isinstance(roles, dict) else []
        raise ValueError(
            f"Role {artifact_role!r} is not ready for model {model_key!r}; "
            f"available roles: {available}"
        )
    run_ids = _validate_member_mapping("inference run selection", roles[artifact_role])
    if not run_ids:
        raise ValueError("Inference run selection contains no model members")
    return run_ids


def load_inference_models(
    *,
    run_selection_file: str | Path | None = None,
    model_key: str | None = None,
    local_model_artifacts: Mapping[str, str | Path] | None = None,
    artifact_role: str,
    device: str,
) -> LoadedPredictorSet:
    """Resolve, validate, and load one declared Safetensors model set."""

    local = _validate_member_mapping("local_model_artifacts", local_model_artifacts)
    has_selection = run_selection_file is not None
    if has_selection == bool(local):
        raise ValueError(
            "Declare exactly one of run_selection_file or local_model_artifacts"
        )
    if artifact_role not in {"best", "final"}:
        raise ValueError("artifact_role must be 'best' or 'final'")

    if has_selection:
        if not isinstance(model_key, str) or not model_key:
            raise ValueError("model_key is required with run_selection_file")
        run_ids = _read_inference_run_ids(
            run_selection_file,
            model_key=model_key,
            artifact_role=artifact_role,
        )
        if any(not isinstance(run_id, str) or not run_id for run_id in run_ids.values()):
            raise ValueError("Every MLflow run ID must be a non-empty string")
        if len(set(run_ids.values())) != len(run_ids):
            raise ValueError("Inference run selection contains duplicate run IDs")
        resolved = find_model_artifacts(
            run_ids=run_ids,
            artifact_role=artifact_role,
        )
    else:
        run_ids = {}
        resolved = local

    artifacts = {member: Path(path).expanduser() for member, path in resolved.items()}
    declared_members = set(run_ids or local)
    if set(artifacts) != declared_members:
        raise ValueError(
            "Resolved artifact members do not match the declared model members"
        )
    missing = [str(path) for path in artifacts.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Model artifacts not found: {missing}")
    invalid = [str(path) for path in artifacts.values() if path.suffix != ".safetensors"]
    if invalid:
        raise ValueError(f"All model artifacts must be Safetensors files: {invalid}")
    canonical = [path.resolve() for path in artifacts.values()]
    if len(set(canonical)) != len(canonical):
        raise ValueError("Declared members resolve to duplicate model artifacts")
    reference_config = None
    for member, path in artifacts.items():
        metadata = read_safetensors_metadata(path)
        expected_run_id = run_ids.get(member)
        if expected_run_id and metadata.get("mlflow_run") != expected_run_id:
            raise ValueError(
                f"{member!r} declares MLflow run {metadata.get('mlflow_run')!r}; "
                f"expected {expected_run_id!r}"
            )
        if metadata.get("artifact_role") != artifact_role:
            raise ValueError(
                f"{member!r} has role {metadata.get('artifact_role')!r}; "
                f"expected {artifact_role!r}"
            )

        inference_config = metadata["inference_config"]
        if inference_config.get("workflow") != "patch":
            raise ValueError(f"{member!r} does not carry a patch inference config")
        if inference_config.get("output") != VS_OUTPUT_SPEC:
            raise ValueError(
                f"{member!r} has output={inference_config.get('output')!r}; "
                f"expected {VS_OUTPUT_SPEC!r}"
            )
        if reference_config is None:
            reference_config = inference_config
        elif inference_config != reference_config:
            raise ValueError(
                "Declared ensemble members have different inference configurations"
            )

    patch_values = patch_config_to_dict(
        reference_config["patch_config"], inference_only=True
    )
    patch_config = PatchConfig(**patch_values)
    if patch_config.keep_largest_component:
        raise ValueError(
            "VS inference artifacts must preserve all predicted components; "
            "re-export the model with keep_largest_component=False"
        )
    models = tuple(
        load_safetensors_model(path, device=device) for path in artifacts.values()
    )
    return LoadedPredictorSet(
        artifacts=artifacts,
        models=models,
        patch_config=patch_config,
    )
