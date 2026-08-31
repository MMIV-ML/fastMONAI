"""Artifact resolution and predictor loading for VS inference notebooks."""

from __future__ import annotations

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
from .config import VS_OUTPUT_SPEC
from .run_selection import (
    merge_fold_run_selections as merge_fold_run_selections,
    read_inference_run_ids as _read_inference_run_ids,
)


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
    invalid = [
        str(path) for path in artifacts.values() if path.suffix != ".safetensors"
    ]
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
