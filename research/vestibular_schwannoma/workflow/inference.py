"""Artifact resolution and predictor loading for VS inference notebooks."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from fastMONAI.vision_all import (
    PatchConfig,
    find_model_artifacts,
    load_safetensors_model,
    make_output_spec,
    patch_config_to_dict,
    read_safetensors_metadata,
)


@dataclass(frozen=True)
class LoadedPredictorSet:
    """Validated models and their shared inference contract."""

    mode: str
    member_ids: tuple[str, ...]
    artifacts: dict[str, Path]
    models: tuple[object, ...]
    patch_config: PatchConfig
    inference_config: dict
    metadata: dict[str, dict]

    @property
    def predictor(self):
        """Return the single model or the model list expected by patch inference."""

        return self.models[0] if self.mode == "single" else list(self.models)


def _validate_member_mapping(name: str, values: Mapping | None) -> dict:
    if values is None:
        return {}
    if not isinstance(values, Mapping):
        raise TypeError(f"{name} must be a mapping of member ID to value")
    resolved = dict(values)
    if any(not isinstance(member, str) or not member for member in resolved):
        raise ValueError(f"Every {name} member ID must be a non-empty string")
    return resolved


def resolve_model_artifacts(
    *,
    member_run_ids: Mapping[str, str] | None = None,
    local_model_artifacts: Mapping[str, str | Path] | None = None,
    artifact_role: str,
) -> dict[str, Path]:
    """Resolve exactly one declared source: MLflow runs or local Safetensors files."""

    run_ids = _validate_member_mapping("member_run_ids", member_run_ids)
    local = _validate_member_mapping("local_model_artifacts", local_model_artifacts)
    if bool(run_ids) == bool(local):
        raise ValueError(
            "Declare exactly one of member_run_ids or local_model_artifacts"
        )
    if artifact_role not in {"best", "final"}:
        raise ValueError("artifact_role must be 'best' or 'final'")

    if run_ids:
        if any(not isinstance(run_id, str) or not run_id for run_id in run_ids.values()):
            raise ValueError("Every MLflow run ID must be a non-empty string")
        if len(set(run_ids.values())) != len(run_ids):
            raise ValueError("member_run_ids contains duplicate run IDs")
        resolved = find_model_artifacts(
            run_ids=run_ids,
            artifact_role=artifact_role,
            expected_members=list(run_ids),
        )
    else:
        resolved = local

    artifacts = {member: Path(path).expanduser() for member, path in resolved.items()}
    missing = [str(path) for path in artifacts.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Model artifacts not found: {missing}")
    invalid = [str(path) for path in artifacts.values() if path.suffix != ".safetensors"]
    if invalid:
        raise ValueError(f"All model artifacts must be Safetensors files: {invalid}")
    canonical = [path.resolve() for path in artifacts.values()]
    if len(set(canonical)) != len(canonical):
        raise ValueError("Declared members resolve to duplicate model artifacts")
    return artifacts


def load_predictor_set(
    artifacts: Mapping[str, str | Path],
    *,
    mode: str,
    artifact_role: str,
    device: str,
    expected_run_ids: Mapping[str, str] | None = None,
    expected_output: dict | None = None,
) -> LoadedPredictorSet:
    """Validate a declared model set and load it only after all metadata agrees."""

    if mode not in {"single", "ensemble"}:
        raise ValueError(f"Unknown deployment mode: {mode!r}")
    if artifact_role not in {"best", "final"}:
        raise ValueError("artifact_role must be 'best' or 'final'")

    artifact_map = _validate_member_mapping("artifacts", artifacts)
    if mode == "single" and len(artifact_map) != 1:
        raise ValueError("single mode requires exactly one declared model")
    if mode == "ensemble" and len(artifact_map) < 2:
        raise ValueError("ensemble mode requires at least two declared models")

    paths = {member: Path(path).expanduser() for member, path in artifact_map.items()}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Model artifacts not found: {missing}")
    if any(path.suffix != ".safetensors" for path in paths.values()):
        raise ValueError("All declared members must be Safetensors artifacts")
    canonical = [path.resolve() for path in paths.values()]
    if len(set(canonical)) != len(canonical):
        raise ValueError("Declared members resolve to duplicate model artifacts")

    expected_runs = _validate_member_mapping("expected_run_ids", expected_run_ids)
    if expected_runs and set(expected_runs) != set(paths):
        raise ValueError(
            "expected_run_ids members must exactly match the declared artifact members"
        )
    output_spec = expected_output or make_output_spec(
        "multiclass_segmentation", classes=2
    )

    metadata_by_member = {}
    reference_config = None
    for member, path in paths.items():
        metadata = read_safetensors_metadata(path)
        expected_run_id = expected_runs.get(member)
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
        if inference_config.get("output") != output_spec:
            raise ValueError(
                f"{member!r} has output={inference_config.get('output')!r}; "
                f"expected {output_spec!r}"
            )
        if reference_config is None:
            reference_config = inference_config
        elif inference_config != reference_config:
            raise ValueError(
                "Declared ensemble members have different inference configurations"
            )
        metadata_by_member[member] = metadata

    patch_values = patch_config_to_dict(
        reference_config["patch_config"], inference_only=True
    )
    patch_config = PatchConfig(**patch_values)
    models = tuple(
        load_safetensors_model(path, device=device) for path in paths.values()
    )
    return LoadedPredictorSet(
        mode=mode,
        member_ids=tuple(paths),
        artifacts=paths,
        models=models,
        patch_config=patch_config,
        inference_config=reference_config,
        metadata=metadata_by_member,
    )
