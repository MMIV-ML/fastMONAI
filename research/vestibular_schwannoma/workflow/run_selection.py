"""Read, validate, and combine MLflow inference run selections."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path


INFERENCE_RUN_IDS_SCHEMA = 1
INFERENCE_RUN_IDS_FILENAME = "inference_run_ids.json"
COMPLETED_RUN_IDS_FILENAME = "completed_run_ids.json"
INFERENCE_SELECTION_KIND = "inference_selection"
COMPLETED_REGISTRY_KIND = "completed_registry"
TRAINING_CONTRACT_SCHEMA = 1
CROSS_VALIDATION_FOLDS = (1, 2, 3, 4, 5)


def _payload_sha256(payload: Mapping) -> str:
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def make_training_contract(payload: Mapping) -> dict:
    """Create a checksummed, JSON-serializable training merge contract."""

    resolved = dict(payload)
    return {
        "schema_version": TRAINING_CONTRACT_SCHEMA,
        "sha256": _payload_sha256(resolved),
        "payload": resolved,
    }


def read_inference_manifest(selection_file: str | Path) -> tuple[Path, dict]:
    """Read and validate the common structure of an inference run manifest."""

    path = Path(selection_file).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Inference run selection not found: {path}")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid inference run selection JSON: {path}") from error
    if not isinstance(manifest, dict):
        raise ValueError("Inference run selection must be a JSON object")
    if manifest.get("schema_version") != INFERENCE_RUN_IDS_SCHEMA:
        raise ValueError(
            f"Unsupported inference run selection schema: "
            f"{manifest.get('schema_version')!r}"
        )
    if not isinstance(manifest.get("run_group"), str) or not manifest["run_group"]:
        raise ValueError("Inference run selection has an invalid run_group")
    manifest_kind = manifest.get("manifest_kind", INFERENCE_SELECTION_KIND)
    if manifest_kind not in {INFERENCE_SELECTION_KIND, COMPLETED_REGISTRY_KIND}:
        raise ValueError(f"Unsupported inference manifest kind: {manifest_kind!r}")
    if not isinstance(manifest.get("models"), dict):
        raise ValueError("Inference run selection has an invalid models mapping")
    return path, manifest


def _run_ids_from_manifest(
    manifest: Mapping,
    path: Path,
    *,
    model_key: str,
    artifact_role: str,
) -> dict[str, str]:
    models = manifest["models"]
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
    values = roles[artifact_role]
    if not isinstance(values, Mapping):
        raise TypeError(
            "inference run selection must be a mapping of member ID to value"
        )
    run_ids = dict(values)
    if any(not isinstance(member, str) or not member for member in run_ids):
        raise ValueError(
            "Every inference run selection member ID must be a non-empty string"
        )
    if not run_ids:
        raise ValueError("Inference run selection contains no model members")
    if any(not isinstance(run_id, str) or not run_id for run_id in run_ids.values()):
        raise ValueError("Every MLflow run ID must be a non-empty string")
    if len(set(run_ids.values())) != len(run_ids):
        raise ValueError("Inference run selection contains duplicate run IDs")
    return run_ids


def read_inference_run_ids(
    selection_file: str | Path,
    *,
    model_key: str,
    artifact_role: str,
) -> dict[str, str]:
    """Read one model role from a validated inference run selection."""

    path, manifest = read_inference_manifest(selection_file)
    if manifest.get("manifest_kind") == COMPLETED_REGISTRY_KIND:
        raise ValueError(
            "completed_run_ids.json is a partial registry and cannot be used "
            "directly for inference; merge it into inference_run_ids.json first"
        )
    return _run_ids_from_manifest(
        manifest,
        path,
        model_key=model_key,
        artifact_role=artifact_role,
    )


def _training_contract(
    manifest: Mapping,
    path: Path,
    *,
    model_key: str,
) -> dict:
    contracts = manifest.get("training_contracts")
    if not isinstance(contracts, Mapping) or model_key not in contracts:
        raise ValueError(
            f"Inference run selection has no training contract for {model_key!r}: "
            f"{path}"
        )
    contract = contracts[model_key]
    if not isinstance(contract, Mapping):
        raise ValueError(f"Invalid training contract for {model_key!r}: {path}")
    contract = dict(contract)
    if contract.get("schema_version") != TRAINING_CONTRACT_SCHEMA:
        raise ValueError(
            f"Unsupported training contract schema for {model_key!r}: "
            f"{contract.get('schema_version')!r}"
        )
    payload = contract.get("payload")
    digest = contract.get("sha256")
    if not isinstance(payload, Mapping) or not isinstance(digest, str) or not digest:
        raise ValueError(f"Invalid training contract for {model_key!r}: {path}")
    if _payload_sha256(payload) != digest:
        raise ValueError(
            f"Training contract checksum mismatch for {model_key!r}: {path}"
        )
    return contract


def merge_fold_run_selections(
    selection_files: Sequence[str | Path],
    *,
    model_key: str,
    output_root: str | Path,
) -> Path:
    """Merge disjoint, contract-compatible folds into one inference manifest."""

    if isinstance(selection_files, (str, Path)):
        raise TypeError("selection_files must be a sequence of manifest paths")
    paths = [Path(path).expanduser() for path in selection_files]
    if not paths:
        raise ValueError("At least one inference run selection is required")
    if not isinstance(model_key, str) or not model_key:
        raise ValueError("model_key must be a non-empty string")

    folds = CROSS_VALIDATION_FOLDS
    expected_members = tuple(f"fold_{fold}" for fold in folds)

    merged = {}
    shared_contract = None
    for path in paths:
        resolved_path, manifest = read_inference_manifest(path)
        run_ids = _run_ids_from_manifest(
            manifest,
            resolved_path,
            model_key=model_key,
            artifact_role="best",
        )
        contract = _training_contract(
            manifest,
            resolved_path,
            model_key=model_key,
        )
        if shared_contract is None:
            shared_contract = contract
        elif contract["sha256"] != shared_contract["sha256"]:
            raise ValueError(
                f"Training contract mismatch for {model_key!r}: {resolved_path}"
            )
        for member, run_id in run_ids.items():
            if member in merged:
                raise ValueError(
                    f"Duplicate model member {member!r} across inference selections"
                )
            merged[member] = run_id

    actual_members = set(merged)
    expected_member_set = set(expected_members)
    if actual_members != expected_member_set:
        missing = [
            member for member in expected_members if member not in actual_members
        ]
        unexpected = sorted(actual_members - expected_member_set)
        raise ValueError(
            "Merged fold members do not match the expected folds; "
            f"missing={missing}, unexpected={unexpected}"
        )
    if len(set(merged.values())) != len(merged):
        raise ValueError("Merged inference selection contains duplicate MLflow run IDs")

    destination_root = Path(output_root).expanduser()
    try:
        destination_root.mkdir(parents=True, exist_ok=False)
    except FileExistsError as error:
        raise FileExistsError(
            f"Merged results root already exists: {destination_root}. "
            "Choose a new --output-root."
        ) from error

    manifest = {
        "schema_version": INFERENCE_RUN_IDS_SCHEMA,
        "manifest_kind": INFERENCE_SELECTION_KIND,
        "run_group": destination_root.name,
        "training_contracts": {model_key: shared_contract},
        "models": {
            model_key: {"best": {member: merged[member] for member in expected_members}}
        },
    }
    destination = destination_root / INFERENCE_RUN_IDS_FILENAME
    temporary = destination.with_suffix(".json.tmp")
    try:
        temporary.write_text(
            json.dumps(manifest, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        try:
            destination_root.rmdir()
        except OSError:
            pass
        raise
    return destination
