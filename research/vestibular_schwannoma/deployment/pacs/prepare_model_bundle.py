#!/usr/bin/env python3
"""Build an explicitly declared Safetensors deployment bundle.

Examples
--------
One model trained on all data::

    python prepare_model_bundle.py --model-type unet \
        --run all_data=RUN_ID --artifact-role final

An intentional ensemble (the member names need not be folds)::

    python prepare_model_bundle.py --model-type unet \
        --run fold_1=RUN_ID_1 --run fold_2=RUN_ID_2 --artifact-role best

A five-fold DynUNet deployment declares all five fold-best models::

    python prepare_model_bundle.py --model-type dynunet \
        --run fold_1=RUN_ID_1 --run fold_2=RUN_ID_2 \
        --run fold_3=RUN_ID_3 --run fold_4=RUN_ID_4 \
        --run fold_5=RUN_ID_5 --artifact-role best

Use ``--artifact member=/path/model.safetensors`` instead of ``--run`` when
the artifacts have already been downloaded.  Local artifacts must still carry
their MLflow run id in Safetensors metadata.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from fastMONAI.vision_all import (
    load_safetensors_model,
    make_output_spec,
    read_safetensors_metadata,
)
from deployment_hashing import (
    bundle_sha256 as _bundle_sha256,
    sha256_file as _sha256_file,
)
from deployment_models import (
    DEPLOYMENT_SCHEMA,
    MODEL_CONFIGS,
    bundle_member_filename,
    validate_registered_uid_prefix,
)


SCRIPT_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class MemberSource:
    member_id: str
    source: str
    run_id: str | None


def _parse_member(value: str, option: str) -> tuple[str, str]:
    try:
        member_id, source = value.split("=", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"{option} must use MEMBER_ID=VALUE, got {value!r}"
        ) from exc
    try:
        bundle_member_filename(member_id)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid member id {member_id!r}: {exc}") from exc
    if not source:
        raise argparse.ArgumentTypeError(f"{option} has an empty value for {member_id!r}")
    return member_id, source


def _declared_sources(args) -> list[MemberSource]:
    sources = []
    for value in args.run:
        member_id, run_id = _parse_member(value, "--run")
        sources.append(MemberSource(member_id, run_id, run_id))
    for value in args.artifact:
        member_id, path = _parse_member(value, "--artifact")
        sources.append(MemberSource(member_id, path, None))

    member_ids = [source.member_id for source in sources]
    duplicates = sorted({member for member in member_ids if member_ids.count(member) > 1})
    if duplicates:
        raise ValueError(f"duplicate declared member ids: {duplicates}")
    if not sources:
        raise ValueError("at least one --run or --artifact member must be declared")
    return sources


def _download_run_artifact(run_id: str, artifact_path: str) -> Path:
    import mlflow

    return Path(mlflow.artifacts.download_artifacts(
        run_id=run_id, artifact_path=artifact_path
    ))


def _requested_artifact_path(args) -> str:
    """Return the explicit override or the standard MLflow path for the declared role."""
    return args.artifact_path or f"model/{args.artifact_role}_model.safetensors"


def _resolve_source(source: MemberSource, artifact_path: str) -> Path:
    path = (_download_run_artifact(source.source, artifact_path)
            if source.run_id else Path(source.source).expanduser())
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"declared artifact for {source.member_id!r} not found: {path}")
    if path.suffix != ".safetensors":
        raise ValueError(
            f"new bundles require .safetensors artifacts, got {path.name!r} "
            f"for {source.member_id!r}"
        )
    return path


def _validate_inference_config(value: dict) -> None:
    if not isinstance(value, dict) or value.get("workflow") != "patch":
        raise ValueError("the model artifact does not contain a supported patch inference_config")
    if str(value.get("config_schema")) != "1":
        raise ValueError(
            f"unsupported patch inference config schema: {value.get('config_schema')!r}"
        )
    if not isinstance(value.get("patch_config"), dict):
        raise ValueError("inference_config.patch_config must be an object")
    if value["patch_config"].get("keep_largest_component") is not False:
        raise ValueError(
            "PACS deployment requires keep_largest_component=False so all predicted "
            "candidate regions are retained"
        )
    expected_output = make_output_spec("multiclass_segmentation", classes=2)
    if value.get("output") != expected_output:
        raise ValueError(
            "PACS deployment requires two-logit multiclass segmentation "
            f"output={expected_output!r}, got {value.get('output')!r}"
        )
    # read_safetensors_metadata already validated the exact inference field set.


def _validate_output_target(out: Path) -> None:
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise FileExistsError(
            f"output directory is not empty: {out}. Use a new directory or remove the "
            "old bundle explicitly so stale model members cannot be shipped."
        )
    out.parent.mkdir(parents=True, exist_ok=True)


def build_bundle(args) -> Path:
    sources = _declared_sources(args)
    registered_prefix = (
        validate_registered_uid_prefix(args.dicom_uid_prefix)
        if args.dicom_uid_prefix is not None
        else None
    )
    artifact_path = _requested_artifact_path(args)
    resolved = [(source, _resolve_source(source, artifact_path)) for source in sources]

    checked = []
    reference_spec = None
    reference_inference = None
    seen_run_ids = set()
    allowed_arch_ids = MODEL_CONFIGS[args.model_type]["arch_ids"]

    for source, path in resolved:
        metadata = read_safetensors_metadata(path)
        if metadata["arch_id"] not in allowed_arch_ids:
            raise ValueError(
                f"member {source.member_id!r} has arch_id={metadata['arch_id']!r}, "
                f"which does not match model type {args.model_type!r}"
            )
        if metadata["artifact_role"] != args.artifact_role:
            raise ValueError(
                f"member {source.member_id!r} has role {metadata['artifact_role']!r}; "
                f"expected {args.artifact_role!r}"
            )
        run_id = metadata.get("mlflow_run")
        if not isinstance(run_id, str) or not run_id:
            raise ValueError(f"member {source.member_id!r} has no MLflow run id in its metadata")
        if source.run_id and run_id != source.run_id:
            raise ValueError(
                f"member {source.member_id!r} metadata run id {run_id!r} does not match "
                f"the requested run {source.run_id!r}"
            )
        if run_id in seen_run_ids:
            raise ValueError(f"MLflow run {run_id!r} was declared more than once")
        seen_run_ids.add(run_id)

        spec = {
            "artifact_schema": metadata["artifact_schema"],
            "arch_id": metadata["arch_id"],
            "arch_kwargs": metadata["arch_kwargs"],
            "wrapper_spec": metadata["wrapper_spec"],
        }
        inference_config = metadata["inference_config"]
        _validate_inference_config(inference_config)
        if reference_spec is None:
            reference_spec = spec
            reference_inference = inference_config
        elif spec != reference_spec:
            raise ValueError(
                f"member {source.member_id!r} has a different model specification from "
                "the first declared member"
            )
        elif inference_config != reference_inference:
            raise ValueError(
                f"member {source.member_id!r} has a different inference configuration from "
                "the first declared member"
            )

        # Strict reconstruction/load proves that the bundle preparation runtime supports the
        # declared architecture and that the file is not merely a readable Safetensors header.
        validated_model = load_safetensors_model(path, device="cpu")
        del validated_model
        checked.append((source, path, metadata))

    out = (args.out or SCRIPT_DIR / "model_bundles" / args.model_type).resolve()
    _validate_output_target(out)
    output_mode = out.stat().st_mode & 0o777 if out.exists() else 0o755
    staging = Path(
        tempfile.mkdtemp(prefix=f".{out.name}-build-", dir=out.parent)
    )
    try:
        staging.chmod(output_mode)
        members = []
        for source, src, metadata in checked:
            artifact_name = bundle_member_filename(source.member_id)
            dst = staging / artifact_name
            shutil.copy2(src, dst)
            members.append({
                "member_id": source.member_id,
                "sha256": _sha256_file(dst),
            })

        manifest = {
            "schema_version": DEPLOYMENT_SCHEMA,
            "model_type": args.model_type,
            "members": members,
        }
        if registered_prefix is not None:
            manifest["registered_prefix"] = registered_prefix
        manifest["bundle_sha256"] = _bundle_sha256(
            DEPLOYMENT_SCHEMA,
            args.model_type,
            [member["sha256"] for member in members],
        )
        staged_manifest = staging / "deployment_config.json"
        staged_manifest.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        if out.exists():
            out.rmdir()
        staging.rename(out)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    manifest_path = out / "deployment_config.json"

    deployment_label = "single-model" if len(members) == 1 else "ensemble"
    print(f"Built {deployment_label} bundle with {len(members)} declared member(s): {out}")
    for member, (_, _, metadata) in zip(members, checked):
        print(
            f"  {member['member_id']}: {bundle_member_filename(member['member_id'])} "
            f"(run {metadata['mlflow_run']})"
        )
    print(f"Deployment config: {manifest_path}")
    return manifest_path


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="PrepareModelBundle",
        description="Build an explicit single-model or ensemble Safetensors bundle.",
    )
    parser.add_argument("--model-type", required=True, choices=tuple(MODEL_CONFIGS))
    parser.add_argument(
        "--dicom-uid-prefix",
        default=None,
        help=(
            "Optional registered numeric UID prefix reserved for this generator, without "
            "a trailing period. Omit it to use deterministic 2.25 UUID UIDs."
        ),
    )
    parser.add_argument(
        "--run", action="append", default=[], metavar="MEMBER_ID=RUN_ID",
        help="Explicit MLflow run member. Repeat for an ensemble.",
    )
    parser.add_argument(
        "--artifact", action="append", default=[], metavar="MEMBER_ID=PATH",
        help="Explicit already-downloaded Safetensors member. Repeat for an ensemble.",
    )
    parser.add_argument(
        "--artifact-path", default=None,
        help=(
            "Optional run-relative MLflow artifact override. By default it is derived "
            "from --artifact-role as model/<role>_model.safetensors."
        ),
    )
    parser.add_argument(
        "--artifact-role", required=True, choices=("best", "final"),
        help="Required artifact_role metadata value and default MLflow filename selector.",
    )
    parser.add_argument(
        "--out", type=Path, default=None,
        help=(
            "New/empty output directory (default: "
            "model_bundles/<model-type> beside this script)."
        ),
    )
    return parser


def main(argv=None):
    args = make_parser().parse_args(argv)
    build_bundle(args)


if __name__ == "__main__":
    main()
