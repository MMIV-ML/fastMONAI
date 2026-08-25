"""Model and DICOM identity registry shared by bundle preparation and inference."""

from __future__ import annotations

import re


DEPLOYMENT_SCHEMA = 1


DICOM_UID_DEFAULT_PREFIX = "2.25"
# Permanent public namespace for schema 1's default UUID5 recipe. It is not a
# DICOM root or a secret. Do not change it: doing so changes every default UID.
DICOM_UID_NAMESPACE = "632c9357-425e-586e-9c02-d19e87080ad1"
DICOM_UID_MIN_SUFFIX_DIGITS = 30
DICOM_APPLICATION_ID = "fastmonai.vestibular_schwannoma.ce_t1w_segmentation"

# Persistent DICOM identity codes: add new values; never reuse existing numbers.
DICOM_OUTPUT_CODES = {
    "segmentation_mask": 1,
    "probability_map": 2,
}


MODEL_CONFIGS = {
    "unet": {
        "arch_ids": frozenset({"monai.unet"}),
        "display_name": "UNet",
        "dicom_model_code": 1,
    },
    "dynunet": {
        "arch_ids": frozenset({"monai.dynunet"}),
        "display_name": "DynUNet",
        "dicom_model_code": 2,
    },
}


_UID_RE = re.compile(r"^(0|[1-9][0-9]*)(\.(0|[1-9][0-9]*))*$")
_MEMBER_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def bundle_member_filename(member_id: str) -> str:
    """Return the conventional Safetensors filename for a safe bundle member ID."""
    if not isinstance(member_id, str) or not _MEMBER_ID_RE.fullmatch(member_id):
        raise ValueError(
            "member ID must start with a letter or number and contain only "
            "letters, numbers, '.', '_' or '-'"
        )
    return f"{member_id}.safetensors"


def validate_registered_uid_prefix(prefix: str) -> str:
    """Validate a registered numeric UID prefix reserved for this generator."""
    if not isinstance(prefix, str) or not prefix:
        raise ValueError("DICOM UID prefix must be a non-empty string")
    if prefix != prefix.strip():
        raise ValueError("DICOM UID prefix must not contain surrounding whitespace")
    if prefix.endswith("."):
        raise ValueError("DICOM UID prefix must not end with '.'")
    if not _UID_RE.fullmatch(prefix):
        raise ValueError(
            "DICOM UID prefix must contain canonical numeric components separated by '.'"
        )

    arcs = [int(value) for value in prefix.split(".")]
    if arcs[0] not in {0, 1, 2}:
        raise ValueError("DICOM UID prefix first component must be 0, 1, or 2")
    if len(arcs) < 2:
        raise ValueError("DICOM UID prefix must contain at least two components")
    if arcs[0] in {0, 1} and arcs[1] > 39:
        raise ValueError(
            "DICOM UID prefix second component must be 0 through 39 when the first is 0 or 1"
        )
    if prefix == DICOM_UID_DEFAULT_PREFIX or prefix.startswith(
        f"{DICOM_UID_DEFAULT_PREFIX}."
    ):
        raise ValueError("omit the prefix to use the default 2.25 UUID recipe")
    dicom_root = "1.2.840.10008"
    if prefix == dicom_root or prefix.startswith(f"{dicom_root}."):
        raise ValueError("1.2.840.10008 is reserved for DICOM-defined UIDs")

    suffix_digits = 64 - len(prefix) - 1
    if suffix_digits < DICOM_UID_MIN_SUFFIX_DIGITS:
        raise ValueError(
            "DICOM UID prefix is too long; it must leave at least "
            f"{DICOM_UID_MIN_SUFFIX_DIGITS} decimal suffix digits"
        )
    return prefix


def _validate_registry() -> None:
    for model_type, config in MODEL_CONFIGS.items():
        arch_ids = config.get("arch_ids")
        if not isinstance(arch_ids, frozenset) or not arch_ids or any(
            not isinstance(arch_id, str) or not arch_id for arch_id in arch_ids
        ):
            raise RuntimeError(f"{model_type!r} arch_ids must be a non-empty frozenset")

    code_groups = {
        "model": [config["dicom_model_code"] for config in MODEL_CONFIGS.values()],
        "output": list(DICOM_OUTPUT_CODES.values()),
    }
    for label, codes in code_groups.items():
        if any(not isinstance(code, int) or isinstance(code, bool) or code < 1
               for code in codes):
            raise RuntimeError(f"DICOM {label} codes must be positive integers")
        if len(codes) != len(set(codes)):
            raise RuntimeError(f"DICOM {label} codes must be unique")


_validate_registry()
