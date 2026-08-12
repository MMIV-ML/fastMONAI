"""Model registry shared by bundle preparation and PACS inference.

The DICOM codes are persistent identifiers. Never reuse a code for another
meaning; add a new code when a model, deployment mode, or output type is added.
"""

MODEL_ARCH_IDS = {
    "unet": frozenset({"monai.unet"}),
    "dynunet": frozenset({"monai.dynunet"}),
}


DICOM_UID_FORMAT_VERSION = 1
DICOM_UID_ROOT = "2.25"
DICOM_UID_NAMESPACE = "632c9357-425e-586e-9c02-d19e87080ad1"

DICOM_DEPLOYMENT_CODES = {
    "single": 1,
    "ensemble": 2,
}

DICOM_OUTPUT_CODES = {
    "segmentation": 1,
    "probability": 2,
}


MODEL_CONFIGS = {
    "unet": {
        "models_dir": "model_bundles/unet",
        "display_name": "UNet",
        "dicom_model_code": 1,
    },
    "dynunet": {
        "models_dir": "model_bundles/dynunet",
        "display_name": "DynUNet",
        "dicom_model_code": 2,
    },
}


if MODEL_ARCH_IDS.keys() != MODEL_CONFIGS.keys():
    raise RuntimeError("deployment model registry is internally inconsistent")


def make_dicom_uid_contract(model_type: str, mode: str, member_count: int) -> dict:
    """Resolve the immutable numeric DICOM identity for one deployment bundle."""
    if model_type not in MODEL_CONFIGS:
        raise ValueError(f"unknown model type: {model_type!r}")
    if mode not in DICOM_DEPLOYMENT_CODES:
        raise ValueError(f"unknown deployment mode: {mode!r}")
    if not isinstance(member_count, int) or isinstance(member_count, bool) or member_count < 1:
        raise ValueError(f"member_count must be a positive integer, got {member_count!r}")
    return {
        "format_version": DICOM_UID_FORMAT_VERSION,
        "root": DICOM_UID_ROOT,
        "generation": "uuid5",
        "namespace_uuid": DICOM_UID_NAMESPACE,
        "model_code": MODEL_CONFIGS[model_type]["dicom_model_code"],
        "deployment_code": DICOM_DEPLOYMENT_CODES[mode],
        "member_count": member_count,
        "output_codes": dict(DICOM_OUTPUT_CODES),
    }


def _validate_registry() -> None:
    code_groups = {
        "model": [config["dicom_model_code"] for config in MODEL_CONFIGS.values()],
        "deployment": list(DICOM_DEPLOYMENT_CODES.values()),
        "output": list(DICOM_OUTPUT_CODES.values()),
    }
    for label, codes in code_groups.items():
        if any(not isinstance(code, int) or isinstance(code, bool) or code < 1
               for code in codes):
            raise RuntimeError(f"DICOM {label} codes must be positive integers")
        if len(codes) != len(set(codes)):
            raise RuntimeError(f"DICOM {label} codes must be unique")


_validate_registry()
