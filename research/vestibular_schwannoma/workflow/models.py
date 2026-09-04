"""VS-specific training model definitions.

Model reconstruction itself is delegated to fastMONAI's versioned model-spec API,
which keeps training and Safetensors inference on the same construction path.
"""

from __future__ import annotations

import importlib
import warnings
from collections.abc import Callable, Iterable
from dataclasses import dataclass

import torch
from monai.losses import DeepSupervisionLoss, DiceCELoss

from fastMONAI.vision_all import (
    CustomLoss,
    build_model_from_spec,
    make_model_spec,
)


SEGMAMBA_INSTALL_HINT = (
    "GPU training: pip install 'segmamba-v2[gpu] @ "
    "git+https://github.com/skaliy/SegMamba-V2.git'; "
    "CPU inference: pip install 'segmamba-v2[cpu] @ "
    "git+https://github.com/skaliy/SegMamba-V2.git'"
)


@dataclass(frozen=True)
class TrainingModelConfig:
    """Everything the VS training workflow needs for one model family."""

    key: str
    display_name: str
    model_spec: dict
    loss_spec: dict
    make_loss: Callable[[], object]
    experiment_name: str
    supports_compile: bool = True

    @property
    def checkpoint_name(self) -> str:
        return f"best_{self.key}"


DICE_CE_LOSS_SPEC = {
    "loss_id": "monai.dice_ce",
    "kwargs": {
        "to_onehot_y": True,
        "softmax": True,
        "include_background": False,
        "batch": True,
    },
}
DYNUNET_LOSS_SPEC = {
    "loss_id": "monai.deep_supervision",
    "kwargs": {"weight_mode": "exp"},
    "base_loss": DICE_CE_LOSS_SPEC,
}


def _make_dice_ce_loss() -> CustomLoss:
    return CustomLoss(
        loss_func=DiceCELoss(
            to_onehot_y=True,
            softmax=True,
            include_background=False,
            batch=True,
        )
    )


def _make_dynunet_loss() -> CustomLoss:
    base_loss = DiceCELoss(
        to_onehot_y=True,
        softmax=True,
        include_background=False,
        batch=True,
    )
    return CustomLoss(loss_func=DeepSupervisionLoss(base_loss, weight_mode="exp"))


UNET_SPEC = make_model_spec(
    "monai.unet",
    {
        "spatial_dims": 3,
        "in_channels": 1,
        "out_channels": 2,
        "channels": (32, 64, 128, 256, 320),
        "strides": (2, 2, 2, 2),
        "num_res_units": 4,
        "norm": "INSTANCE",
        "act": ("LEAKYRELU", {"negative_slope": 0.01, "inplace": True}),
    },
)


def _make_dynunet_spec(filters: list[int]) -> dict:
    return make_model_spec(
        "monai.dynunet",
        {
            "spatial_dims": 3,
            "in_channels": 1,
            "out_channels": 2,
            "kernel_size": [[3, 3, 3]] * 5,
            "strides": [[1, 1, 1]] + [[2, 2, 2]] * 4,
            "upsample_kernel_size": [[2, 2, 2]] * 4,
            "filters": filters,
            "res_block": True,
            "deep_supervision": True,
            "deep_supr_num": 3,
        },
        wrapper_spec={
            "wrapper_id": "fastmonai.dynunet_ds_adapter",
            "wrapper_kwargs": {},
        },
    )


DYNUNET_SPEC = _make_dynunet_spec([32, 64, 128, 256, 320])
DYNUNET_SMALL_SPEC = _make_dynunet_spec([16, 32, 64, 128, 160])
DYNUNET_XS_SPEC = _make_dynunet_spec([8, 16, 32, 64, 80])

SEGMAMBA_SPEC = make_model_spec(
    "segmamba.v2",
    {
        "in_chans": 1,
        "out_chans": 2,
        "depths": [2, 2, 2, 2],
        "feat_size": [32, 64, 128, 256],
        "hidden_size": 320,
        "mamba_backend": "mamba_ssm",
    },
)


TRAINING_MODEL_CONFIGS = {
    "unet": TrainingModelConfig(
        key="unet",
        display_name="UNet",
        model_spec=UNET_SPEC,
        loss_spec=DICE_CE_LOSS_SPEC,
        make_loss=_make_dice_ce_loss,
        experiment_name="vestibular_schwannoma_unet",
    ),
    "dynunet": TrainingModelConfig(
        key="dynunet",
        display_name="DynUNet",
        model_spec=DYNUNET_SPEC,
        loss_spec=DYNUNET_LOSS_SPEC,
        make_loss=_make_dynunet_loss,
        experiment_name="vestibular_schwannoma_dynunet",
    ),
    "dynunet_small": TrainingModelConfig(
        key="dynunet_small",
        display_name="DynUNet Small (16-160)",
        model_spec=DYNUNET_SMALL_SPEC,
        loss_spec=DYNUNET_LOSS_SPEC,
        make_loss=_make_dynunet_loss,
        experiment_name="vestibular_schwannoma_dynunet_small",
    ),
    "dynunet_xs": TrainingModelConfig(
        key="dynunet_xs",
        display_name="DynUNet XS (8-80)",
        model_spec=DYNUNET_XS_SPEC,
        loss_spec=DYNUNET_LOSS_SPEC,
        make_loss=_make_dynunet_loss,
        experiment_name="vestibular_schwannoma_dynunet_xs",
    ),
    "segmamba": TrainingModelConfig(
        key="segmamba",
        display_name="SegMamba V2",
        model_spec=SEGMAMBA_SPEC,
        loss_spec=DICE_CE_LOSS_SPEC,
        make_loss=_make_dice_ce_loss,
        experiment_name="vestibular_schwannoma_segmamba",
        supports_compile=False,
    ),
}


def segmamba_available() -> bool:
    """Return whether the optional supported SegMamba fork can be imported."""

    try:
        module = importlib.import_module("models_segmamba.segmambav2")
        return getattr(module, "SegMamba", None) is not None
    except (ImportError, ModuleNotFoundError):
        return False


def get_training_model_configs(
    model_keys: Iterable[str], *, skip_unavailable: bool = True
) -> dict[str, TrainingModelConfig]:
    """Resolve declared model keys in order, optionally skipping missing SegMamba."""

    keys = list(model_keys)
    if len(keys) != len(set(keys)):
        raise ValueError("model_keys contains duplicates")
    unknown = sorted(set(keys) - set(TRAINING_MODEL_CONFIGS))
    if unknown:
        raise ValueError(f"Unknown model keys: {unknown}")

    configs = {}
    for key in keys:
        if key == "segmamba" and not segmamba_available():
            message = f"SegMamba was requested but is unavailable. {SEGMAMBA_INSTALL_HINT}"
            if not skip_unavailable:
                raise ImportError(message)
            warnings.warn(message, stacklevel=2)
            continue
        configs[key] = TRAINING_MODEL_CONFIGS[key]
    if not configs:
        raise RuntimeError("None of the requested models is available")
    return configs


def build_training_model(config: TrainingModelConfig, *, compile_model: bool):
    """Build from the persisted specification, then optionally compile for training."""

    model = build_model_from_spec(config.model_spec)
    return torch.compile(model) if compile_model and config.supports_compile else model
