"""VS-specific model and loss recipes.

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
class ModelRecipe:
    """Everything the VS training workflow needs for one model family."""

    key: str
    display_name: str
    model_spec: dict
    make_loss: Callable[[], object]
    experiment_name: str
    checkpoint_name: str
    supports_compile: bool = True


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
        "channels": (64, 128, 256, 512, 1024),
        "strides": (2, 2, 2, 2),
        "num_res_units": 4,
        "norm": "INSTANCE",
        "act": ("LEAKYRELU", {"negative_slope": 0.01, "inplace": True}),
    },
)

DYNUNET_SPEC = make_model_spec(
    "monai.dynunet",
    {
        "spatial_dims": 3,
        "in_channels": 1,
        "out_channels": 2,
        "kernel_size": [[3, 3, 3]] * 5,
        "strides": [[1, 1, 1]] + [[2, 2, 2]] * 4,
        "upsample_kernel_size": [[2, 2, 2]] * 4,
        "filters": [64, 128, 256, 512, 1024],
        "res_block": True,
        "deep_supervision": True,
        "deep_supr_num": 3,
    },
    wrapper_spec={
        "wrapper_id": "fastmonai.dynunet_ds_adapter",
        "wrapper_kwargs": {},
    },
)

SEGMAMBA_SPEC = make_model_spec(
    "segmamba.v2",
    {
        "in_chans": 1,
        "out_chans": 2,
        "depths": [2, 2, 2, 2],
        "feat_size": [48, 96, 192, 384],
        "hidden_size": 768,
        "mamba_backend": "mamba_ssm",
    },
)


MODEL_RECIPES = {
    "unet": ModelRecipe(
        key="unet",
        display_name="UNet",
        model_spec=UNET_SPEC,
        make_loss=_make_dice_ce_loss,
        experiment_name="vs5f_unet",
        checkpoint_name="best_unet",
    ),
    "dynunet": ModelRecipe(
        key="dynunet",
        display_name="DynUNet",
        model_spec=DYNUNET_SPEC,
        make_loss=_make_dynunet_loss,
        experiment_name="vs5f_dynunet",
        checkpoint_name="best_dynunet",
    ),
    "segmamba": ModelRecipe(
        key="segmamba",
        display_name="SegMamba V2",
        model_spec=SEGMAMBA_SPEC,
        make_loss=_make_dice_ce_loss,
        experiment_name="vs5f_segmamba",
        checkpoint_name="best_segmamba",
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


def get_model_recipes(
    model_keys: Iterable[str], *, skip_unavailable: bool = True
) -> dict[str, ModelRecipe]:
    """Resolve declared model keys in order, optionally skipping missing SegMamba."""

    keys = list(model_keys)
    if len(keys) != len(set(keys)):
        raise ValueError("model_keys contains duplicates")
    unknown = sorted(set(keys) - set(MODEL_RECIPES))
    if unknown:
        raise ValueError(f"Unknown model keys: {unknown}")

    recipes = {}
    for key in keys:
        if key == "segmamba" and not segmamba_available():
            message = f"SegMamba was requested but is unavailable. {SEGMAMBA_INSTALL_HINT}"
            if not skip_unavailable:
                raise ImportError(message)
            warnings.warn(message, stacklevel=2)
            continue
        recipes[key] = MODEL_RECIPES[key]
    if not recipes:
        raise RuntimeError("None of the requested models is available")
    return recipes


def build_training_model(recipe: ModelRecipe, *, compile_model: bool):
    """Build from the persisted specification, then optionally compile for training."""

    model = build_model_from_spec(recipe.model_spec)
    return torch.compile(model) if compile_model and recipe.supports_compile else model
