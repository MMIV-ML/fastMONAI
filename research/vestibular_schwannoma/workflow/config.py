"""Configuration and pipeline factories for the VS segmentation experiments."""

from __future__ import annotations

import os
from dataclasses import dataclass

from fastMONAI.vision_all import GpuPatchAugmentation, PatchConfig


SUPPORTED_MODEL_KEYS = frozenset({"unet", "dynunet", "segmamba"})


@dataclass(frozen=True)
class ExperimentConfig:
    """Immutable settings shared by cross-validation and all-data training."""

    model_keys: tuple[str, ...] = ("unet", "dynunet", "segmamba")
    folds: tuple[int, ...] = (1, 2, 3, 4, 5)
    run_cross_validation: bool = True
    train_all_data: bool = False
    all_data_monitor_seed: int = 42
    epochs: int = 500
    batch_size: int = 4
    learning_rate: float = 1e-3
    use_tta: bool = True
    compile_models: bool = True
    target_spacing: tuple[float, float, float] = (0.4102, 0.4102, 1.5)
    patch_size: tuple[int, int, int] = (192, 192, 48)
    preprocess_workers: int = min(32, os.cpu_count() or 1)
    samples_per_volume: int = 4
    queue_num_workers: int = 16
    queue_length: int = 1200
    continue_on_error: bool = True

    def __post_init__(self) -> None:
        if not self.model_keys:
            raise ValueError("model_keys must contain at least one model")
        if len(set(self.model_keys)) != len(self.model_keys):
            raise ValueError("model_keys contains duplicates")
        unknown_models = sorted(set(self.model_keys) - SUPPORTED_MODEL_KEYS)
        if unknown_models:
            raise ValueError(f"Unknown model keys: {unknown_models}")

        if not self.run_cross_validation and not self.train_all_data:
            raise ValueError("Enable cross-validation, all-data training, or both")
        if self.run_cross_validation:
            if not self.folds:
                raise ValueError("folds must not be empty when cross-validation is enabled")
            if len(set(self.folds)) != len(self.folds):
                raise ValueError("folds contains duplicates")

        positive = {
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "preprocess_workers": self.preprocess_workers,
            "samples_per_volume": self.samples_per_volume,
            "queue_num_workers": self.queue_num_workers,
            "queue_length": self.queue_length,
        }
        invalid = [name for name, value in positive.items() if value <= 0]
        if invalid:
            raise ValueError(f"These settings must be positive: {invalid}")

        if len(self.target_spacing) != 3 or any(value <= 0 for value in self.target_spacing):
            raise ValueError("target_spacing must contain three positive values")
        if len(self.patch_size) != 3 or any(value <= 0 for value in self.patch_size):
            raise ValueError("patch_size must contain three positive values")


def make_patch_config(config: ExperimentConfig, normalization: list) -> PatchConfig:
    """Build the shared VS patch configuration from the declared experiment settings."""

    return PatchConfig(
        patch_size=list(config.patch_size),
        samples_per_volume=config.samples_per_volume,
        sampler_type="label",
        label_probabilities={0: 0.2, 1: 0.8},
        patch_overlap=0.5,
        keep_largest_component=True,
        target_spacing=list(config.target_spacing),
        preprocessed=True,
        normalization=normalization,
        aggregation_mode="hann",
        queue_num_workers=config.queue_num_workers,
        queue_length=config.queue_length,
    )


def make_gpu_augmentation(config: ExperimentConfig) -> GpuPatchAugmentation:
    """Build the VS GPU augmentation policy in voxel units for the target spacing."""

    spacing = config.target_spacing
    return GpuPatchAugmentation(
        affine={
            "scales": (0.7, 1.4),
            "degrees": (5, 5, 30),
            "translation": (
                25 / spacing[0],
                25 / spacing[1],
                5 / spacing[2],
            ),
            "default_pad_value": 0.0,
            "p": 0.2,
        },
        anisotropy={"axes": (0, 1, 2), "downsampling": (2, 4), "p": 0.25},
        flip={"axes": (0, 1, 2), "p": 0.5},
        gamma={"log_gamma": (-0.3, 0.3), "p": 0.3},
        intensity_scale={"scale_range": (0.75, 1.25), "p": 0.1},
        noise={"std": 0.1, "p": 0.1},
        blur={"std": (0.5, 1.0), "p": 0.2},
    )
