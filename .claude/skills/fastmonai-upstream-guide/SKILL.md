---
name: fastmonai-upstream-guide
description: Consult MONAI, TorchIO, and nnU-Net documentation and source code for implementation guidance in fastMONAI. Use this skill proactively when implementing new transforms, loss functions, metrics, preprocessing pipelines, or patch-based workflows. Helps ensure fastMONAI implementations align with upstream library patterns and medical imaging best practices.
---

# fastMONAI Upstream Implementation Guide

This skill provides implementation guidance by consulting MONAI, TorchIO, and nnU-Net patterns via live documentation and source code fetching.

## When to Use This Skill

Use this skill proactively when:
- Implementing new transforms or augmentations
- Adding loss functions or metrics
- Working on patch-based training/inference workflows
- Establishing preprocessing or postprocessing pipelines
- Ensuring consistency with upstream library conventions

## Library-to-Module Mapping

| fastMONAI Module | Primary Upstream | What to Check |
|------------------|------------------|---------------|
| vision_augmentation | TorchIO | Transform classes, wrapper patterns |
| vision_metrics | MONAI + nnU-Net | Metric functions, accumulated patterns |
| vision_loss | MONAI | Loss class signatures, parameters |
| vision_patch | TorchIO | Queue, Sampler, GridAggregator |
| vision_core | TorchIO | ScalarImage, LabelMap, resampling |

## Documentation URLs

### TorchIO
- **Main docs:** https://torchio.readthedocs.io/
- **Transforms:** https://torchio.readthedocs.io/transforms/transforms.html
- **Augmentation:** https://torchio.readthedocs.io/transforms/augmentation.html
- **Preprocessing:** https://torchio.readthedocs.io/transforms/preprocessing.html
- **Patch-based:** https://torchio.readthedocs.io/patches/index.html

### MONAI
- **Main docs:** https://docs.monai.io/
- **Metrics:** https://docs.monai.io/en/stable/metrics.html
- **Losses:** https://docs.monai.io/en/stable/losses.html
- **Transforms:** https://docs.monai.io/en/stable/transforms.html

### nnU-Net v2
- **Repository:** https://github.com/MIC-DKFZ/nnUNet (complete rewrite of v1, Apache 2.0)
- **Documentation index:** https://github.com/MIC-DKFZ/nnUNet/tree/master/documentation
- **How to use:** https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/how_to_use_nnunet.md
- **Dataset format:** https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_format.md
- **Extending nnU-Net:** https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/extending_nnunet.md
- **Normalization explained:** https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/explanation_normalization.md
- **Plans files explained:** https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/explanation_plans_files.md
- **Pretraining/finetuning:** https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/pretraining_and_finetuning.md
- **Region-based training:** https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/region_based_training.md
- **Ignore label:** https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/ignore_label.md
- **ResEnc presets:** https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/resenc_presets.md

## Source Code URLs (for implementation details)

### TorchIO GitHub
- **Transforms base:** https://raw.githubusercontent.com/fepegar/torchio/main/src/torchio/transforms/transform.py
- **Spatial transforms:** https://raw.githubusercontent.com/fepegar/torchio/main/src/torchio/transforms/augmentation/spatial/
- **Intensity transforms:** https://raw.githubusercontent.com/fepegar/torchio/main/src/torchio/transforms/augmentation/intensity/
- **Queue:** https://raw.githubusercontent.com/fepegar/torchio/main/src/torchio/data/queue.py
- **GridSampler:** https://raw.githubusercontent.com/fepegar/torchio/main/src/torchio/data/sampler/grid.py

### MONAI GitHub
- **Metrics:** https://raw.githubusercontent.com/Project-MONAI/MONAI/dev/monai/metrics/meandice.py
- **Hausdorff:** https://raw.githubusercontent.com/Project-MONAI/MONAI/dev/monai/metrics/hausdorff_distance.py
- **Losses:** https://raw.githubusercontent.com/Project-MONAI/MONAI/dev/monai/losses/

### nnU-Net v2 GitHub (source code)
- **Package root:** https://github.com/MIC-DKFZ/nnUNet/tree/master/nnunetv2
- **Preprocessing root:** https://github.com/MIC-DKFZ/nnUNet/tree/master/nnunetv2/preprocessing
- **Normalization schemes:** https://raw.githubusercontent.com/MIC-DKFZ/nnUNet/master/nnunetv2/preprocessing/normalization/default_normalization_schemes.py
- **Resampling:** https://github.com/MIC-DKFZ/nnUNet/tree/master/nnunetv2/preprocessing/resampling
- **Training root:** https://github.com/MIC-DKFZ/nnUNet/tree/master/nnunetv2/training
- **Base trainer:** https://raw.githubusercontent.com/MIC-DKFZ/nnUNet/master/nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py
- **Trainer variants:** https://github.com/MIC-DKFZ/nnUNet/tree/master/nnunetv2/training/nnUNetTrainer/variants
- **Dice loss:** https://raw.githubusercontent.com/MIC-DKFZ/nnUNet/master/nnunetv2/training/loss/dice.py
- **Compound losses (DC+CE, DC+BCE, DC+TopK):** https://raw.githubusercontent.com/MIC-DKFZ/nnUNet/master/nnunetv2/training/loss/compound_losses.py
- **Robust CE loss:** https://raw.githubusercontent.com/MIC-DKFZ/nnUNet/master/nnunetv2/training/loss/robust_ce_loss.py
- **Deep supervision loss:** https://raw.githubusercontent.com/MIC-DKFZ/nnUNet/master/nnunetv2/training/loss/deep_supervision.py
- **Evaluation metrics:** https://raw.githubusercontent.com/MIC-DKFZ/nnUNet/master/nnunetv2/evaluation/evaluate_predictions.py
- **Sliding window inference:** https://raw.githubusercontent.com/MIC-DKFZ/nnUNet/master/nnunetv2/inference/sliding_window_prediction.py
- **Predict from raw data:** https://raw.githubusercontent.com/MIC-DKFZ/nnUNet/master/nnunetv2/inference/predict_from_raw_data.py

## Workflow Steps

1. **Identify the implementation area**
   - Transform/augmentation → TorchIO
   - Metric → MONAI (compute functions) + nnU-Net (accumulated patterns)
   - Loss function → MONAI
   - Patch workflow → TorchIO (Queue, GridSampler)
   - Preprocessing/normalization → nnU-Net v2 conventions
   - Sliding window inference → nnU-Net v2 (Gaussian weighting, tile step size)
   - Evaluation metrics → nnU-Net v2 (TP/FP/FN accumulation, Dice, IoU)

2. **Fetch relevant documentation**
   Use WebFetch to retrieve the appropriate documentation page from the URLs above.

3. **Fetch source code if needed**
   For implementation details, fetch the raw GitHub file to see:
   - Function signatures and parameters
   - Default values and conventions
   - Internal implementation logic

4. **Identify the pattern**
   - What parameters does the upstream function accept?
   - What tensor shapes does it expect?
   - What are the default behaviors?

5. **Apply to fastMONAI context**
   - Make changes in `nbs/*.ipynb` notebooks (not .py files)
   - Follow existing wrapper patterns
   - Run `nbdev_prepare` after changes

## fastMONAI-Specific Conventions

### Transform Wrappers
All transform wrappers must expose `.tio_transform` property:
```python
class MyTransform(DictTransform):
    def __init__(self, ...):
        self._transform = tio.SomeTransform(...)

    @property
    def tio_transform(self):
        return self._transform
```

### Padding Mode
Always use `padding_mode=0` (zero padding) for CropOrPad - this aligns with nnU-Net standards:
```python
tio.CropOrPad(target_size, padding_mode=0)  # Correct
tio.CropOrPad(target_size)  # Avoid - uses TorchIO default
```

### Tensor Shape Conventions
- MONAI metrics expect: [B, C, D, H, W]
- TorchIO uses: [C, D, H, W] for single images
- fastMONAI MedImage/MedMask: 4D tensors with affine tracking

### Accumulated Metrics (nnU-Net v2 pattern)
For patch-based training, use accumulated TP/FP/FN rather than per-batch averaging:
```python
class AccumulatedDice(Metric):
    # Accumulates across batches, not averages
```

### nnU-Net v2 Quick Reference

**Architecture:** The `nnunetv2/` package is organized into: preprocessing, training, inference, evaluation, experiment_planning, postprocessing, utilities.

**Normalization schemes** (in `default_normalization_schemes.py`):
- `ZScoreNormalization` - z-score with optional foreground masking (default for MR)
- `CTNormalization` - clip to 0.5th-99.5th percentiles then z-score (for CT)
- `NoNormalization` - passthrough (dtype conversion only)
- `RescaleTo01Normalization` - min-max to [0,1]
- `RGBTo01Normalization` - divide uint8 by 255

**Loss functions** (in `training/loss/`):
- `SoftDiceLoss(apply_nonlin, batch_dice, do_bg, smooth=1.0, ddp, clip_tp)` - standard soft Dice
- `MemoryEfficientSoftDiceLoss` - same API, ~1.6GB less memory via no-grad one-hot encoding
- `DC_and_CE_loss(soft_dice_kwargs, ce_kwargs, weight_ce=1, weight_dice=1, ignore_label)` - Dice + Cross-Entropy (default nnU-Net loss)
- `DC_and_BCE_loss(bce_kwargs, soft_dice_kwargs, weight_ce=1, weight_dice=1)` - Dice + Binary CE
- `DC_and_topk_loss(soft_dice_kwargs, ce_kwargs, weight_ce=1, weight_dice=1)` - Dice + TopK

**Evaluation metrics** (in `evaluation/evaluate_predictions.py`):
- Computes per-class: Dice, IoU, TP, FP, FN, TN
- `compute_tp_fp_fn_tn(mask_ref, mask_pred, ignore_mask)` - core confusion matrix
- `compute_metrics(reference_file, prediction_file, image_reader_writer, labels_or_regions, ignore_label)` - per-file metrics
- `compute_metrics_on_folder()` / `compute_metrics_on_folder_simple()` - batch evaluation with multiprocessing

**Sliding window inference** (in `inference/sliding_window_prediction.py`):
- `compute_steps_for_sliding_window(image_size, tile_size, tile_step_size)` - tile_step_size is 0-1 (0.5 = 50% overlap)
- `compute_gaussian(tile_size)` - Gaussian importance weighting for smooth tile aggregation (center-weighted, no zeros)
- Supports test-time augmentation via mirror axes

**Trainer variants** (in `training/nnUNetTrainer/variants/`):
Subdirectories for: benchmarking, data_augmentation, loss, lr_schedule, network_architecture, optimizer, sampling, training_length

### nbdev Workflow
- All code changes in `nbs/*.ipynb`
- Run `nbdev_prepare` to regenerate .py files and run tests
- Never edit `fastMONAI/*.py` directly

## Example Queries

When implementing, fetch relevant docs/code:

**"I need to implement a new spatial transform"**
→ Fetch TorchIO spatial transform docs and source

**"What's the signature for MONAI's compute_dice?"**
→ Fetch MONAI metrics documentation

**"How does nnU-Net handle validation metrics?"**
→ Fetch nnU-Net v2 evaluation code for TP/FP/FN accumulation patterns

**"What loss does nnU-Net v2 use by default?"**
→ Fetch compound_losses.py for DC_and_CE_loss (Dice + Cross-Entropy)

**"How does nnU-Net v2 do sliding window inference?"**
→ Fetch sliding_window_prediction.py for tile stepping and Gaussian weighting

**"What normalization does nnU-Net v2 use for MR images?"**
→ Fetch default_normalization_schemes.py for ZScoreNormalization

**"I want to add a new loss function"**
→ Fetch MONAI losses documentation and nnU-Net v2 compound_losses.py

## Quick Reference Commands

To check upstream implementations during development:

```
# TorchIO transform documentation
WebFetch: https://torchio.readthedocs.io/transforms/augmentation.html

# MONAI metric implementation
WebFetch: https://raw.githubusercontent.com/Project-MONAI/MONAI/dev/monai/metrics/meandice.py

# nnU-Net v2 usage guide
WebFetch: https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/how_to_use_nnunet.md

# nnU-Net v2 normalization schemes (ZScore, CT, RescaleTo01, etc.)
WebFetch: https://raw.githubusercontent.com/MIC-DKFZ/nnUNet/master/nnunetv2/preprocessing/normalization/default_normalization_schemes.py

# nnU-Net v2 default loss (Dice + CE compound)
WebFetch: https://raw.githubusercontent.com/MIC-DKFZ/nnUNet/master/nnunetv2/training/loss/compound_losses.py

# nnU-Net v2 Dice loss implementation
WebFetch: https://raw.githubusercontent.com/MIC-DKFZ/nnUNet/master/nnunetv2/training/loss/dice.py

# nnU-Net v2 evaluation metrics (Dice, IoU, TP/FP/FN)
WebFetch: https://raw.githubusercontent.com/MIC-DKFZ/nnUNet/master/nnunetv2/evaluation/evaluate_predictions.py

# nnU-Net v2 sliding window inference (Gaussian weighting)
WebFetch: https://raw.githubusercontent.com/MIC-DKFZ/nnUNet/master/nnunetv2/inference/sliding_window_prediction.py

# nnU-Net v2 base trainer (training loop, validation, scheduling)
WebFetch: https://raw.githubusercontent.com/MIC-DKFZ/nnUNet/master/nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py

# nnU-Net v2 extending guide (custom trainers, architectures)
WebFetch: https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/extending_nnunet.md
```
