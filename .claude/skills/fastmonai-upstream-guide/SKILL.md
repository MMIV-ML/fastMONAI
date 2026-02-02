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

### nnU-Net
- **Repository:** https://github.com/MIC-DKFZ/nnUNet
- **Documentation:** https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/

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

### nnU-Net GitHub
- **Preprocessing:** https://github.com/MIC-DKFZ/nnUNet/tree/master/nnunetv2/preprocessing
- **Training:** https://github.com/MIC-DKFZ/nnUNet/tree/master/nnunetv2/training

## Workflow Steps

1. **Identify the implementation area**
   - Transform/augmentation → TorchIO
   - Metric → MONAI (compute functions) + nnU-Net (accumulated patterns)
   - Loss function → MONAI
   - Patch workflow → TorchIO (Queue, GridSampler)
   - Preprocessing → nnU-Net conventions

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

### Accumulated Metrics (nnU-Net pattern)
For patch-based training, use accumulated TP/FP/FN rather than per-batch averaging:
```python
class AccumulatedDice(Metric):
    # Accumulates across batches, not averages
```

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
→ Fetch nnU-Net training code for accumulated patterns

**"I want to add a new loss function"**
→ Fetch MONAI losses documentation and source

## Quick Reference Commands

To check upstream implementations during development:

```
# TorchIO transform documentation
WebFetch: https://torchio.readthedocs.io/transforms/augmentation.html

# MONAI metric implementation
WebFetch: https://raw.githubusercontent.com/Project-MONAI/MONAI/dev/monai/metrics/meandice.py

# nnU-Net preprocessing patterns
WebFetch: https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/how_to_use_nnunet.md
```
