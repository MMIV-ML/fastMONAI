# Vestibular Schwannoma Segmentation (CE-T1w)

This folder reproduces our fastMONAI results for segmenting vestibular schwannoma (VS) in
contrast-enhanced T1-weighted (CE-T1w) MRI. The task is binary 3D segmentation: for every
voxel, tumor (1) or background (0), trained and evaluated with the patch-based workflow.

**Status: under development.** Downloading the data and preprocessing it into the layout the
notebooks expect are not part of this folder yet and will be added.

## Notebooks

- `01_five_fold_cross_validation.ipynb` - five-fold cross-validation comparing three models
  (UNet, DynUNet, and a SegMamba fork) on the same fixed folds. Each fold is trained and then
  evaluated with sliding-window inference, and the notebook ends with a cross-model summary.
- `02_inference_new_cases.ipynb` - runs a soft-vote ensemble of the five
  folds, on new cases, reusing the exact preprocessing contract from training.

## Data

`ml_dataset.csv` has one row per case: image and mask paths (relative to this folder, under
`../nii_data/...`), a pre-assigned `fold` (1..5), and a `split` column. The scans themselves
are not included here yet. The steps that download the data and build `../nii_data/` and this
CSV will be added (see Status).

## Requirements

An editable install of this repository (`pip install -e '.[dev]'` from the repo root). Both
notebooks resolve their paths from `fastMONAI.__file__` and `chdir` into
`research/vestibular_schwannoma`, so a plain PyPI install of fastMONAI will not work: there
is no `research/` directory under `site-packages`.

UNet and DynUNet are MONAI built-ins and need nothing extra. SegMamba needs our fork; section 1
("Environment setup") of `01_five_fold_cross_validation.ipynb` prints the one-line GitHub install
if the import fails.
