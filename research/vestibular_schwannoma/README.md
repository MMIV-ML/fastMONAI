# Vestibular Schwannoma Segmentation (CE-T1w)

Patch-based 3D segmentation of vestibular schwannoma in contrast-enhanced T1-weighted MRI,
with five-fold cross-validation, optional all-data training, inference, and PACS deployment.

## Contents

- `scripts/train_5fold.py`: command-line five-fold training and evaluation.
- `scripts/merge_parallel_folds.py`: validate and combine parallel fold subsets for inference.
- `notebooks/`: educational walkthroughs of cross-validation and inference.
- `workflow/`: shared configuration, training, evaluation, and artifact handling.
- `data/ml_dataset.csv`: public case index and fixed fold assignments.
- `scripts/prepare_data.py`: download, prepare, and verify the exact study dataset.
- `deployment/pacs/`: Safetensors bundle builder and ROR/PACS container.
- `tests/`: workflow and deployment tests.

## Setup and data

Use fastMONAI 0.10.1 or a matching development checkout. From the fastMONAI repository root:

```bash
pip install -e '.[dev]'
```

Download and prepare the 344-case dataset with `scripts/prepare_data.py`; see
[data/README.md](data/README.md) for scripted acquisition, corrected labels, and
reference verification. The default prepared layout is `../nii_data/`, matching
the shared CSV. UNet and DynUNet use MONAI. The training notebook provides
setup instructions for optional SegMamba support.

## Training

Use the CLI for unattended training. Run the commands below from
`research/vestibular_schwannoma/`; relative data and output paths resolve from that
project directory:

```bash
python scripts/train_5fold.py --models unet --folds 1 --epochs 5 --no-compile
python scripts/train_5fold.py --models unet  # One model, all five folds
python scripts/train_5fold.py --models dynunet_small dynunet_xs
python scripts/train_5fold.py --skip-unavailable
```

The default is three models, five folds, and 500 epochs. Models and folds run sequentially
within a launcher. Run `python scripts/train_5fold.py --help` for all options.

DynUNet variants use the same architecture, deep supervision, patch size, and training
settings. They differ only in channel widths: the existing `dynunet` uses
`[32, 64, 128, 256, 320]`, `dynunet_small` uses `[16, 32, 64, 128, 160]`, and
`dynunet_xs` uses `[8, 16, 32, 64, 80]`.

For one process per GPU, assign one visible GPU and a new `--results-root` to each process:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_5fold.py --models unet --folds 1 --results-root cv_results/unet_fold_1 &
CUDA_VISIBLE_DEVICES=1 python scripts/train_5fold.py --models unet --folds 2 --results-root cv_results/unet_fold_2 &
CUDA_VISIBLE_DEVICES=2 python scripts/train_5fold.py --models unet --folds 3 --results-root cv_results/unet_fold_3 &
CUDA_VISIBLE_DEVICES=3 python scripts/train_5fold.py --models unet --folds 4 --results-root cv_results/unet_fold_4 &
CUDA_VISIBLE_DEVICES=4 python scripts/train_5fold.py --models unet --folds 5 --results-root cv_results/unet_fold_5 &
wait
```

Populate `preprocessed/` with one process before launching parallel jobs; concurrent initial
cache creation is unsupported. Each process sees its assigned GPU as CUDA device 0.

All jobs must share one MLflow tracking store. Jobs from the same checkout do this
automatically when `MLFLOW_TRACKING_URI` is unset. For multiple machines, set the same remote
URI in every shell:

```bash
export MLFLOW_TRACKING_URI=http://mlflow.example:5000
```

Never merge run IDs from different tracking stores.

## Merge parallel folds

Each launcher updates `completed_run_ids.json` after every successful fold. A subset run is not
inference-ready. Merge disjoint registries into a new results root; the merger requires folds
1-5, a matching training contract, distinct run IDs, no overlaps, and a new output root:

```bash
python scripts/merge_parallel_folds.py \
  cv_results/unet_fold_1/completed_run_ids.json \
  cv_results/unet_fold_2/completed_run_ids.json \
  cv_results/unet_fold_3/completed_run_ids.json \
  cv_results/unet_fold_4/completed_run_ids.json \
  cv_results/unet_fold_5/completed_run_ids.json \
  --model unet \
  --output-root cv_results/unet_5fold_merged
```

Training and merging fail if their output directory already exists; they never overwrite
weights or manifests. For example, replace an existing `cv_results/unet_fold_1` like this:

```bash
python scripts/train_5fold.py \
  --models unet \
  --folds 1 \
  --results-root cv_results/unet_fold_1_retrained_20260902

python scripts/merge_parallel_folds.py \
  cv_results/unet_fold_1_retrained_20260902/completed_run_ids.json \
  cv_results/unet_fold_2/completed_run_ids.json \
  cv_results/unet_fold_3/completed_run_ids.json \
  cv_results/unet_fold_4/completed_run_ids.json \
  cv_results/unet_fold_5/completed_run_ids.json \
  --model unet \
  --output-root cv_results/unet_5fold_merged_20260902
```

Contracts must match. The original directory and MLflow run remain unchanged.

## Inference and artifacts

- Cross-validation uses fixed folds. Optional all-data training keeps every case in training;
  its duplicated validation case is only an internal fastai monitor.
- Inference loads declared models from `cv_results/<RESULTS_RUN>/inference_run_ids.json` and
  enforces their preprocessing and output contracts.
- Evaluation and inference use TTA by default and preserve every predicted region. Predictions
  require clinical review.
- Fold checkpoints live under `<results-root>/<model>/fold_<n>/checkpoints/`. They support
  warm-starting, not exact resume. MLflow stores the final/best artifacts; inference and
  deployment strict-load `.safetensors` files.
- Generated data, results, tracking stores, checkpoints, and bundles are excluded from Git.

For container preparation and execution, see
[deployment/pacs/README.md](deployment/pacs/README.md).

## Citation and license

This project is part of fastMONAI. Use the parent repository's `CITATION.cff` and Apache-2.0
license, and cite the originating datasets under their applicable terms.
