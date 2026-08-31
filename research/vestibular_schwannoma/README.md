# Vestibular Schwannoma Segmentation (CE-T1w)

This project uses fastMONAI for patch-based 3D segmentation of vestibular schwannoma in
contrast-enhanced T1-weighted MRI. It includes five-fold cross-validation, optional all-data
training, inference on new cases, and PACS deployment.

## Contents

- `train_5fold.py`: command-line five-fold training and evaluation.
- `merge_inference_manifests.py`: validate and combine parallel fold subsets for inference.
- `notebooks/01_five_fold_cross_validation.ipynb`: train and compare UNet, DynUNet, and
  optional SegMamba models.
- `notebooks/02_inference_new_cases.ipynb`: run one declared model or an explicit ensemble.
- `workflow/`: project-local configuration, training model definitions, result aggregation,
  and inference artifact handling shared by the CLI and notebooks.
- `data/ml_dataset.csv`: public case index and fixed fold assignments.
- `deployment/pacs/`: Safetensors bundle builder and ROR/PACS container.
- `tests/workflow/`: CPU-only workflow contract and orchestration tests.
- `tests/deployment/`: deployment tests.

## Setup and data

Use fastMONAI 0.10.1 or a matching development checkout. From the fastMONAI repository root:

```bash
pip install -e '.[dev]'
```

Medical images are not included. A reproducible workflow for preprocessing the downloaded
public datasets is in progress. Until it lands, the CSV expects prepared data under
`../nii_data/`; see [data/README.md](data/README.md).
UNet and DynUNet use MONAI. The training notebook prints the installation command for the
optional SegMamba fork when needed.

## Run

Use the CLI for unattended training. Examples for a quick check, one complete model, and the
full comparison:

```bash
python train_5fold.py --models unet --folds 1 --epochs 5 --no-compile
python train_5fold.py --models unet  # One model, all five folds
python train_5fold.py --skip-unavailable
```

The default requests three models across five folds for 500 epochs; run
`python train_5fold.py --help` before starting. A launcher processes its requested models
and folds sequentially. With five GPUs, run one process per fold, assign each process one
visible GPU, and give it a distinct, previously nonexistent `--results-root`:

```bash
CUDA_VISIBLE_DEVICES=0 python train_5fold.py --models unet --folds 1 --results-root cv_results/unet_fold_1 &
CUDA_VISIBLE_DEVICES=1 python train_5fold.py --models unet --folds 2 --results-root cv_results/unet_fold_2 &
CUDA_VISIBLE_DEVICES=2 python train_5fold.py --models unet --folds 3 --results-root cv_results/unet_fold_3 &
CUDA_VISIBLE_DEVICES=3 python train_5fold.py --models unet --folds 4 --results-root cv_results/unet_fold_4 &
CUDA_VISIBLE_DEVICES=4 python train_5fold.py --models unet --folds 5 --results-root cv_results/unet_fold_5 &
wait
```

Inside each process, its assigned physical GPU is exposed to PyTorch as CUDA device 0.

Before starting parallel jobs, populate `preprocessed/` once with a single process;
concurrent first-time cache creation is not supported. When `MLFLOW_TRACKING_URI` is unset,
processes launched on the same machine from the same fastMONAI checkout automatically share
fastMONAI's repository-root SQLite tracking store
(`sqlite:////absolute/path/to/fastMONAI/mlruns.db`). For multiple machines or a central
tracking service, configure the same remote URI in every shell:

```bash
export MLFLOW_TRACKING_URI=http://mlflow.example:5000
```

Do not merge run IDs from independent local MLflow databases: inference must be able to
resolve every run ID through one tracking URI.

Each launcher atomically updates `completed_run_ids.json` after every successful fold, so
completed work remains mergeable if a later fold is interrupted. A subset job intentionally
does not create `inference_run_ids.json`; its completed registry is also rejected by the
inference loader. Combine disjoint completed-fold registries into a new inference-only
results root. The merger requires exactly folds 1-5, verifies that dataset/splits,
preprocessing, model/loss, and training settings match, and rejects missing or overlapping
folds, duplicate MLflow run IDs, and an existing output root:

```bash
python merge_inference_manifests.py \
  cv_results/unet_fold_1/completed_run_ids.json \
  cv_results/unet_fold_2/completed_run_ids.json \
  cv_results/unet_fold_3/completed_run_ids.json \
  cv_results/unet_fold_4/completed_run_ids.json \
  cv_results/unet_fold_5/completed_run_ids.json \
  --model unet \
  --output-root cv_results/unet_5fold_merged
```

To replace only fold 1, train it into a new results root and merge that new
`completed_run_ids.json` with registries containing folds 2-5; omit the old fold-1
registry. The replacement is accepted only when its training contract matches, and the
merged `--output-root` must also be new.

Use `cv_results/unet_5fold_merged/inference_run_ids.json` in notebook 02. For interactive
inspection and visualizations, start Jupyter from this directory or `notebooks/`:

```bash
jupyter lab notebooks/01_five_fold_cross_validation.ipynb
```

Notebook 01 uses the fixed `fold` column for cross-validation and can optionally train one
final model on all cases. For all-data fitting, one stable case is duplicated only for fastai's
validation phase; it remains in training, does not select a best checkpoint, and is not held-out
evaluation. Notebook 02 reads the preprocessing and output contract embedded
in each declared Safetensors model. Completed training runs are handed to notebook 02 through
`cv_results/<RESULTS_RUN>/inference_run_ids.json`; models remain stored in MLflow.

Evaluation and inference preserve all predicted regions without size filtering and use TTA by
default. Predictions require clinical review.

The notebooks keep the scientific choices visible but delegate reusable project orchestration
to `workflow/`. Training model configs contain the VS-specific architecture and loss settings, while
model reconstruction, patch inference, metrics, and artifact formats remain fastMONAI
responsibilities.

Training writes selected fold checkpoints below
`<results-root>/<model>/fold_<n>/checkpoints/`, so different folds and models cannot overwrite
each other. All-data learners are independently scoped below
`<results-root>/<model>/all_data/`, but their final artifacts are stored in MLflow rather than as
a local checkpoint. The fold `.pth` files support warm-starting or further fitting with a newly
initialized optimizer and learning-rate schedule; they are not exact training-resume
checkpoints. Final and best inference artifacts remain isolated in their MLflow runs. Inference
and deployment use strict-loaded `.safetensors` artifacts. Generated data, results, tracking
stores, checkpoints, and model bundles are excluded from Git.

For container preparation and execution, see
[deployment/pacs/README.md](deployment/pacs/README.md).

## Citation and license

This project is part of fastMONAI. Use the parent repository's `CITATION.cff` and Apache-2.0
license, and cite the originating datasets under their applicable terms.
