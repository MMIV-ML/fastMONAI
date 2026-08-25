# Vestibular Schwannoma Segmentation (CE-T1w)

This project uses fastMONAI for patch-based 3D segmentation of vestibular schwannoma in
contrast-enhanced T1-weighted MRI. It includes five-fold cross-validation, optional all-data
training, inference on new cases, and PACS deployment.

## Contents

- `notebooks/01_five_fold_cross_validation.ipynb`: train and compare UNet, DynUNet, and
  optional SegMamba models.
- `notebooks/02_inference_new_cases.ipynb`: run one declared model or an explicit ensemble.
- `workflow/`: project-local configuration, training model definitions, result aggregation,
  and inference artifact handling used by the notebooks.
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

Start Jupyter from this directory or `notebooks/`:

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

Training retains weights-only `.pth` checkpoints for warm-starting or further fitting with a
newly initialized optimizer and learning-rate schedule; they are not exact training-resume
checkpoints. Inference and deployment use strict-loaded `.safetensors` artifacts. Generated
data, results, tracking stores, checkpoints, and model bundles are excluded from Git.

For container preparation and execution, see
[deployment/pacs/README.md](deployment/pacs/README.md).

## Citation and license

This project is part of fastMONAI. Use the parent repository's `CITATION.cff` and Apache-2.0
license, and cite the originating datasets under their applicable terms.
