# Vestibular Schwannoma Segmentation (CE-T1w)

This project uses fastMONAI for patch-based 3D segmentation of vestibular schwannoma in
contrast-enhanced T1-weighted MRI. It includes five-fold cross-validation, optional all-data
training, inference on new cases, and PACS deployment.

## Contents

- `notebooks/01_five_fold_cross_validation.ipynb`: train and compare UNet, DynUNet, and
  optional SegMamba models.
- `notebooks/02_inference_new_cases.ipynb`: run one declared model or an explicit ensemble.
- `workflow/`: project-local configuration, model recipes, training, result aggregation,
  and inference artifact handling used by the notebooks.
- `data/ml_dataset.csv`: public case index and fixed fold assignments.
- `deployment/pacs/`: Safetensors bundle builder and ROR/PACS container.
- `tests/workflow/`: CPU-only workflow contract and orchestration tests.
- `tests/deployment/`: deployment tests.

## Setup and data

Use fastMONAI 0.10.0 or a matching development checkout. From the fastMONAI repository root:

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
final model on all cases. Notebook 02 reads the preprocessing and output contract embedded
in each declared Safetensors model.

The notebooks keep the scientific choices visible but delegate reusable project orchestration
to `workflow/`. Model recipes contain the VS-specific architecture and loss settings, while
model reconstruction, patch inference, metrics, and artifact formats remain fastMONAI
responsibilities.

Training retains weights-only `.pth` checkpoints for continued training. Inference and
deployment use strict-loaded `.safetensors` artifacts. Generated data, results, tracking
stores, checkpoints, and model bundles are excluded from Git.

For container preparation and execution, see
[deployment/pacs/README.md](deployment/pacs/README.md).

## Reproducibility

Preprocessing cache reuse is validated through `preprocessing_manifest.json`. MLflow records
the dataset, preprocessing cache, actual DataLoader split, metrics, and model artifacts.

Run the workflow tests from the parent repository with the development environment active:

```bash
python -m unittest discover -s research/vestibular_schwannoma/tests/workflow -v
```

## Citation and license

This project is part of fastMONAI. Use the parent repository's `CITATION.cff` and Apache-2.0
license, and cite the originating datasets under their applicable terms.
