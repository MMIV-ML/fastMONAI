# Vestibular schwannoma segmentation (CE-T1w)

Patch-based 3D segmentation of vestibular schwannoma in contrast-enhanced T1-weighted MRI:
five-fold cross-validation, inference and a Research PACS container.

| Path | Contents |
| --- | --- |
| `data/` | Case index with fixed folds, data preparation ([data/README.md](data/README.md)) |
| `scripts/train_5fold.py` | Five-fold training and evaluation |
| `scripts/merge_parallel_folds.py` | Combine folds trained in parallel |
| `notebooks/` | Walkthroughs of cross-validation and inference |
| `workflow/` | Shared configuration, training, evaluation and inference code |
| `deployment/pacs/` | Model bundle builder and PACS container ([README](deployment/pacs/README.md)) |
| `tests/` | Workflow and deployment tests |

## Setup

From the repository root, install this checkout and prepare the 344-case dataset
(see [data/README.md](data/README.md)):

```bash
pip install -e '.[dev]'
cd research/vestibular_schwannoma
python scripts/prepare_data.py download
python scripts/prepare_data.py prepare
```

UNet and DynUNet use MONAI. SegMamba is optional; training prints the install commands
(`SEGMAMBA_INSTALL_HINT` in `workflow/models.py`) when it is requested but missing.

## Training

Run from `research/vestibular_schwannoma/`. The default trains UNet, DynUNet and SegMamba on
all five folds for 500 epochs; see `--help` for all options.

```bash
python scripts/train_5fold.py --models unet --folds 1 --epochs 5 --no-compile  # quick test
python scripts/train_5fold.py --models dynunet
python scripts/train_5fold.py --skip-unavailable
```

`dynunet_small` and `dynunet_xs` are DynUNet with half and a quarter of the channels.

To use several GPUs, run one fold per process, each with its own `--results-root`, then merge:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_5fold.py --models unet --folds 1 --results-root cv_results/unet_fold_1 &
# ... folds 2-5 on GPUs 1-4
wait
python scripts/merge_parallel_folds.py cv_results/unet_fold_{1..5}/completed_run_ids.json \
  --model unet --output-root cv_results/unet_5fold_merged
```

- Fill `preprocessed/` with a single process first; parallel jobs can't build the cache.
- All jobs must use the same MLflow tracking store. From one checkout this is automatic; across
  machines set the same `MLFLOW_TRACKING_URI` everywhere.
- Folds from different machines, or from notebook 01 and the script, merge if every machine
  prepared and verified the data with `scripts/prepare_data.py` and all jobs use the same
  commit, conda environment and training settings, including `--queue-workers` and
  `--queue-length`.
- Training and merging never overwrite an existing output folder. To replace a fold, train it
  into a new folder and merge again.

## Inference and results

- Inference loads the models listed in `cv_results/<run>/inference_run_ids.json` and applies
  their preprocessing. TTA is on by default and every predicted region is kept. Predictions
  need clinical review.
- Fold checkpoints are in `<results-root>/<model>/fold_<n>/checkpoints/` (warm start only).
  MLflow stores the final and best weights; inference and deployment load `.safetensors`.
- Data, results, tracking stores, checkpoints and bundles are not in Git.

## Citation and license

Part of fastMONAI: use its `CITATION.cff` and Apache-2.0 license, and cite the datasets under
their own terms.
