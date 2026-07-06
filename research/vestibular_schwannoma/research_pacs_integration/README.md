# UNet 5-Fold Ensemble ROR Container (Vestibular Schwannoma)

## 1. What this is

A ROR/PACS inference container for vestibular schwannoma segmentation that runs the five cross-validation UNet folds as a single soft-vote ensemble. `PatchInferenceEngine` receives the folds as a LIST and averages their per-patch softmax probabilities before the final argmax, so the folds behave as one logical model: the 5 folds averaged per patch.

This container mirrors the production `vs_seg` container at `research/vs_seg/fastmonai_inference/integration`. Beyond model loading (a list of fold learners instead of one) and the provenance (ensemble run-ids), it also loads the full patch config from one file (section 3) rather than vs_seg's 3-key settings plus hardcoded values. Preprocessing, sliding-window inference and the output contract are otherwise the same.

## 2. How it differs from vs_seg

| Aspect | vs_seg (`research/vs_seg/fastmonai_inference/integration`) | This container |
|--------|-----------------------------------------------------------|----------------|
| Model loading | one exported learner (`unet_models/final_unet_learner.pkl`) | a LIST of fold learners (`vs5f_unet_models/fold_*.pkl`) |
| Averaging | none (single model) | per-patch softmax averaged by `PatchInferenceEngine` |
| SoftwareVersions | single-model tag plus one run-id | ensemble tag plus every fold run-id |
| Entrypoint INFO | `unet: <run-id>` | `unet N-fold ensemble` (N = shipped folds) |
| DICOM UID suffixes | `UNS1` / `UNP1` | `U5S1` / `U5P1` |
| display_name | `UNet` | `UNet 5-fold ensemble` |

The distinct UID pair (`U5S1` segmentation, `U5P1` probability) means the ensemble series do not overwrite the vs_seg single-model series in PACS.

Ensemble provenance is read from `vs5f_unet_models/mlflow_run_ids.txt` (one fold run-id per line, read with a guard for a missing file). `SoftwareVersions` becomes `[model_type + "-" + str(n) + "fold"]` (for this container, `unet-<n>fold`) followed by the first 8 characters of each fold run-id and `"fastMONAI " + fastMONAI.__version__`, where `n` is the number of folds actually loaded. Apart from the documented changes (model loading, provenance, and the patch-config loading described in section 3), the container follows vs_seg.

## 3. Prepare weights (before building)

The fold weights are not committed to the repo (they are large, and unlike the untracked `vs_seg` tree this folder is tracked - see `.gitignore`). Generate `vs5f_unet_models/` on the build host before you build. In the `fastmonai-dev` conda env:

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate fastmonai-dev
python prepare_fold_models.py
```

This writes three things into `vs5f_unet_models/`:

- `fold_1.pkl .. fold_5.pkl` - the exported fold learners, copied AS-IS with no cleaning and no re-export, exactly like the vs_seg `unet_models/final_unet_learner.pkl`.
- `inference_patch_config.json` - the full patch-inference config, sourced from the canonical training config `../inference_patch_config.json`.
- `mlflow_run_ids.txt` - one fold run-id per line, used for DICOM provenance.

Equivalent manual step (copy each exported fold `best_learner.pkl`, plus the config):

```bash
mkdir -p vs5f_unet_models
cp /path/to/fold_1/best_learner.pkl vs5f_unet_models/fold_1.pkl
cp /path/to/fold_2/best_learner.pkl vs5f_unet_models/fold_2.pkl
# ... repeat for fold_3, fold_4, fold_5
cp ../inference_patch_config.json vs5f_unet_models/inference_patch_config.json
```

`inference_patch_config.json` is read at inference with `load_patch_variables` and rebuilt into a full `PatchConfig(**config)`, so `patch_size`, `patch_overlap`, `aggregation_mode`, `apply_reorder`, `target_spacing`, `keep_largest_component` and `normalization` all come from the one training config - the single source of truth. A retrain that changes any of them then propagates to inference automatically (unlike vs_seg, which persists only `[patch_size, apply_reorder, target_spacing]` and hardcodes the rest in the stub).

Note: only four UNet fold runs currently exist on disk, not a full five. The stub ensembles whatever N folds are present, so the container runs a 4-fold ensemble until the fifth fold is trained and prepared.

## 4. Build, run, export

These steps mirror `../../ROR_DOCKER_WORKFLOW.md` sections 3-7 with this container's values. Run every command from this integration folder.

Pull the base image (unchanged from vs_seg):

```bash
docker pull haukebartsch/fiona-component-python:latest
```

Build (Dockerfile `ARG VERSION=jul032026`):

```bash
docker build --build-arg conda_env="fastmonai" -f .ror/virt/Dockerfile -t vs-seg-5fold:latest .
```

Run through ror trigger:

```bash
ror trigger -cont vs-seg-5fold:latest -each -keep
```

Test-time augmentation (8-flip mirror) is on by default and improves accuracy, but it is much slower on CPU (~7-9x). Because this is an N-fold ensemble the cost stacks across folds: expect roughly 8-10 min/case without TTA and ~60-80 min/case with it for a 4-5 fold ensemble. Disable it with the `tta` key in `ROR_CONT_OPTIONS` (accepts true/false):

```bash
ror trigger -cont vs-seg-5fold:latest -each -keep -envs '{"tta":false}'
```

Manual docker run (mount input read-only at `/data`, output at `/output`):

```bash
docker run --rm \
  -v <INPUT_DIR>:/data:ro \
  -v <OUTPUT_DIR>:/output \
  vs-seg-5fold:latest
```

Export for distribution:

```bash
docker save vs-seg-5fold:latest | gzip > vs-seg-5fold_docker_image_jul032026.tar.gz
```

See `../../ROR_DOCKER_WORKFLOW.md` for ror install, project init, and the full ror trigger option tables.

## 5. Research PACS deployment

Deployment mirrors vs_seg. The select statement is the same VS MR study selection; only the `docker_image` changes to `vs-seg-5fold:latest`.

Select-statement entry:

```json
"MMIVVestSchAI": {
    "select": "SELECT study FROM study WHERE series named \"everything\" has Modality regexp \"MR\"",
    "ROR_CONT_OPTIONS": "{}",
    "docker_image": "vs-seg-5fold:latest"
}
```

Set `"ROR_CONT_OPTIONS": "{\"tta\":false}"` to disable TTA. The `config.json` trigger entry is unchanged from vs_seg. See `../../ROR_DOCKER_WORKFLOW.md` section 8 for the full fiona registration steps and the submission token.

## 6. Output

The container returns four series to PACS: `fused`, `fused_vote_map`, `reports`, and `mask`. The stub writes `/output/mask` (segmentation) and `/output/vote_map` (foreground probability); `entrypoint.sh` runs pr2mask, which produces `fused`, `fused_vote_map`, and `reports`. This is the same output contract as vs_seg.

## 7. Caveat

The exported `.pkl` fold learners are version-bound. The container's fastMONAI and fastai must match the versions used to train the folds, or the pickled learners will fail to load.
