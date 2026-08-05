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

The fold weights are not committed to the repo (they are large, and unlike the untracked `vs_seg` tree this folder is tracked - see `.gitignore`). Generate `vs5f_unet_models/` on the build host before you build:

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate fastmonai-latest
python prepare_fold_models.py
```

Run this in an environment that can unpickle the source learners, meaning fastMONAI, fastai, monai and torchio are installed at compatible versions, and the Python minor version matches the container. See the environment note below for which of the local envs qualify.

The default `--experiment vs5f_unet` targets the experiment that notebook 01 writes; it does not exist yet, so a bare run finds nothing. The bundle currently shipped was built from `vs_five_fold_patch` with the `--pkl-paths` form shown below.

Each fold is re-exported rather than copied, with any `torch.compile` wrapper stripped. A fold trained under `torch.compile` pickles its dynamo state, and that state only works on the torch that wrote it: `load_learner` still succeeds, so the mismatch would not surface until the first forward pass, on a real case, inside the container. Stripping the wrapper is what makes the bundled `.pkl` portable across torch versions.

MLflow discovery downloads artifacts to a temp dir, and the run id is parsed from the artifact path, so `--experiment` alone writes fold numbers into `mlflow_run_ids.txt` instead of run ids. Pass the in-place `mlruns` paths, in fold order, to keep the real ids:

```bash
M=../../vs_seg/mlruns/3
python prepare_fold_models.py --pkl-paths \
  $M/<fold-1-run-id>/artifacts/model/best_learner.pkl \
  ... one per fold, in order
```

As of 2026-08-03, `fastmonai-dev` and `fastmonai-latest` are valid export environments: both are Python 3.11 and both agree on the four packages the pickles bind to (fastMONAI 0.9.2, fastai 2.8.7, monai 1.6.0, torchio 1.2.1). They differ only in torch, which does not matter once the compile wrapper is stripped.

**`fastmonai-py312` is not valid.** It is Python 3.12, and `requirements.yml` pins the container to `python=3.11`. The exported learner carries `EmptyMedPatchDataLoaders`, a class defined inside `MedPatchDataLoaders.new_empty()`, so cloudpickle serialises it by value including its CPython bytecode. A bundle exported under 3.12 unpickles cleanly under 3.11 and then segfaults on first use, inside the container, on a real case. Match the export environment's Python minor version to `requirements.yml`, and keep fastMONAI and fastai matched too.

This writes three things into `vs5f_unet_models/`:

- `fold_1.pkl .. fold_5.pkl` - the fold learners, re-exported with any `torch.compile` wrapper removed.
- `inference_patch_config.json` - the full patch-inference config, sourced from the canonical training config `../inference_patch_config.json`.
- `mlflow_run_ids.txt` - one fold run-id per line, used for DICOM provenance.

To check a bundle by hand, `load_learner` a fold and read `type(learn.model).__name__`: it must be the network (`UNet`), never `OptimizedModule`.

Only the config has a manual equivalent. The fold `.pkl` must go through `prepare_fold_models.py`, which strips the `torch.compile` wrapper; copying `best_learner.pkl` straight across reintroduces it.

```bash
mkdir -p vs5f_unet_models
cp ../inference_patch_config.json vs5f_unet_models/inference_patch_config.json
```

`inference_patch_config.json` is read at inference with `load_patch_variables` and rebuilt into a full `PatchConfig(**config)`, so `patch_size`, `patch_overlap`, `aggregation_mode`, `apply_reorder`, `target_spacing`, `keep_largest_component` and `normalization` all come from the one training config - the single source of truth. A retrain that changes any of them then propagates to inference automatically (unlike vs_seg, which persists only `[patch_size, apply_reorder, target_spacing]` and hardcodes the rest in the stub).

As of 2026-08-03 the bundled folds come from MLflow experiment `vs_five_fold_patch`, which holds all five (`tags.fold` 1-5, one `dataset_version`). That experiment also contains a `final_all_data_clean` run trained on every case; it is deliberately excluded, and needs no special handling - `find_fold_learners` selects on `tags.fold`, which that run does not carry. The stub ensembles whatever N folds are present, so a partial bundle silently produces an N-fold ensemble: check the `Folds ensembled:` line in the run output.

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

## 7. Version pinning (important)

A `.pkl` learner unpickles by importing the classes it references, so the container needs fastMONAI, fastai, monai and torchio installed at compatible versions. It is **not** bound to a particular torch: once `prepare_fold_models.py` has stripped the `torch.compile` wrapper, the pickle loads and runs on any torch (verified in both directions, a bundle written under torch 2.12.1 runs under 2.8.0 and vice versa). `requirements.yml` pins the whole stack explicitly for build reproducibility rather than letting the resolver choose:

```yaml
- --extra-index-url https://download.pytorch.org/whl/cpu
- torch==2.12.1+cpu
- torchvision==0.27.1+cpu
- fastMONAI==0.9.2
- fastai==2.8.7
- monai==1.6.0
- torchio==1.2.1
```

Two things to know before changing these:

- **torch must come from the CPU index.** `monai==1.6.0` requires `torch>=2.8`. A conda `pytorch=2.6.0` + `cpuonly` pin (as used previously) is silently overridden by pip, which then pulls the default CUDA wheel and several GB of `nvidia-*` packages into what is supposed to be a CPU-only image. The torch pin does **not** have to track the environment the folds were exported from; any recent CPU build works.
- **Re-check after every retrain.** Confirm the export environment's Python minor version and its `fastMONAI` / `fastai` / `monai` / `torchio` versions are compatible with the block above. Python matters because the pickle embeds cloudpickled bytecode (see section 3). A newer fastMONAI that adds classes the pinned version lacks would fail to unpickle.

Verify inside the built image before shipping:

```bash
docker run --rm --entrypoint bash vs-seg-5fold:latest -lc \
  'python -c "import torch;print(torch.__version__, torch.cuda.is_available())"'
```

Expect a `+cpu` build and `False`.
