# Vestibular Schwannoma ROR/PACS Container

This directory builds a PACS container from one declared Safetensors model or an explicit
ensemble. Current model types are `unet` and `dynunet`; TTA is enabled by default.

For shared ROR installation, versioned image builds, qualification, and PACS handoff, see
[Research PACS deployment with ROR](../../../RESEARCH_PACS_DEPLOYMENT.md).

## Prepare a model bundle

Use fastMONAI 0.10.0 or the pinned environment in `requirements.yml`. Set the run-ID variables
below to full MLflow run IDs.

One model trained on all data:

```bash
python prepare_model_bundle.py \
  --mode single \
  --model-type unet \
  --run "all_data=${ALL_DATA_RUN_ID}" \
  --artifact-role final
```

Five-fold ensemble:

```bash
python prepare_model_bundle.py \
  --mode ensemble \
  --model-type dynunet \
  --run "fold_1=${FOLD_1_RUN_ID}" \
  --run "fold_2=${FOLD_2_RUN_ID}" \
  --run "fold_3=${FOLD_3_RUN_ID}" \
  --run "fold_4=${FOLD_4_RUN_ID}" \
  --run "fold_5=${FOLD_5_RUN_ID}" \
  --artifact-role best
```

Repeat `--run MEMBER=RUN_ID` for any intentional ensemble size. To use downloaded models,
replace each `--run` with `--artifact MEMBER=/path/model.safetensors`. Local artifacts must
still contain matching MLflow run metadata. Output defaults to
`model_bundles/<model-type>/`; the directory must be new or empty.

The builder strict-loads every model and writes `deployment_config.json` with the exact
members, roles, run IDs, model/inference specifications, hashes, and resolved DICOM UID
contract.

## DICOM identity

`deployment_models.py` is the authoritative registry for persistent numeric model,
deployment, and output codes. Codes must never be reused for a different meaning. The
builder copies the resolved contract into each bundle's `deployment_config.json`, and the
runtime rejects a bundle whose contract no longer matches the registry.

Format version 1 uses deterministic numeric `2.25.<UUID integer>` Series and SOP Instance
UIDs. The UUID input binds the source series/instance, bundle hash, model code, deployment
code, member count, and output code. The readable identity remains in DICOM metadata:
`SeriesDescription` names the model and output, while `SoftwareVersions` records the bundle,
run IDs, and fastMONAI version.

| Code family | Value | Meaning |
| --- | ---: | --- |
| Model | 1 | UNet |
| Model | 2 | DynUNet |
| Deployment | 1 | Single model |
| Deployment | 2 | Ensemble |
| Output | 1 | Segmentation mask |
| Output | 2 | Foreground probability |

The probability DICOM stores `round(probability × 65535)` as uint16 and records the inverse
scale in `DerivationDescription`. It intentionally omits modality-rescale tags because the
deployed `pr2mask` vote-map reader expects the stored uint16 values and `--votemapmax 65535`.

## Build and run

After preparing every model type that should ship, run from this directory:

```bash
BUILD_VERSION="$(date -u +%Y%m%dT%H%M%SZ)"

docker build --pull \
  --build-arg conda_env=fastmonai \
  --build-arg VERSION="$BUILD_VERSION" \
  -f .ror/virt/Dockerfile \
  -t "vs-seg:$BUILD_VERSION" \
  -t "vs-seg:latest" \
  .
```

Both tags point to the same image. `latest` is a convenient pointer, while the dated tag remains
selectable after later builds and identifies the qualified release. Export both references so
the recipient can call either tag; the archive filename alone does not become an image tag.
Record the test against the exact dated image:

```bash
ror trigger -cont "vs-seg:$BUILD_VERSION" -each -keep
ror trigger -cont "vs-seg:$BUILD_VERSION" -each -keep \
  -envs '{"model-type":"dynunet","tta":false}'
```

Or run that image directly:

```bash
docker run --rm \
  -v /path/to/input:/data:ro \
  -v /path/to/output:/output \
  "vs-seg:$BUILD_VERSION"
```

Supported ROR options are `model-type` and `tta`. Unknown keys, malformed JSON, unsupported
models, and invalid Boolean values fail before inference.

The container returns `fused`, `fused_vote_map`, `reports`, and `mask` DICOM series. Before
release, run the deployment tests and validate both a single model and an ensemble on real
cases:

```bash
python -m unittest discover -s ../../tests -p 'test_*.py'
bash -n entrypoint.sh
```
