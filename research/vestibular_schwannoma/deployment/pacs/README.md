# Vestibular Schwannoma ROR/PACS Container

This directory builds the CE-T1w vestibular schwannoma segmentation container.
It can ship `unet`, `dynunet`, or both. One Safetensors member is a single model;
multiple members form an ensemble and are evaluated sequentially. Eight-flip TTA
is enabled by default.

See [Research PACS deployment with ROR](../../../RESEARCH_PACS_DEPLOYMENT.md)
for the general build, qualification, export, and handoff workflow.

## Prepare model bundles

Use fastMONAI 0.10.1 or the pinned `fastmonai` environment from
`requirements.yml`. Supply complete MLflow run IDs.

One model trained on all data:

```bash
python prepare_model_bundle.py \
  --model-type unet \
  --run "all_data=${ALL_DATA_RUN_ID}" \
  --artifact-role final
```

Five-fold ensemble:

```bash
python prepare_model_bundle.py \
  --model-type dynunet \
  --run "fold_1=${FOLD_1_RUN_ID}" \
  --run "fold_2=${FOLD_2_RUN_ID}" \
  --run "fold_3=${FOLD_3_RUN_ID}" \
  --run "fold_4=${FOLD_4_RUN_ID}" \
  --run "fold_5=${FOLD_5_RUN_ID}" \
  --artifact-role best
```

Repeat `--run MEMBER=RUN_ID` for other ensemble sizes. For an existing local
artifact, use `--artifact MEMBER=/path/model.safetensors`. The builder validates
and strict-loads every member, then writes the ignored
`model_bundles/<model-type>/` directory.

Derived DICOM UIDs use deterministic `2.25` UIDs by default. A site that owns a
registered prefix reserved for this application can bind it at bundle creation:

```bash
python prepare_model_bundle.py ... --dicom-uid-prefix "<registered-prefix>"
```

## Test and build

Run the deployment tests first:

```bash
conda activate fastmonai
python -m unittest discover -s ../../tests/deployment -p 'test_*.py'
bash -n entrypoint.sh
```

Build dated and `latest` tags for the same image:

```bash
BUILD_VERSION="$(date -u +%Y%m%dT%H%M%SZ)"

docker build --pull \
  --build-arg VERSION="$BUILD_VERSION" \
  -f .ror/virt/Dockerfile \
  -t "vs-seg:$BUILD_VERSION" \
  -t "vs-seg:latest" \
  .
```

Qualify and deliver the dated tag:

```bash
ror trigger -cont "vs-seg:$BUILD_VERSION" -each -keep \
  -envs '{"model-type":"unet","tta":true}'

ror trigger -cont "vs-seg:$BUILD_VERSION" -each -keep \
  -envs '{"model-type":"dynunet","tta":true}'
```

## Runtime contract

`ROR_CONT_OPTIONS` accepts:

- `model-type`: `unet` (default) or `dynunet`.
- `tta`: JSON Boolean, default `true`.

Unknown keys and invalid values fail before inference. A header-only preflight
then rejects inconsistent Study, Series, SOP, modality, or geometry information.
Nonstandard source UID syntax and missing optional Frame of Reference metadata
produce aggregated warnings instead of repeated per-slice warnings.

The runtime writes `mask` and intermediate `vote_map` DICOM series. Fiona's
`pr2mask` tools then create `fused`, `fused_vote_map`, and `reports`; `vote_map`
is not published. Probability values are stored as `round(probability x 65535)`
for the vote-map reader. Existing `mask`, `fused`, `fused_vote_map`, and
`reports` directories are rejected to avoid overwriting previous results.
