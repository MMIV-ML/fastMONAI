# Vestibular Schwannoma ROR/PACS Container

Build the CE-T1w vestibular schwannoma container with `unet`, `dynunet`, or both.
Multiple Safetensors members form a sequential ensemble. Eight-flip TTA is enabled by default.

See [Research PACS deployment with ROR](../../../RESEARCH_PACS_DEPLOYMENT.md)
for the general build, qualification, export, and handoff workflow.

## Prepare model bundles

Use fastMONAI 0.10.1 or the environment pinned in `requirements.yml`. Provide complete
MLflow run IDs.

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

Repeat `--run MEMBER=RUN_ID` for other ensemble sizes, or use
`--artifact MEMBER=/path/model.safetensors` for a local artifact. The builder validates and
strict-loads each member into the ignored `model_bundles/<model-type>/` directory.

Derived DICOM UIDs use deterministic `2.25` values by default. To use a registered
application-specific prefix:

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

Unknown keys and invalid values fail before inference. Header preflight rejects
inconsistent Study, Series, SOP, modality, or geometry data; nonstandard source UIDs and
missing optional Frame of Reference metadata produce aggregated warnings.

The runtime writes `mask` and intermediate `vote_map` DICOM series. Fiona's `pr2mask` creates
`fused`, `fused_vote_map`, and `reports`; `vote_map` is not published. Vote-map probabilities
are `round(probability x 65535)`. Existing `mask`, `fused`, `fused_vote_map`, and `reports`
directories are rejected to prevent overwrites.
