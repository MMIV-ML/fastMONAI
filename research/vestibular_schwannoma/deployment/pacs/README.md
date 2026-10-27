# Vestibular Schwannoma ROR/PACS Container

Build the CE-T1w vestibular schwannoma container with the five-fold DynUNet ensemble. The
UNet bundle was retired in October 2026 because it had more false positives than DynUNet.
Multiple Safetensors members run concurrently in separate CPU workers with three
threads per model. Five-fold deployment therefore requires at least 15 available CPU
threads; the qualified target has 16. TTA is enabled by default and averages two passes, the
original and one copy flipped anterior-posterior and inferior-superior together (`TTA_FLIP_AXES`
in `parallel_inference.py`), as in the nnU-Net container. On the 344 held-out cases this gave
about the same false positives as all 8 flips at a quarter of the cost.

See [Research PACS deployment with ROR](../../../RESEARCH_PACS_DEPLOYMENT.md)
for the general build, qualification, export, and handoff workflow.

## Prepare model bundles

Use the environment pinned in `requirements.yml` to prepare bundles. Strict model
metadata checks also depend on the PyTorch version; matching fastMONAI alone is
insufficient. Provide complete MLflow run IDs.

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

The intermediate `model_mask` and `vote_map` series use deterministic `2.25` UIDs.
Published series keep pr2mask's UIDs.

## Test and build

Run the deployment tests first (the test discovery runs from `research/`):

```bash
conda activate fastmonai
(cd ../../.. && python -m unittest discover -s vestibular_schwannoma/tests/deployment -t .)
bash -n entrypoint.sh
```

Build dated and `latest` tags for the same image:

```bash
BUILD_VERSION="$(date -u +%Y%m%dT%H%M%SZ)"

docker build --pull \
  --build-arg VERSION="$BUILD_VERSION" \
  -f .ror/virt/Dockerfile \
  -t "fastmonai-vsseg:$BUILD_VERSION" \
  -t "fastmonai-vsseg:latest" \
  .
```

Qualify and deliver the dated tag:

```bash
ror trigger -cont "fastmonai-vsseg:$BUILD_VERSION" -each -keep \
  -envs '{"model-type":"dynunet","tta":true}'
```

As a ror workflow, add this entry to the select statements. Release packages also
include it as `ror_select_entry.txt`. Replace `ENTRY_NAME` with the workflow `name`
in the PACS `config.json` and fill in the series selector:

```json
"ENTRY_NAME": {
  "select": "<VS CE-T1 series selector>",
  "ROR_CONT_OPTIONS": "{}",
  "docker_image": "fastmonai-vsseg:<BUILD_VERSION>"
},
```

`"{}"` runs the defaults. Use `"{\"tta\":false}"` to turn TTA off. Don't write `"\"\""`
here: that passes two quote characters and is rejected.

## Runtime contract

`ROR_CONT_OPTIONS` accepts:

- `model-type`: `dynunet` (the default and only model).
- `tta`: JSON Boolean, default `true`.

Unknown keys and invalid values fail before inference. Header preflight rejects
inconsistent Study, Series, SOP, modality, or geometry data; nonstandard source UIDs and
missing optional Frame of Reference metadata produce aggregated warnings.

The runtime writes intermediate `model_mask` and `vote_map` DICOM series. Fiona's `pr2mask`
reads them and creates `fused`, `fused_vote_map`, `reports`, and its own mask series in
`labels/`, using pr2mask's folder layout. fastMONAI provenance (SeriesDescription,
SoftwareVersions, DerivationDescription, and a `MASK` ImageType value) is added to the
`labels` series before it is published. Its study identity, window, and UIDs stay as pr2mask
wrote them, so the PACS files the mask with the other pr2mask outputs. `model_mask` and
`vote_map` are not published. Vote-map probabilities are `round(probability x 65535)`.
pr2mask draws values from 32768 up in blue, so values inside the mask are raised to at
least 32768 and values outside it lowered to at most 32767: the blue region is exactly the
mask, and yellow marks voxels with some foreground probability outside it. Existing `fused`, `fused_vote_map`, `reports`, `labels`, and `redcap` directories are rejected
to prevent overwrites.

### REDCap output

The final output includes pr2mask's `redcap/<mask-series-UID>/output.json` and
`output_data_dictionary.zip`, in pr2mask's EAV row format and instrument. `redcap` is an
owned output directory and must not already exist. There is no direct REDCap uploader.
The exporter sends only pr2mask's own columns, in the model's permanent `pr2mask` repeat
instance. An empty measurement list is routed with the source PatientID and
ReferringPhysicianName (`EventName:` prefix removed), matching pr2mask routing. The mask
itself goes to the PACS as the `labels` series.

Up to 20261026T134845Z the exporter also sent eight `vs_*` fields, among them the mask as
`vs_mask_json`. They are no longer sent.

### Stable model destinations and overwrites

`redcap_model_instances.json` assigns each full bundle SHA-256 a permanent positive
integer of at least 10. Instances 1-9 of the `pr2mask` instrument are left to the original
(manual) mask and pr2mask's own region numbering. DynUNet uses instance 11, and the nnU-Net
large container in the separate nnU-Net repository uses 10. Instance 12 is free for the next
model. They moved from 3 and 2 on 2026-10-21, and results imported before that stay at the
old instances. The retired UNet bundle was registered at instance 2.
Keep allocations unique across both projects, and don't renumber or reuse entries after
REDCap imports. Register each new bundle with an unused integer before building a release;
unknown bundles, duplicate allocations and instances below 10 fail closed.
The registry is copied into the container and validated before inference.

All fields for a model share its repeat instance. Within the same record/event:

- The same bundle overwrites its previous result, including when TTA, deployment
  version, timestamp or the predicted mask changes.
- A different bundle uses a different instance, including different weights of
  the same architecture.

pr2mask writes each region as its own repeat instance (1, 2, 3, ...), which would collide
with the model instances, so regions are not sent as separate instances. pr2mask's columns
in the model instance hold the **largest** region (most voxels; ties keep the lower region
number). pr2mask numbers regions in scan order, so the largest is not always region 1. An
empty mask has no regions, so pr2mask lists no columns; the exporter then takes them from
`pr2mask_dictionary.csv` and sends them blank, so the importer must overwrite blanks to
clear values from an earlier run.

Since the Fiona base of 2026-10-19, pr2mask also stores the mask in seven `mask_*` fields,
which fill the columns for the largest region. `mask_base64_gzip_compressed` is about 84,000
characters for a 512×512×120 scan, and REDCap cuts values longer than 65,535 characters, so
it only decodes for small volumes. The full mask is in the PACS `labels` series.

`pr2mask_dictionary.csv` holds the `instrument.csv` rows of a non-empty run of the packaged
pr2mask. When the Fiona base changes, regenerate it from a smoke test and check that a
non-empty run lists the same columns.

Install pr2mask's dictionary in the receiving REDCap project and qualify a
PACS-to-REDCap round trip before use. The importer must honor the supplied numeric
repeat instance and update existing values. Live importer behavior and field limits
have not been verified. The record/event must already identify the intended case/scan;
multiple scans routed to the same record/event and model intentionally share a slot.
If earlier prototype exports were imported, reconcile those old instances before
using the permanent mapping.
