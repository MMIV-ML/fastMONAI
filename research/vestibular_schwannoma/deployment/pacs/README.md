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
  "docker_image": "fastmonai-vsseg:latest"
},
```

The entry uses `latest`, so a new release needs no config change. The release archive
carries both tags, and loading it moves `latest` to the new image. The dated tag shows
which release `latest` is.

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

pr2mask splits the mask into connected regions (26-connected) and numbers them 1, 2, 3, ...
in scan order. The exporter writes each region to its own repeat instance of the `pr2mask`
instrument: region k goes to the model's block base + k. DynUNet's base is 200, so a mask
with one region fills instance 201, and a mask with two regions fills 201 and 202. Instance
200 + k always holds pr2mask's "label k of N" (`mask_label_info`). Only instances that have a
region are written. An empty mask fills instance 201 with blank pr2mask columns, which records
that the model ran and found nothing. It is routed with the source PatientID and
ReferringPhysicianName (`EventName:` prefix removed), matching pr2mask routing.

Every written instance has all of pr2mask's columns plus `model_version`, for example
`dynunet 5-fold, fastmonai-vsseg:20261110T081500Z`. pr2mask itself has no such field. It is on
the REDCap form, and the exporter adds it to `output_data_dictionary.zip` as a text field. If
pr2mask ever writes its own `model_version`, the exporter's value replaces it. The mask itself
goes to the PACS as the `labels` series, whose DICOM header also records the full bundle hash
and the TTA setting.

pr2mask stores each region's mask in seven `mask_*` fields. `mask_base64_gzip_compressed`
covers the whole scan at 16 bits per voxel, so its length depends on the scan size, not the
region: about 84,000 characters for a 512×512×120 scan and 105,000 for 512×448×176. REDCap may
cut values longer than 65,535 characters; this has not been checked in the receiving project.
The exporter sends the value as pr2mask wrote it, and the full mask is always in the PACS
`labels` series.

`pr2mask_dictionary.csv` holds the `instrument.csv` rows of a non-empty run of the packaged
pr2mask. An empty mask, where pr2mask lists no columns, takes them from there. When the Fiona
base changes, regenerate it from a smoke test and check that a non-empty run lists the same
columns.

Up to 20261026T134845Z the exporter also sent eight `vs_*` fields, among them the mask as
`vs_mask_json`. They are no longer sent.

### Stable model destinations and overwrites

`redcap_model_instances.json` gives each full bundle SHA-256 a permanent block of 100
instances, stored as the block's base (a multiple of 100, at least 100). The DynUNet bundle
has block 200 (instances 201-299). Instances below 100 belong to the original (manual) mask
at 1, pr2mask's own region numbering and the earlier one-instance-per-model scheme: the nnU-Net
large container in the separate nnU-Net repository still writes its largest region to 10.
DynUNet wrote its largest region to 11 up to 20261027T083045Z (3 before 2026-10-21; the
retired UNet used 2); those results are no longer updated. Block 100 is planned for nnU-Net.
Keep blocks unique across both projects, and don't renumber or reuse them after REDCap
imports. Register each new bundle with an unused block before building a release. Unknown
bundles, duplicate blocks, invalid bases and region numbers of 100 or more fail closed. The
registry is copied into the container and validated before inference.

Within the same record/event:

- The same bundle overwrites its earlier instances, including when TTA, deployment version,
  timestamp or the predicted mask changes. A rerun that finds fewer regions leaves the higher
  instances of the earlier run as they were. The same bundle with the same TTA setting finds
  the same regions.
- A different bundle uses a different block, including different weights of the same
  architecture.

Install pr2mask's dictionary plus `model_version` in the receiving REDCap project. The import
of the one-instance release 20261027T083045Z has been confirmed; confirm instances 201+ and
`model_version` with a PACS-to-REDCap round trip. The importer must honor the supplied numeric
repeat instance and update existing values. The record/event must already identify the
intended case/scan; multiple scans routed to the same record/event and model intentionally
share the same instances.
