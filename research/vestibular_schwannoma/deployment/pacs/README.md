# Vestibular Schwannoma ROR/PACS Container

Build the CE-T1w vestibular schwannoma container with `dynunet`, `unet`, or both.
DynUNet is the default model.
Multiple Safetensors members run concurrently in separate CPU workers with three
threads per model. Five-fold deployment therefore requires at least 15 available CPU
threads; the qualified target has 16. Eight-flip TTA is enabled by default.

See [Research PACS deployment with ROR](../../../RESEARCH_PACS_DEPLOYMENT.md)
for the general build, qualification, export, and handoff workflow.

## Prepare model bundles

Use the environment pinned in `requirements.yml` to prepare bundles. Strict model
metadata checks also depend on the PyTorch version; matching fastMONAI alone is
insufficient. Provide complete MLflow run IDs.

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

- `model-type`: `dynunet` (default) or `unet`.
- `tta`: JSON Boolean, default `true`.

Unknown keys and invalid values fail before inference. Header preflight rejects
inconsistent Study, Series, SOP, modality, or geometry data; nonstandard source UIDs and
missing optional Frame of Reference metadata produce aggregated warnings.

The runtime writes `mask` and intermediate `vote_map` DICOM series. Fiona's `pr2mask` creates
`fused`, `fused_vote_map`, and `reports`; `vote_map` is not published. Vote-map probabilities
are `round(probability x 65535)`. Existing `mask`, `fused`, `fused_vote_map`, `reports`, and `redcap`
directories are rejected to prevent overwrites.

### Recoverable masks in REDCap JSON

The final output now includes `redcap/<report-series-UID>/output.json` and
`output_data_dictionary.zip`, using pr2mask's EAV row format and instrument.
`redcap` is an owned output directory and must not already exist.
The exporter reads the **written DICOM mask**, matches it to the source images by
geometry, and stores the full binary mask once in the model's permanent `pr2mask` repeat instance.
It also handles an empty measurement list using the source PatientID and
ReferringPhysicianName (`EventName:` prefix removed), matching pr2mask routing.

Added fields: `vs_mask_json` and `vs_measurements_json` (Notes), `vs_model_type`, `vs_bundle_sha256`,
`vs_prediction_id`, `vs_deployment_version`, `vs_tta`, and `vs_created_at`.
The existing data dictionary is extended; measurement definitions and OriginID
are retained. Compression happens in memory, so the mask is JSON text rather
than a separate binary attachment. There is no direct REDCap uploader.

`vs_mask_json` is a JSON string with schema version 1:

- Binary voxels: C-order `(slice, row, column)`, foreground 1, little-order bits,
  gzip-compressed and Base64-encoded (`packbits-gzip-base64-v1`).
- Shape, foreground count, and SHA-256 of the decoded C-order uint8 voxel bytes.
- DICOM LPS geometry: `pixel_spacing` in row/column order,
  `image_orientation_patient`, and `image_positions_patient` in encoded slice
  order. **All lengths are millimetres by schema convention; no units field.**
  Slice order follows increasing position along the DICOM orientation normal.
  Per-slice positions preserve irregular spacing; SliceThickness is not used
  as a substitute for the distance between slice centres.
- Source Study/Series/Frame of Reference and ordered SOP UIDs, mask series UID,
  model type, member IDs, bundle hash, deployment version and TTA.
- Stable prediction identity (hash of mask, geometry and provenance) and UTC
  export timestamp. The timestamp is excluded from the prediction identity.

Recover voxels with `redcap_output.decode_mask(json.loads(row['value']))` for the
`vs_mask_json` row. For voxel `[s, r, c]`, its LPS position is
`position[s] + r * pixel_spacing[0] * orientation[3:6] +
c * pixel_spacing[1] * orientation[0:3]`.
Use the same spatial grid as the reference before computing Dice. Distance
metrics must respect geometry; irregular grids require physical-coordinate
surface distances or a documented resampling step.

### Stable model destinations and overwrites

`redcap_model_instances.json` assigns each full bundle SHA-256 a permanent positive
integer. The current UNet bundle uses instance 1 and DynUNet uses instance 2.
Never renumber or reuse entries. Register each new bundle with an unused integer
before building a release; unknown bundles and duplicate allocations fail closed.
The registry is copied into the container and validated before inference.

All fields for a model share its repeat instance. Within the same record/event:

- The same bundle overwrites its previous result, including when TTA, deployment
  version, timestamp or the predicted mask changes.
- A different bundle uses a different instance, including different weights of
  the same architecture. `vs_prediction_id` remains provenance, not a routing key.

pr2mask's per-region measurements are preserved in `vs_measurements_json`, a JSON
list of `{redcap_repeat_instance, field_name, value}` entries. The instance inside
this list is the original **region** number, not the model destination. Measurements
are no longer emitted as separate top-level region instances or scalar measurement
fields: consumers of those fields must read this list instead. This avoids collisions
and ensures that reruns with fewer regions (or an empty mask) replace the entire old
measurement list without leaving stale regions. Existing dictionary definitions
remain alongside the new fields.

Install the extended dictionary in the receiving REDCap project and qualify a
PACS-to-REDCap round trip before use. The importer must honor the supplied numeric
repeat instance and update existing values. Live importer behavior and field limits
have not been verified. The record/event must already identify the intended case/scan;
multiple scans routed to the same record/event and model intentionally share a slot.
If earlier prototype exports were imported, reconcile those old instances before
using the permanent mapping.
