# Vestibular schwannoma PACS container

ROR/PACS container that runs the five-fold DynUNet ensemble on a CE-T1w series. CPU only:
each of the five models runs in its own worker with three threads, so it needs at least 15
CPU threads. TTA is on by default and averages the original and one copy flipped
anterior-posterior and inferior-superior (`TTA_FLIP_AXES` in `parallel_inference.py`). For the
general workflow see
[Research PACS deployment with ROR](../../../RESEARCH_PACS_DEPLOYMENT.md).

## Build a release

1. Prepare the model bundle in the environment pinned in `requirements.yml` (the strict
   metadata check depends on the PyTorch version), with full MLflow run IDs:

   ```bash
   python prepare_model_bundle.py --model-type dynunet --artifact-role best \
     --run "fold_1=$RUN_1" --run "fold_2=$RUN_2" --run "fold_3=$RUN_3" \
     --run "fold_4=$RUN_4" --run "fold_5=$RUN_5"
   ```

   The bundle goes to `model_bundles/dynunet/` (not in Git). `--artifact MEMBER=/path.safetensors`
   adds a local file instead of a run. A new bundle must be registered in
   `redcap_model_instances.json` (see below) or the container refuses to run.

2. Test (from `research/`) and build:

   ```bash
   (cd ../../.. && python -m unittest discover -s vestibular_schwannoma/tests/deployment -t .)
   BUILD_VERSION="$(date -u +%Y%m%dT%H%M%SZ)"
   docker build --pull --build-arg VERSION="$BUILD_VERSION" -f .ror/virt/Dockerfile \
     -t "fastmonai-vsseg:$BUILD_VERSION" -t fastmonai-vsseg:latest .
   ```

   Qualify on test data with `ror trigger -cont "fastmonai-vsseg:$BUILD_VERSION" -each -keep`.

3. Package both tags in `release_packages/$BUILD_VERSION/` with a short README and
   `ror_select_entry.txt`. Don't add a `.sha256`; the image ID after `docker load` is the check.

   ```bash
   docker save "fastmonai-vsseg:$BUILD_VERSION" fastmonai-vsseg:latest \
     | gzip > "release_packages/$BUILD_VERSION/fastmonai-vsseg-$BUILD_VERSION.tar.gz"
   ```

The PACS entry uses `fastmonai-vsseg:latest`, so loading a new archive is enough:

```json
"ENTRY_NAME": {
  "select": "<VS CE-T1 series selector>",
  "ROR_CONT_OPTIONS": "{}",
  "docker_image": "fastmonai-vsseg:latest"
},
```

## Options

`ROR_CONT_OPTIONS` accepts `model-type` (`dynunet`, the default and only model) and `tta`
(JSON Boolean, default `true`). `"{}"` runs the defaults and `"{\"tta\":false}"` turns TTA
off. Unknown keys and invalid values fail before inference, as do inconsistent study, series,
SOP, modality or geometry headers.

## Outputs

The container writes two intermediate series, `model_mask` and `vote_map`, which pr2mask
turns into the published outputs:

- `fused`: the mask drawn on the MR.
- `fused_vote_map`: the vote map on the MR. Blue is exactly the mask; yellow marks voxels
  outside it with some foreground probability.
- `reports`: pr2mask's report, with measurements per region.
- `labels`: the mask series, in pr2mask's study, window and UIDs, plus our provenance
  (SeriesDescription, SoftwareVersions, DerivationDescription with the bundle hash and TTA, and a
  `MASK` ImageType value).
- `redcap`: pr2mask's REDCap rows, see below.

These folders must not exist in the output folder beforehand.

## REDCap

`redcap/<mask-series-UID>/output.json` holds pr2mask's EAV rows for the `pr2mask` instrument.
There is no direct uploader; the PACS imports the file.

- **One instance per region.** pr2mask numbers the mask's connected regions 1, 2, 3, ...
  (26-connected, scan order). Region k goes to repeat instance base + k, where the base comes
  from `redcap_model_instances.json`. DynUNet's base is 200, so its regions fill 201, 202, ...
  Only instances with a region are written.
- **Empty mask:** instance 201 with blank columns, so REDCap shows the model ran. It is routed
  by the source PatientID and ReferringPhysicianName (`EventName:` removed), like pr2mask.
- **`model_version`** is added to every instance, for example
  `dynunet 5-fold, fastmonai-vsseg:20261110T082650Z`. The field is on the REDCap form, not in
  pr2mask; the exporter also adds it to `output_data_dictionary.zip`.
- **Mask text.** pr2mask's `mask_base64_gzip_compressed` covers the whole scan at 16 bits per
  voxel: about 84,000 characters at 512x512x120 and 105,000 at 512x448x176. REDCap may cut
  values over 65,535 characters; this hasn't been checked in the receiving project. The value
  is sent as pr2mask wrote it, and the full mask is always in `labels`.
- **Empty pr2mask dictionary.** For an empty mask pr2mask lists no columns, so the exporter
  takes them from `pr2mask_dictionary.csv`. Regenerate that file from a non-empty run when
  the Fiona base changes pr2mask's columns.

### Instance registry

`redcap_model_instances.json` gives each bundle hash a permanent block of 100 instances,
stored as the base (a multiple of 100). DynUNet `c5642793...` has 200. Instance 1 is the manual
mask. Older DynUNet results at 11 (and 2/3 before that) are no longer updated.

- A rerun of the same bundle overwrites its instances. If it finds fewer regions, the higher
  instances of the earlier run stay.
- New weights mean a new bundle hash. Give it an unused block, or replace the old entry to
  overwrite the old model's results.
- Never renumber blocks after REDCap imports. Unknown bundles, duplicate or invalid bases and
  region numbers of 100 or more fail before any output is published.
