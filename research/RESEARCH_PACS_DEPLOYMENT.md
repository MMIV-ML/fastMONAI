# Research PACS deployment with ROR

This guide explains how to build, qualify, and deliver a fastMONAI research container through
ROR and the Research PACS. Project-specific models and outputs belong in the project's
`deployment/pacs/README.md`.

## Values used below

Commands below use the placeholders in this table. Replace them with your
project's values.

| Placeholder | Meaning | Example (`vestibular_schwannoma`) |
|-------------|---------|--------------------|
| `<PROJECT_DIR>` | Path to your local fastMONAI checkout | `~/ml_projects/fastMONAI` |
| `<PROJECT>` | Research project folder under `research/` | `vestibular_schwannoma` |
| `<CONTAINER_NAME>` | Docker image name you build | `vs-seg` |
| `<MODEL_TYPE>` | Registered model key | `unet`, `dynunet` |
| `<BUILD_VERSION>` | UTC build identifier | `20260812T140500Z` |
| `<DICOM_UID_PREFIX>` | Optional registered DICOM prefix reserved for this generator | site-specific |

The canonical project deployment directory is
`<PROJECT_DIR>/research/<PROJECT>/deployment/pacs/`.

## Prerequisites

- Docker is installed and running.
- The current user can install and run ROR.
- An approved DICOM test series is available locally.

## 1. Install or update the ROR CLI

Install the latest ROR CLI on a new machine and update it before qualifying a new release:

```bash
wget -qO /tmp/ror \
  https://github.com/mmiv-center/Research-Information-System/raw/master/components/Workflow-Image-AI/build/linux-amd64/ror
sudo install -m 0755 /tmp/ror /usr/local/bin/ror
```

Verify the installed executable before continuing:

```bash
which ror
ror --help
sha256sum "$(command -v ror)"
```

## 2. Initialize ROR Project

Navigate to the project's `deployment/pacs/` directory and initialize the local ROR working
directory if it has not already been initialized:

```bash
cd <PROJECT_DIR>/research/<PROJECT>/deployment/pacs

ror init .
ror config -data <PATH_TO_DICOM_DATA> --working_directory "$(pwd)"
ror status --all

ror config --suggest
ror config --select "<suggested select statement>"
```

The select statement must match the approved test series.

## 3. Prepare the model bundle

Build the project's declared Safetensors bundle before building the image. With no prefix, the
bundle uses deterministic `2.25` UUID-derived UIDs and requires no registration. If the deploying
organization owns and has reserved a registered DICOM prefix for this generator, pass it without
a trailing period:

```bash
python prepare_model_bundle.py ...

# Optional site-controlled identity
export DICOM_UID_PREFIX="<DICOM_UID_PREFIX>"
python prepare_model_bundle.py ... --dicom-uid-prefix "$DICOM_UID_PREFIX"
```

The declared members determine the deployment form: exactly one member is a
single-model deployment, while two or more members form an ensemble. The
builder stores the optional prefix in `deployment_config.json`; it cannot be
overridden at runtime. The prefix is public metadata once used, so keep
site-specific values out of Git and include it in controlled release records.

Bundle preparation is the static model-validation boundary: it validates lineage,
roles, architecture, ensemble compatibility, inference configuration, strict model
loading, and hashes before publishing the bundle. The immutable image trusts that
prepared bundle. Runtime strictly loads the declared Safetensors files and reads the
embedded `PatchConfig`, but does not repeat file hashing or release-lineage checks for
every patient.

For a registered prefix, ask the institution's DICOM/OID administrator first. If the institution
does not own one, use a recognized allocation authority such as the
[Medical Connections free UID service](https://www.medicalconnections.co.uk/FreeUID/). Never
invent a root or use another organization's root. The root owner is responsible for preventing
duplicate subordinate UIDs and should reserve a subtree for each generator.
Omitting the prefix selects the public `2.25` form.

## 4. Pull Base Docker Image

Fiona intentionally publishes the ROR/pr2mask base through its `latest` tag. Refresh it before
each release build:

```bash
docker pull haukebartsch/fiona-component-python:latest
docker image inspect haukebartsch/fiona-component-python:latest \
  --format 'base_id={{.Id}} repo_digests={{json .RepoDigests}}'
```

The build command below also uses `--pull`, so a cached base cannot silently replace the
current Fiona image. Record the base image ID or digest with the release.

## 5. Build Container

Build from the project's `deployment/pacs/` directory. Generate the version at build time; do
not edit a date into the Dockerfile manually.

```bash
cd <PROJECT_DIR>/research/<PROJECT>/deployment/pacs

BUILD_VERSION="$(date -u +%Y%m%dT%H%M%SZ)"

docker build --pull \
  --build-arg VERSION="$BUILD_VERSION" \
  -f .ror/virt/Dockerfile \
  -t "<CONTAINER_NAME>:$BUILD_VERSION" \
  -t "<CONTAINER_NAME>:latest" \
  .
```

For vestibular schwannoma, substitute `vs-seg` and use the project's
`deployment/pacs/` directory.

The date must be a Docker tag, not only part of the exported archive filename. An archive
containing only `<CONTAINER_NAME>:latest` loses its dated identity after `docker load`, and
`latest` then depends on archive loading order. The build assigns both tags to the same image:
the dated tag remains a stable identity, while `latest` is a convenient pointer to the most
recent build. Record and qualify the dated tag. Section 8 exports both tags so the delivered
image can be called by either one. The Fiona base image still uses its separately managed
`latest` tag as described in Section 4.

## 6. Qualify the image locally

Before handing it over, run the finished image on the build computer with approved real DICOM
input. Use the dated tag so the test record identifies the exact release:

```bash
cd <PROJECT_DIR>/research/<PROJECT>/deployment/pacs

# Run on all matching series, keep output for inspection
ror trigger -cont "<CONTAINER_NAME>:$BUILD_VERSION" -each -keep
```

### Model Selection

Your container registers one or more model types in `deployment_models.py`
(see [Adding your own model](#10-adding-your-own-model)).
Select which one runs with the `model-type` key via the `-envs` option (omit it
to use the container's default):

```bash
ror trigger -cont "<CONTAINER_NAME>:$BUILD_VERSION" -each -keep -envs '{"model-type":"<MODEL_TYPE>"}'
```

For vestibular schwannoma, run this once with `unet` and once with `dynunet`. The dated and
`latest` tags behave identically while they point to the same image; no test tag is needed.

Check that every shipped model loads and runs, all expected derived DICOM products are created,
and the outputs are readable with correct geometry, valid UIDs, and plausible values. `-each`
processes every selected series and `-keep` retains output for
inspection. See `ror trigger --help` for resource limits and dry-run options.

## 7. Manual Docker run (optional)

For direct testing without ROR:

```bash
docker run --rm \
  -e ROR_CONT_OPTIONS='{"model-type":"<MODEL_TYPE>"}' \
  -v <DICOM_SERIES_DIR>:/data/input:ro \
  -v <OUTPUT_DIR>:/output \
  "<CONTAINER_NAME>:$BUILD_VERSION"
```

Omit `ROR_CONT_OPTIONS` to use the default model. For an interactive shell, add
`-it --entrypoint /bin/bash` before the image name.

### Runtime input, preflight, and output

ROR normally supplies the selected source series at `/data/input` and a separate writable
`/output`. They remain separate container paths even though the derived objects preserve the
source Study Instance UID and return to the same PACS study. A direct run may bind the same
physical directory to both paths when required; keep the input mount read-only.

Before loading a model, the application reads DICOM headers only. Missing or inconsistent
Study, Series, SOP, modality, or core geometry values stop the run. Frame of Reference and
file-meta SOP identity problems produce a warning. Legacy source UIDs containing nonstandard
hexadecimal characters produce a warning but are accepted as opaque source identity. The
derived objects preserve the source Study Instance UID and preserve the Frame of Reference UID
when it is consistently available, while every fastMONAI-generated Series Instance UID and SOP
Instance UID is valid numeric DICOM syntax.

The output root may already contain source DICOM, ROR bookkeeping, or unrelated files. Final
output directories owned by the application must not exist when a run starts, so an earlier
result is never overwritten. Intermediate products remain in a separate container work
directory. After postprocessing succeeds and the required output directories are present, the
final directories are copied to `/output`, and `pacs_command.log` is replaced atomically. Each
project README must document its exact output names. Where input and output share a physical
directory, the same preflight and no-overwrite rules apply.

## 8. Export Container

Record the image identity. Reassign `latest` to the qualified dated image immediately before
export, then save both references so the recipient can call either tag:

```bash
docker image inspect "<CONTAINER_NAME>:$BUILD_VERSION" --format '{{.Id}}'
docker tag "<CONTAINER_NAME>:$BUILD_VERSION" "<CONTAINER_NAME>:latest"

ARCHIVE="<CONTAINER_NAME>-$BUILD_VERSION.tar.gz"
docker save "<CONTAINER_NAME>:$BUILD_VERSION" "<CONTAINER_NAME>:latest" | gzip > "$ARCHIVE"

sha256sum "$ARCHIVE" > "$ARCHIVE.sha256"
sha256sum -c "$ARCHIVE.sha256"
```

The SHA-256 detects archive corruption during transfer; it does not test the image. After
`docker load`, either tag can be called. The dated tag is the stable release and rollback
identity; `latest` follows whichever archive was loaded or tagged last.

## 9. Research PACS Deployment

Register the qualified image and its DICOM selection criteria with the Research PACS.

Provide the Research PACS administrator with:

- the dated tag, `latest` alias, image ID or registry digest, and archive checksum;
- the Git commit, ROR executable identity, and Fiona base image ID or digest;
- the model-bundle names and hashes;
- the bundle's DICOM schema and optional registered prefix;
- the series selection criteria;
- the supported `ROR_CONT_OPTIONS` keys and defaults;
- the expected output series and DICOM identity contract; and
- the qualification results and an approved example case.

Site-specific Fiona configuration, credentials, submission tokens, paths, and trigger rules
belong in controlled infrastructure documentation; obtain them from the Research PACS administrator.

## 10. Adding Your Own Model

Model types are defined in `deployment_models.py`, while Safetensors metadata
rebuilds the allow-listed architecture. Supporting a new architecture also
requires registering its constructor in fastMONAI.

To register a new model type `<name>`:

1. Add a `MODEL_CONFIGS["<name>"]` entry with its allowed `arch_ids`,
   `display_name`, and a new, never-reused positive `dicom_model_code`.
2. Build a declared Safetensors bundle with `prepare_model_bundle.py`.
3. Rebuild the container (Section 5). The new model is selectable with
   `-envs '{"model-type":"<name>"}'`.

A new architecture serving the same task can retain the pipeline application identity. A new
modality, task, or input combination needs a distinct stable application identity. Multi-input
pipelines must bind every source Series UID and ordered SOP-sequence digest in a fixed role order;
the single-input project implementation must be extended before such a model is deployed.

Do not encode readable names in DICOM UIDs. The numeric code registry is documented in the
project PACS README. `SeriesDescription` and `DerivationDescription` carry readable model
identity, while `SoftwareVersions` records the producing fastMONAI version.

## Project Structure

```
research/<PROJECT>/
├── README.md
├── deployment/
│   └── pacs/
│       ├── .ror/
│       │   └── virt/
│       │       └── Dockerfile
│       ├── README.md
│       ├── entrypoint.sh
│       ├── requirements.yml
│       ├── deployment_hashing.py
│       ├── deployment_bundle.py
│       ├── deployment_models.py
│       ├── dicom_output.py
│       ├── prepare_model_bundle.py
│       ├── pacs_inference.py
│       └── model_bundles/<model>/  # generated locally and ignored by Git
├── notebooks/
├── workflow/
└── tests/
```

## Troubleshooting

1. **"not a ror directory" error**: Run commands from the project's `deployment/pacs/` directory
2. **No matching series**: Check `ror status --all` to verify data configuration and series filter
3. **Container build fails**: Ensure you're in `deployment/pacs/` with Dockerfile at `.ror/virt/Dockerfile`
