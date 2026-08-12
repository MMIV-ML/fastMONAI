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
| `<conda_env>` | Conda env name in your `requirements.yml` | `fastmonai` |
| `<MODEL_TYPE>` | Registered model key | `unet`, `dynunet` |
| `<BUILD_VERSION>` | UTC build identifier | `20260812T140500Z` |

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

## 3. Pull Base Docker Image

Fiona intentionally publishes the ROR/pr2mask base through its `latest` tag. Refresh it before
each release build:

```bash
docker pull haukebartsch/fiona-component-python:latest
docker image inspect haukebartsch/fiona-component-python:latest \
  --format 'base_id={{.Id}} repo_digests={{json .RepoDigests}}'
```

The build command below also uses `--pull`, so a cached base cannot silently replace the
current Fiona image. Record the base image ID or digest with the release.

## 4. Build Container

Build from the project's `deployment/pacs/` directory. Generate the version at build time; do
not edit a date into the Dockerfile manually.

```bash
cd <PROJECT_DIR>/research/<PROJECT>/deployment/pacs

BUILD_VERSION="$(date -u +%Y%m%dT%H%M%SZ)"

docker build --pull \
  --build-arg conda_env="<conda_env>" \
  --build-arg VERSION="$BUILD_VERSION" \
  -f .ror/virt/Dockerfile \
  -t "<CONTAINER_NAME>:$BUILD_VERSION" \
  -t "<CONTAINER_NAME>:latest" \
  .
```

For vestibular schwannoma, substitute `fastmonai`, `vs-seg`, and the project's
`deployment/pacs/` directory in the command above.

The project Dockerfile should require the same version that is used as the Docker tag:

```dockerfile
ARG VERSION
RUN test -n "${VERSION}" || \
    (echo "VERSION build argument is required" >&2; exit 1)
LABEL org.opencontainers.image.version="${VERSION}"
ENV VERSION="${VERSION}"
```

The date must be a Docker tag, not only part of the exported archive filename. An archive
containing only `<CONTAINER_NAME>:latest` loses its dated identity after `docker load`, and
`latest` then depends on archive loading order. The build assigns both tags to the same image:
the dated tag remains a stable identity, while `latest` is a convenient pointer to the most
recent build. Record and qualify the dated tag. Section 7 exports both tags so the delivered
image can be called by either one. The Fiona base image still uses its separately managed
`latest` tag as described in Section 3.

## 5. Qualify the image locally

Before handing it over, run the finished image on the build computer with approved real DICOM
input. Use the dated tag so the test record identifies the exact release:

```bash
cd <PROJECT_DIR>/research/<PROJECT>/deployment/pacs

# Run on all matching series, keep output for inspection
ror trigger -cont "<CONTAINER_NAME>:$BUILD_VERSION" -each -keep
```

### Model Selection

Your container registers one or more model types in its inference script
(`stub_inference.py`; see [Adding your own model](#9-adding-your-own-model)).
Select which one runs with the `model-type` key via the `-envs` option (omit it
to use the container's default):

```bash
ror trigger -cont "<CONTAINER_NAME>:$BUILD_VERSION" -each -keep -envs '{"model-type":"<MODEL_TYPE>"}'
```

For vestibular schwannoma, run this once with `unet` and once with `dynunet`. The dated and
`latest` tags behave identically while they point to the same image; no test tag is needed.

Check that every shipped model loads and runs, the expected mask/probability/report series are
created, and the outputs are readable with correct geometry, valid UIDs, documented model
codes, and plausible values. `-each` processes every selected series and `-keep` retains output
for inspection. See `ror trigger --help` for resource limits and dry-run options.

## 6. Manual Docker run (optional)

For direct testing without ROR:

```bash
docker run --rm \
  -e ROR_CONT_OPTIONS='{"model-type":"<MODEL_TYPE>"}' \
  -v <INPUT_DIR>:/data:ro \
  -v <OUTPUT_DIR>:/output \
  "<CONTAINER_NAME>:$BUILD_VERSION"
```

Omit `ROR_CONT_OPTIONS` to use the default model. For an interactive shell, add
`-it --entrypoint /bin/bash` before the image name.

## 7. Export Container

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

## 8. Research PACS Deployment

Register the qualified image and its DICOM selection criteria with the Research PACS.

Provide the Research PACS administrator with:

- the dated tag, `latest` alias, image ID or registry digest, and archive checksum;
- the Git commit, ROR executable identity, and Fiona base image ID or digest;
- the model-bundle names and hashes;
- the series selection criteria;
- the supported `ROR_CONT_OPTIONS` keys and defaults;
- the expected output series and DICOM identity contract; and
- the qualification results and an approved example case.

Site-specific Fiona configuration, credentials, submission tokens, paths, and trigger rules
belong in controlled infrastructure documentation; obtain them from the Research PACS administrator.

## 9. Adding Your Own Model

Model types are defined in `deployment_models.py`, while Safetensors metadata rebuilds the
allow-listed architecture. Supporting a new architecture also requires registering its
constructor in fastMONAI.
`entrypoint.sh` must parse `ROR_CONT_OPTIONS` as JSON without shell `eval`, reject unknown keys
and invalid values, and forward only a validated `model-type`.

To register a new model type `<name>`:

1. Add `<name>` and its allow-listed Safetensors `arch_id` to `MODEL_ARCH_IDS`.
2. Add a `MODEL_CONFIGS["<name>"]` entry with `models_dir`, `display_name`, and a new,
   never-reused positive `dicom_model_code`.
3. Build a declared Safetensors bundle with `prepare_model_bundle.py`; its generated
   `deployment_config.json` records members, hashes, inference contract, and DICOM UID contract.
4. Rebuild the container (Section 4). The new model is selectable with
   `-envs '{"model-type":"<name>"}'`.

Do not encode readable names in DICOM UIDs. The numeric code registry is documented in the
project PACS README; `SeriesDescription` and `SoftwareVersions` carry the readable identity.

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
│       ├── deployment_models.py
│       ├── prepare_model_bundle.py
│       ├── stub_inference.py
│       └── model_bundles/<model>/  # generated locally and ignored by Git
├── notebooks/
├── workflow/
└── tests/
```

## Troubleshooting

1. **"not a ror directory" error**: Run commands from the project's `deployment/pacs/` directory
2. **No matching series**: Check `ror status --all` to verify data configuration and series filter
3. **Container build fails**: Ensure you're in `deployment/pacs/` with Dockerfile at `.ror/virt/Dockerfile`
