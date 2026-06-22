# Docker ROR Workflow for fastMONAI Inference

This guide explains how to build and run a Docker container for medical-image
inference through ROR (Research Workflow Processing) on Linux and MacOS. It is
written to work for **any** fastMONAI inference container, model, and research
project; the vestibular-schwannoma project (`vs_seg`) is used throughout as a
concrete, copy-pasteable example.

## Conventions

Commands below use the placeholders in this table. Replace them with your
project's values; the `vs_seg` example values are shown for reference.

| Placeholder | Meaning | Example (`vs_seg`) |
|-------------|---------|--------------------|
| `<PROJECT_DIR>` | Path to your local fastMONAI checkout | `~/ml_projects/fastMONAI` |
| `<PROJECT>` | Research project folder under `research/` | `vs_seg` |
| `<CONTAINER_NAME>` | Docker image name you build | `vs-seg` |
| `<conda_env>` | Conda env name in your `requirements.yml` | `fastmonai` |
| `<MODEL_TYPE>` | A model your container registers (see [Adding your own model](#9-adding-your-own-model)) | `unet`, `dynunet` |

> The integration folder for the example lives at
> `<PROJECT_DIR>/research/vs_seg/fastmonai_inference/integration`.

## Prerequisites

- Docker installed and running (current user has access to docker)
- current user has access to install ror
- provided DICOM test data in a directory

## 1. Install ROR CLI

Download and install the latest ror CLI tool:

```bash
wget -qO- https://github.com/mmiv-center/Research-Information-System/raw/master/components/Workflow-Image-AI/build/linux-amd64/ror > /tmp/ror
sudo mv /tmp/ror /usr/local/bin/ror
sudo chmod +x /usr/local/bin/ror
```

Verify installation:

```bash
# print ror location if found
which ror
# print help for ror
ror --help
```

## 2. Initialize ROR Project

Navigate to your project's `integration/` folder and initialize a ror project
(if not already done):

```bash
cd <PROJECT_DIR>/research/<PROJECT>/fastmonai_inference/integration

# Initialize (first time only)
ror init .

# Configure data source (you should see data imported)
ror config -data <PATH_TO_DICOM_DATA> --working_directory $(pwd)

# Check status (lists all loaded data)
ror status --all

# Get a suggestion for a select statement
ror config --suggest

# set the select statement (needs to select your example dataset)
ror config --select "<suggested select statement>"
```

> Example (`vs_seg`): `cd <PROJECT_DIR>/research/vs_seg/fastmonai_inference/integration`

## 3. Pull Base Docker Image

Update the base image (includes pr2mask tools):

```bash
docker pull haukebartsch/fiona-component-python:latest
```

## 4. Build Container

Build from the `integration/` folder (where entrypoint.sh and requirements.yml are located):

```bash
cd <PROJECT_DIR>/research/<PROJECT>/fastmonai_inference/integration

# conda_env must match the environment name in your requirements.yml
docker build \
  --build-arg conda_env="<conda_env>" \
  -f .ror/virt/Dockerfile \
  -t <CONTAINER_NAME>:latest .
```

Example (`vs_seg`):
```bash
docker build --build-arg conda_env="fastmonai" -f .ror/virt/Dockerfile -t vs-seg:latest .
```

## 5. Run Inference Using ror trigger

The recommended way to run inference is using `ror trigger` with a container:

```bash
cd <PROJECT_DIR>/research/<PROJECT>/fastmonai_inference/integration

# Run on all matching series, keep output for inspection
ror trigger -cont <CONTAINER_NAME>:latest -each -keep
```

### Model Selection

Your container registers one or more model types in its inference script
(`stub_inference.py`; see [Adding your own model](#9-adding-your-own-model)).
Select which one runs with the `model-type` key via the `-envs` option (omit it
to use the container's default):

```bash
ror trigger -cont <CONTAINER_NAME>:latest -each -keep -envs '{"model-type":"<MODEL_TYPE>"}'
```

> Example (`vs_seg`): the container ships `unet` (default) and `dynunet`:
> ```bash
> # Use UNet (default)
> ror trigger -cont vs-seg:latest -each -keep -envs '{"model-type":"unet"}'
>
> # Use DynUNet
> ror trigger -cont vs-seg:latest -each -keep -envs '{"model-type":"dynunet"}'
> ```

### ror trigger Options

| Option | Description |
|--------|-------------|
| `-cont <name>` | Container name to use for processing |
| `-each` | Process all matching series (not just one random) |
| `-keep` | Keep the output directory for inspection |
| `-envs '{"key":"value"}'` | Pass environment variables (e.g., model-type) |
| `-test` | Dry run - show what would happen without executing |
| `-cpus <n>` | Limit available CPUs (default: 2) |
| `-mem <size>` | Limit memory (e.g., "4g") |

## 6. Manual Docker Run (Alternative)

For manual testing without ror, mount input/output directories directly:

```bash
docker run --rm \
  -v <INPUT_DIR>:/data:ro \
  -v <OUTPUT_DIR>:/output \
  <CONTAINER_NAME>:latest
```

Example:
```bash
docker run --rm \
  -v /path/to/dicom/input:/data:ro \
  -v /path/to/output:/output \
  vs-seg:latest
```

With model selection (omit `ROR_CONT_OPTIONS` for the container's default model,
or pick another registered model with):
```bash
docker run --rm \
  -e ROR_CONT_OPTIONS='{"model-type":"<MODEL_TYPE>"}' \
  -v /path/to/dicom/input:/data:ro \
  -v /path/to/output:/output \
  <CONTAINER_NAME>:latest
```

> Example (`vs_seg`): `-e ROR_CONT_OPTIONS='{"model-type":"dynunet"}' ... vs-seg:latest`

To debug from inside the container use:

```bash
docker run --rm -it \
  -v <INPUT_DIR>:/data:ro \
  -v <OUTPUT_DIR>:/output \
  --entrypoint /bin/bash \
  <CONTAINER_NAME>:latest
# ./entrypoint.sh <conda_env>
# exit
```

## 7. Export Container

Save the container for distribution:

```bash
docker save <CONTAINER_NAME>:latest | gzip > <FILENAME>.tar.gz
```

Example (`vs_seg`):
```bash
docker save vs-seg:latest | gzip > vs-seg.tar.gz
```

## 8. Research PACS Deployment

To deploy the container in the research PACS (fiona), register the workflow with
a select statement and the container image. The model is chosen via the
`model-type` key in `ROR_CONT_OPTIONS` (the container's default, or any
registered alternative). Per study the container ships the predicted mask and
target-probability series (`/output/mask`, `/output/vote_map`), plus pr2mask's
`labels/`, `fused/`, `fused_vote_map/` (probability overlay), `reports/`, and
`redcap/`.

The workflow name, `AETitleCalled`, select statement, description, model choice,
and output series UIDs in the entries below are specific to the `vs_seg`
example -- change them for your project.

Select-statement entry (example, `vs_seg`):

```json
"MMIVVestSchAI": {
    "select": "SELECT study FROM study WHERE series named \"everything\" has Modality regexp \"MR\"",
    "ROR_CONT_OPTIONS": "{\"model-type\":\"unet\"}",
    "docker_image": "vs-seg:latest"
}
```

`config.json` trigger entry (example, `vs_seg`):

```json
{
    "log": "/home/processing/logs/Workflows_trigger_VestSch_AI.log",
    "name": "MMIVVestSchAI",
    "description": "Vestibular Schwanoma Cancer Segmentation",
    "trigger": { "AETitleCalled": "^MMIVVESTSCH$" },
    "trigger-study": [
        { "type": "exec", "cmd": [ "echo", "triggered this service", "@StudyInstanceUID@", "@SeriesInstanceUID@", "@PATH@", "@WorkflowsFolder@", "@DESCRIPTION@", "@StreamName@" ] },
        { "type": "exec", "cmd": [ "mkdir", "-p", "/tmp/site/proc/@StudyInstanceUID@" ] },
        { "type": "exec", "cmd": [ "@WorkflowsFolder@/php/ror", "init", "-type", "python", "/tmp/site/proc/@StudyInstanceUID@" ] },
        { "type": "exec", "cmd": [ "@WorkflowsFolder@/php/ror", "config", "--working_directory", "/tmp/site/proc/@StudyInstanceUID@", "--temp_directory", "/tmp/site/proc/@StudyInstanceUID@", "--data", "/tmp/site/archive/@StudyInstanceUID@" ] },
        { "type": "exec", "cmd": [ "@WorkflowsFolder@/php/pullSelect.sh", "/tmp/site/archive/@StudyInstanceUID@", "/tmp/site/proc/@StudyInstanceUID@/select.statement", "@StreamName@" ] },
        { "type": "exec", "cmd": [ "@WorkflowsFolder@/php/ror", "config", "--working_directory", "/tmp/site/proc/@StudyInstanceUID@", "--call", "unused", "-select", "/tmp/site/proc/@StudyInstanceUID@/select.statement" ] },
        { "type": "exec", "cmd": [ "@WorkflowsFolder@/php/addJob.sh", "/tmp/site/archive/@StudyInstanceUID@", "/tmp/site/proc/@StudyInstanceUID@", "@destination@", "@StreamName@" ] }
    ],
    "trigger-series": []
}
```

A submission token for the research project is required to upload the workflow;
obtain it from https://fiona.ihelse.net/applications/User/index.php

## 9. Adding Your Own Model

Model types are defined in the container's inference script
(`stub_inference.py`), so supporting a new architecture (e.g. SegResNet,
SegMamba, MedNeXt, or a custom MONAI net) is a small change in your project.
`entrypoint.sh` already forwards any `model-type` value from `ROR_CONT_OPTIONS`
to the script -- you do not need to change it.

To register a new model type `<name>`:

1. Add a `build_<name>()` function in `stub_inference.py` that rebuilds your
   network architecture, matching how it was trained.
2. Add a `MODEL_CONFIGS["<name>"]` entry with its `models_dir`, `weights_file`,
   `build_model` (the builder from step 1), `weight_prefix` (the saved
   state-dict key prefix to strip, e.g. `_orig_mod.` for torch.compile),
   `seg_uid`/`prob_uid` (unique DICOM UID suffixes), and `display_name`.
3. Add `"<name>"` to the `--model-type` argparse `choices` list.
4. Create `models_dir/` and drop in the model's `weights_file` plus its
   `inference_settings.pkl` (patch_size, apply_reorder, target_spacing --
   mirroring the preprocessing used during training).
5. Rebuild the container (Section 4). The new model is now selectable with
   `-envs '{"model-type":"<name>"}'`.

> Example (`vs_seg`): the `unet` and `dynunet` entries in `MODEL_CONFIGS` follow
> exactly this pattern, with weights under `unet_models/` and `dynunet_models/`.

## Project Structure

```
research/<PROJECT>/fastmonai_inference/integration/
├── .ror/
│   └── virt/
│       └── Dockerfile          # Docker build file
├── entrypoint.sh               # Container entrypoint (forwards model-type)
├── requirements.yml            # Conda environment definition
├── stub_inference.py           # Inference script (defines MODEL_CONFIGS)
└── <model>_models/             # Per-model weights + inference_settings.pkl
                                #   (vs_seg example: unet_models/, dynunet_models/)
```

## Notes

### Model Types

Available model types are defined by your container's `MODEL_CONFIGS` in
`stub_inference.py` (see [Adding your own model](#9-adding-your-own-model)). Each
model type loads a single "final clean" model (trained on all available data
with cleaned masks) and runs patch-based sliding-window inference (no ensemble).
The selected type is passed via the `model-type` key in `ROR_CONT_OPTIONS` and
forwarded by `entrypoint.sh`.

> Example (`vs_seg`):
> - **unet**: UNet model (default)
> - **dynunet**: DynUNet model

### Output Structure
When using pr2mask features (auto_report_mode), outputs include:
- `/output/mask/` - Predicted segmentation mask, raw DICOM series (uint16, 0/1)
- `/output/vote_map/` - Target-class (foreground) probability, raw DICOM series (uint16; [0,1] scaled to [0,65535])
- `/output/labels/` - Predicted segmentation mask as DICOM labels (from pr2mask)
- `/output/fused/` - Predicted mask overlaid on the input image
- `/output/fused_vote_map/` - Probability overlay (blue = probability >= 0.5, yellow = probability < 0.5)
- `/output/reports/` - Generated reports
- `/output/redcap/` - REDCap export files

### Troubleshooting

1. **"not a ror directory" error**: Run commands from the `integration/` folder
2. **No matching series**: Check `ror status --all` to verify data configuration and series filter
3. **Container build fails**: Ensure you're in the `integration/` folder with Dockerfile at `.ror/virt/Dockerfile`
