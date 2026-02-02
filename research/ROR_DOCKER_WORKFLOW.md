# Docker ROR Workflow for fastMONAI Inference

This guide is for Linux and MacOS and explains how to build and run a Docker container for inference using ROR (Research Workflow Processing).

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

Navigate to the `integration/` folder and initialize a ror project (if not already done):

```bash
cd <PROJECT_DIR>/research/vs_seg/fastmonai_inference/integration

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

## 3. Pull Base Docker Image

Update the base image (includes pr2mask tools):

```bash
docker pull haukebartsch/fiona-component-python:latest
```

## 4. Build Container

Build from the `integration/` folder (where entrypoint.sh and requirements.yml are located):

```bash
cd <PROJECT_DIR>/research/vs_seg/fastmonai_inference/integration

# assume that fastmonai is your conda environments name (see your requirements.yml)
docker build \
  --build-arg conda_env="fastmonai" \
  -f .ror/virt/Dockerfile \
  -t <CONTAINER_NAME>:latest .
```

Example:
```bash
docker build --build-arg conda_env="fastmonai" -f .ror/virt/Dockerfile -t vs-seg-fastmonai:latest .
```

## 5. Run Inference Using ror trigger

The recommended way to run inference is using `ror trigger` with a container:

```bash
cd <PROJECT_DIR>/research/vs_seg/fastmonai_inference/integration

# Run on all matching series, keep output for inspection
ror trigger -cont <CONTAINER_NAME>:latest -each -keep
```

### Model Selection

Select model type (unet or mednext) using the `-envs` option:

```bash
# Use UNet ensemble (default)
ror trigger -cont <CONTAINER_NAME>:latest -each -keep -envs '{"model-type":"unet"}'

# Use MedNeXt ensemble
ror trigger -cont <CONTAINER_NAME>:latest -each -keep -envs '{"model-type":"mednext"}'
```

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
  vs-seg-fastmonai:latest
```

With model selection (UNet is default, use MedNeXt with):
```bash
docker run --rm \
  -e ROR_CONT_OPTIONS='{"model-type":"mednext"}' \
  -v /path/to/dicom/input:/data:ro \
  -v /path/to/output:/output \
  vs-seg-fastmonai:latest
```

To debug from inside the container use:

```bash
docker run --rm -it \
  -v <INPUT_DIR>:/data:ro \
  -v <OUTPUT_DIR>:/output \
  --entrypoint /bin/bash
  <CONTAINER_NAME>:latest
# ./entrypoint.py
# exit
```


## 7. Export Container

Save the container for distribution:

```bash
docker save <CONTAINER_NAME>:latest | gzip > <FILENAME>.tar.gz
```

Example:
```bash
docker save vs-seg-fastmonai:latest | gzip > vs-seg-fastmonai.tar.gz
```

## Project Structure

```
research/vs_seg/fastmonai_inference/integration/
├── .ror/
│   └── virt/
│       └── Dockerfile      # Docker build file
├── entrypoint.sh           # Container entrypoint script
├── requirements.yml        # Conda environment definition
└── stub_inference.py       # Inference script
```

## Notes

### Model Types
- **unet**: UNet ensemble model (default)
- **mednext**: MedNeXt ensemble model

The model type is passed via `ROR_CONT_OPTIONS` environment variable and handled by `entrypoint.sh`.

### Output Structure
When using pr2mask features (auto_report_mode), outputs include:
- `/output/fused/` - Fused images
- `/output/labels/` - Segmentation labels
- `/output/reports/` - Generated reports
- `/output/redcap/` - REDCap export files

### Troubleshooting

1. **"not a ror directory" error**: Run commands from the `integration/` folder
2. **No matching series**: Check `ror status --all` to verify data configuration and series filter
3. **Container build fails**: Ensure you're in the `integration/` folder with Dockerfile at `.ror/virt/Dockerfile`
