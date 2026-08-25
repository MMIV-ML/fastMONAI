#!/bin/bash

set -euo pipefail

readonly INPUT_DIR="/data/input"
readonly OUTPUT_DIR="/output"
readonly CONDA_ENV="fastmonai"

if [ ! -d "$INPUT_DIR" ]; then
    echo "Error: expected an input DICOM directory at ${INPUT_DIR}" >&2
    exit 2
fi

# ROR exposes the JSON passed with -envs as ROR_CONT_OPTIONS:
#   ror trigger ... -envs '{"model-type":"dynunet","tta":false}'
ror_options="${ROR_CONT_OPTIONS:-}"
if [ -z "$ror_options" ]; then
    ror_options='{}'
fi
if ! jq -e '
    type == "object"
    and ((keys - ["model-type", "tta"]) | length == 0)
    and (if has("model-type") then (.["model-type"] | type == "string") else true end)
    and (if has("tta") then (.tta | type == "boolean") else true end)
  ' >/dev/null <<< "$ror_options"; then
    echo "Error: invalid ROR_CONT_OPTIONS." >&2
    echo "Supported options: model-type (string), tta (Boolean)." >&2
    exit 2
fi

model_type=$(jq -r '.["model-type"] // "unet"' <<< "$ror_options")
if [[ ! "$model_type" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
    echo "Error: invalid model-type syntax: ${model_type}" >&2
    exit 2
fi
tta=$(jq -r 'if has("tta") then .tta else true end' <<< "$ror_options")

script_dir="$(dirname "$0")"
inference_script="${script_dir}/pacs_inference.py"
if [ ! -f "$inference_script" ]; then
    echo "Error: bundled inference script is missing: ${inference_script}" >&2
    exit 1
fi

cmd=(
    conda run --no-capture-output -n "$CONDA_ENV"
    python "$inference_script" "$INPUT_DIR" "$OUTPUT_DIR"
    --model-type "$model_type"
)
if [ "$tta" = "true" ]; then
    cmd+=(--tta)
else
    cmd+=(--no-tta)
fi

echo "Using ${model_type} model (TTA: $([ "$tta" = "true" ] && echo on || echo off))"
printf 'run now:'
printf ' %q' "${cmd[@]}"
printf '\n'
exec "${cmd[@]}"
