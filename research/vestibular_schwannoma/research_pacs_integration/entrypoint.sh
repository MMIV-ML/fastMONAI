#!/bin/bash --login
# The --login ensures the bash configuration is loaded.

output="/output"

if [ ! -z "$CONDA_DEFAULT_ENV" ]; then
  conda_env="$CONDA_DEFAULT_ENV"
fi

if [ -z "$conda_env" ]; then
    echo "Usage: <conda-env>"
    exit -1
fi


MODELTOUSE="unet"
USE_TTA=1
# Handle environment variables provided to ror as --envs "ROR_CONT_OPTIONS={\"model-type\":\"unet\",\"tta\":false}"
if [ ! -z "$ROR_CONT_OPTIONS" ]; then
    for key in $(echo "$ROR_CONT_OPTIONS" | jq -r "keys[]"); do
        value=$(echo "$ROR_CONT_OPTIONS" | jq -r '."'$key'"')
        if [ $key == "model-type" ]; then
          MODELTOUSE=$value
        fi
        # 8-flip test-time augmentation is on by default; tta false/0/no/off
        # disables it (true/1/yes/on re-enables).
        if [ $key == "tta" ]; then
          case "$(echo "$value" | tr '[:upper:]' '[:lower:]')" in
            true|1|yes|on) USE_TTA=1 ;;
            false|0|no|off) USE_TTA=0 ;;
          esac
        fi
    done
fi


# This container ships a single unet 5-fold ensemble (model-type unet). TTA is on
# by default; set the tta key in ROR_CONT_OPTIONS to false to disable it.
SCRIPT_DIR="$(dirname "$0")"
STUB_SCRIPT="${SCRIPT_DIR}/stub_inference.py"
MODEL_ARG="--model-type ${MODELTOUSE}"
TTA_ARG=""
if [ "$USE_TTA" -eq 1 ]; then
  TTA_ARG="--tta"
  echo "Using ${MODELTOUSE} model (TTA: on)"
else
  echo "Using ${MODELTOUSE} model (TTA: off)"
fi

# where is pr2mask?
export PATH="/pr2mask:$PATH"

# Fold count for the DICOM info field. Count the fold_*.pkl actually shipped in
# vs5f_unet_models/ (the same files the stub globs and loads), so this label and
# the stub's SoftwareVersions tag (len(models)) always agree. nullglob makes a
# missing directory / no matches yield 0 rather than a literal glob pattern.
shopt -s nullglob
fold_pkls=("${SCRIPT_DIR}"/vs5f_unet_models/fold_*.pkl)
shopt -u nullglob
n_folds=${#fold_pkls[@]}
INFO="${MODELTOUSE} ${n_folds}-fold ensemble, Predicted $(date '+%b%d%Y')"

# if we find imageAndMask2Report and json2SR in this container
auto_report_mode=0
output2="/output_tmp"
if [ -f /pr2mask/imageAndMask2Report ]; then
    auto_report_mode=1
else
    output2="${output}"
fi

# relax strict mode for conda activate (its hooks aren't 'set -euo pipefail' safe), then restore
set +euo pipefail
conda activate "${conda_env}"
if [ $? -ne 0 ]; then
   echo "Error: activating conda environment \"$1\" failed."
   exit -1
fi
set -euo pipefail

log_file="${output}"/stub_command.log
# prefer the bundled inference script; otherwise run the command passed to the container
if [ -n "$STUB_SCRIPT" ] && [ -f "$STUB_SCRIPT" ]; then
    cmd="python ${STUB_SCRIPT} /data ${output2} ${MODEL_ARG} ${TTA_ARG}"
else
    cmd="$@ ${output2}"
fi
echo "run now: $cmd"
# eval splits the assembled $cmd into command + arguments
eval $cmd

if [ "$auto_report_mode" -eq 1 ]; then
    echo "imageAndMask2Report:"
    /pr2mask/imageAndMask2Report /data/input "${output2}/mask" "${output2}" -u "$VERSION" -i "$VERSION" --reporttype mosaic -t "${INFO} " >> "${log_file}" 2>&1
    echo "imageAndMask2Fused:"
    /pr2mask/imageAndMask2Fused /data/input "${output2}/mask" "${output2}" -u "${VERSION}_fused" -i "$VERSION" >> "${log_file}" 2>&1
    echo "imageAndMask2Fused (vote map):"
    /pr2mask/imageAndMask2Fused /data/input "${output2}/vote_map" "${output2}" --votemapmax 65535 --votemapagree 0.5 -u "${VERSION}_votemap" -s "peak agreement {peak_agreement}" -i "$VERSION" >> "${log_file}" 2>&1

    # Four DICOM series are sent back to PACS: the report (reports/), the raw
    # segmentation mask (mask/), the fused mask overlay (fused/) and the fused
    # vote-map / agreement overlay (fused_vote_map/). The raw probability map
    # (vote_map/), the B&W labels, the redcap JSON and the structured report
    # (*.dcm) are intentionally not copied.
    if [ ! -d "${output2}"/fused ]; then
        echo "Error: no /fused folder found in ${output2}"
    fi
    cp -R "${output2}"/fused "${output}"

    if [ ! -d "${output2}"/fused_vote_map ]; then
        echo "Error: no /fused_vote_map folder found in ${output2}"
    fi
    cp -R "${output2}"/fused_vote_map "${output}"

    if [ ! -d "${output2}"/reports ]; then
        echo "Error: no /report folder found in ${output2}"
    fi
    cp -R "${output2}"/reports "${output}"

    if [ ! -d "${output2}"/mask ]; then
        echo "Error: no /mask folder found in ${output2}"
    fi
    cp -R "${output2}"/mask "${output}"

    chmod -R 777 /output
fi
echo "$(date): processing done" >> "${log_file}"
