#!/bin/bash --login
# The --login ensures the bash configuration is loaded.

set -euo pipefail

output="/output"
conda_env="${CONDA_DEFAULT_ENV:-}"
if [ -z "$conda_env" ]; then
    echo "Usage: <conda-env>"
    exit 1
fi

MODELTOUSE="unet"
USE_TTA=1

# ROR passes container options as JSON, for example:
#   --envs 'ROR_CONT_OPTIONS={"model-type":"unet","tta":false}'
# Validate the complete object before reading values. Model support remains
# authoritative in stub_inference.py; the shell only validates safe name syntax.
ror_options="${ROR_CONT_OPTIONS:-}"
if [ -z "$ror_options" ]; then
    ror_options='{}'
fi

if ! jq -e '
    def valid_bool:
      if type == "boolean" then true
      elif type == "number" then (. == 0 or . == 1)
      elif type == "string" then
        ascii_downcase as $value
        | ["true", "false", "1", "0", "yes", "no", "on", "off"]
        | index($value) != null
      else false
      end;
    type == "object"
    and ((keys - ["model-type", "tta"]) | length == 0)
    and (if has("model-type") then (.["model-type"] | type == "string") else true end)
    and (if has("tta") then (.tta | valid_bool) else true end)
  ' >/dev/null <<< "$ror_options"; then
    echo "Error: invalid ROR_CONT_OPTIONS." >&2
    echo "Supported options: model-type, tta." >&2
    exit 2
fi

MODELTOUSE=$(jq -r '.["model-type"] // "unet"' <<< "$ror_options")
if [[ ! "$MODELTOUSE" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
    echo "Error: invalid model-type syntax: ${MODELTOUSE}" >&2
    exit 2
fi

bool_to_int() {
    local option_name="$1"
    local value="${2,,}"
    case "$value" in
      true|1|yes|on) printf '1' ;;
      false|0|no|off) printf '0' ;;
      *)
        echo "Error: ${option_name} must be a boolean value." >&2
        return 2
        ;;
    esac
}

tta_value=$(jq -r 'if has("tta") then .tta else true end' <<< "$ror_options")
USE_TTA=$(bool_to_int "tta" "$tta_value")

# The bundle declaration selects either one model or an intentional N-model ensemble.
# TTA is on by default; set the tta key in ROR_CONT_OPTIONS to false to disable it.
SCRIPT_DIR="$(dirname "$0")"
STUB_SCRIPT="${SCRIPT_DIR}/stub_inference.py"
if [ "$USE_TTA" -eq 1 ]; then
  echo "Using ${MODELTOUSE} model (TTA: on)"
else
  echo "Using ${MODELTOUSE} model (TTA: off)"
fi
# where is pr2mask?
export PATH="/pr2mask:$PATH"

# Describe what the deployment config declares; never infer ensemble cardinality
# from whatever model files happen to match a glob.
deployment_config="${SCRIPT_DIR}/model_bundles/${MODELTOUSE}/deployment_config.json"
if [ -f "$deployment_config" ]; then
    declared_model_type=$(jq -r '.model_type // empty' "$deployment_config")
    if [ "$declared_model_type" != "$MODELTOUSE" ]; then
        echo "Error: bundle declares model-type ${declared_model_type:-<missing>}, requested ${MODELTOUSE}." >&2
        exit 1
    fi
    deployment_mode=$(jq -r '.mode' "$deployment_config")
    member_count=$(jq -r '.expected_member_count' "$deployment_config")
else
    echo "Error: no declared ${MODELTOUSE} bundle found." >&2
    exit 1
fi
if [ "$deployment_mode" = "single" ]; then
    deployment_label="single model"
else
    deployment_label="${member_count}-model ensemble"
fi
INFO="${MODELTOUSE} ${deployment_label}, Predicted $(date '+%b%d%Y')"

# if we find imageAndMask2Report and json2SR in this container
auto_report_mode=0
output2="/output_tmp"
if [ -f /pr2mask/imageAndMask2Report ]; then
    auto_report_mode=1
else
    output2="${output}"
fi

# Relax strict mode only for conda activation (its hooks are not strict-mode safe), then restore it.
set +e
set +u
set +o pipefail
conda activate "${conda_env}"
activate_status=$?
set -euo pipefail
if [ "$activate_status" -ne 0 ]; then
   echo "Error: activating conda environment \"${conda_env}\" failed."
   exit 1
fi

log_file="${output}"/stub_command.log
# Prefer the bundled inference script; otherwise run the command passed to the container.
if [ -n "$STUB_SCRIPT" ] && [ -f "$STUB_SCRIPT" ]; then
    cmd=(python "$STUB_SCRIPT" /data "$output2" --model-type "$MODELTOUSE")
    if [ "$USE_TTA" -eq 1 ]; then
        cmd+=(--tta)
    fi
else
    if [ "$#" -eq 0 ]; then
        echo "Error: no bundled inference script or fallback command was provided." >&2
        exit 1
    fi
    cmd=("$@" "$output2")
fi
printf 'run now:'
printf ' %q' "${cmd[@]}"
printf '\n'
"${cmd[@]}"

if [ "$auto_report_mode" -eq 1 ]; then
    echo "imageAndMask2Report:"
    /pr2mask/imageAndMask2Report /data/input "${output2}/mask" "${output2}" -u "$VERSION" -i "$VERSION" --reporttype mosaic -t "${INFO} " >> "${log_file}" 2>&1
    echo "imageAndMask2Fused:"
    /pr2mask/imageAndMask2Fused /data/input "${output2}/mask" "${output2}" -u "${VERSION}_fused" -i "$VERSION" >> "${log_file}" 2>&1
    echo "imageAndMask2Fused (vote map):"
    /pr2mask/imageAndMask2Fused /data/input "${output2}/vote_map" "${output2}" --votemapmax 65535 --votemapagree 0.5 -u "${VERSION}_votemap" -s "peak agreement {peak_agreement}" -i "$VERSION" >> "${log_file}" 2>&1

    # Four DICOM series are sent back to PACS: the report (reports/), the raw
    # segmentation mask (mask/), the fused mask overlay (fused/) and the fused
    # vote-map / agreement overlay (fused_vote_map/).
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
