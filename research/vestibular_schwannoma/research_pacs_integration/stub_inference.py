#!/usr/bin/env python3
'''
5-fold soft-vote ensemble patch-based inference for vestibular schwanoma segmentation.
./stub_inference.py <input_dir> <output_dir> --model-type [unet]

Loads the five cross-validation fold learners (fold_*.pkl) and runs patch-based
sliding-window inference as one ensemble. PatchInferenceEngine averages the
per-patch class probabilities across the folds before argmax, so the folds
soft-vote into a single segmentation. Each fold is an exported fastai learner,
loaded as-is with load_learner.
'''

import argparse
import glob
import json
from pathlib import Path

import numpy as np
from fastai.learner import load_learner
import fastMONAI
from fastMONAI.vision_all import *
from fastMONAI.vision_patch import PatchConfig, PatchInferenceEngine
from fastMONAI.vision_inference import keep_largest
from imagedata.series import Series


# Model configuration (single model type; five folds soft-voted as one ensemble)
MODEL_CONFIGS = {
    "unet": {
        "models_dir": "vs5f_unet_models",
        "weights_glob": "fold_*.pkl",
        "seg_uid": "U5S1",
        "prob_uid": "U5P1",
        "display_name": "UNet 5-fold ensemble",
    },
}

SCRIPT_DIR = Path(__file__).parent

# Runtime batching knob, not a PatchConfig field.
SW_BATCH_SIZE = 2


def save_series_pred(series_obj, save_dir, val='1234'):
    """Save series prediction with updated DICOM UIDs.

    Makes sure we get derived UIDs to allow for overwrite of image objects in PACS.
    """
    my_seriesInstanceUID = series_obj.seriesInstanceUID[:-4] + val
    series_obj.seriesInstanceUID = my_seriesInstanceUID

    if hasattr(series_obj, 'patientID') and series_obj.patientID:
        my_studyID = series_obj.patientID[3:] if len(series_obj.patientID) > 3 else series_obj.patientID
        series_obj.studyID = my_studyID

    base_sop_uid = series_obj.getDicomAttribute('SOPInstanceUID')

    for slice_idx in range(series_obj.slices):
        my_SOPInstanceUID = base_sop_uid[:-8] + val + str(slice_idx).zfill(4)
        series_obj.setDicomAttribute('SOPInstanceUID', my_SOPInstanceUID, slice=slice_idx)

    series_obj.write(save_dir, opts={'keep_uid': True}, formats=['dicom'])


def create_dicom_mask(pred, dicom_input_path, output_dir, uid_suffix, software_versions):
    """
    Create DICOM series from segmentation mask using template series metadata.

    `pred` is the prediction in original image space (a tensor/array, already
    resized and reoriented by PatchInferenceEngine.predict()).
    """
    mask_obj = Series(str(dicom_input_path), opts={'slice_tolerance': 1e-2})

    # Any SoftwareVersions carried over from the source series is replaced.
    mask_obj.setDicomAttribute('SoftwareVersions', software_versions)

    # Mark the mask as a derived/secondary object in ImageType (0008,0008):
    # value 1 -> DERIVED, value 2 -> SECONDARY, keep any trailing values, and
    # append a MASK marker (e.g. ORIGINAL\PRIMARY\M\ND -> DERIVED\SECONDARY\M\ND\MASK).
    image_type = mask_obj.getDicomAttribute('ImageType')
    image_type = [] if image_type is None else (
        [image_type] if isinstance(image_type, str) else list(image_type))
    mask_obj.setDicomAttribute('ImageType', ['DERIVED', 'SECONDARY'] + image_type[2:] + ['MASK'])

    new_mask = pred.numpy()
    new_mask = new_mask.squeeze()
    new_mask = np.transpose(new_mask, (-1, 1, 0))
    new_mask = new_mask.copy()  # contiguous, required by the Series setter
    new_mask = new_mask.astype(np.uint16)

    mask_obj[:] = new_mask

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    save_series_pred(mask_obj, str(output_path), val=uid_suffix)

    return output_path


def create_dicom_prob_mask(pred, dicom_input_path, output_dir, uid_suffix):
    """
    Create DICOM series from probability mask.
    Scales probabilities from [0,1] to uint16 [0,65535].

    `pred` is the foreground probability in original image space (a tensor/array).
    """
    mask_obj = Series(str(dicom_input_path), opts={'slice_tolerance': 1e-2})

    prob_data = pred.numpy().squeeze()
    prob_data = np.transpose(prob_data, (-1, 1, 0))
    prob_data = prob_data.copy()  # contiguous, required by the Series setter
    prob_scaled = (prob_data * 65535).astype(np.uint16)

    mask_obj[:] = prob_scaled

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    save_series_pred(mask_obj, str(output_path), val=uid_suffix)

    return output_path


def load_models(config):
    """Load the exported fold learners and return their eval-mode networks as a list.

    Each fold learner (fold_*.pkl) is an exported fastai learner, loaded as-is with
    load_learner. The returned list is handed to PatchInferenceEngine, which
    soft-votes the folds (per-patch probability averaging) at inference time.
    """
    paths = sorted(glob.glob(str(SCRIPT_DIR / config["models_dir"] / config["weights_glob"])))
    assert paths, (
        f"No fold_*.pkl found in {config['models_dir']}; "
        "copy the exported fold learners there "
        "(see README / prepare_fold_models.py)")

    print(f"\nLoading {config['display_name']} folds from: "
          f"{SCRIPT_DIR / config['models_dir']}")
    models = []
    for path in paths:
        learn = load_learner(path, cpu=True)
        model = learn.model
        model.eval()
        models.append(model)
        print(f"  Loaded fold: {path}")

    if len(paths) != 5:
        print(f"  WARNING: expected 5 folds, loaded {len(paths)}")

    return models


def build_software_versions(models_dir, model_type, n_folds):
    """Build the SoftwareVersions values recording the ensemble provenance.

    Reads one MLflow run id per line from mlflow_run_ids.txt (guarded so a
    missing or unreadable file never aborts inference) and returns the model
    tag, the per-fold run-id prefixes, and the fastMONAI version.
    """
    try:
        run_ids = [line.strip() for line
                   in (models_dir / "mlflow_run_ids.txt").read_text().splitlines()
                   if line.strip()]
    except OSError:
        run_ids = []
    return ([f"{model_type}-{n_folds}fold"]
            + [rid[:8] for rid in run_ids]
            + [f"fastMONAI {fastMONAI.__version__}"])


def run_inference(datafolder, output, model_type, use_tta=False):
    """Run 5-fold soft-vote ensemble patch-based inference for the model type.

    use_tta enables 8-flip mirror test-time augmentation (default off).
    """
    config = MODEL_CONFIGS[model_type]
    display_name = config["display_name"]

    print("=" * 60)
    print(f"{display_name} Inference - Vestibular Schwanoma Segmentation")
    print("=" * 60)

    models_dir = SCRIPT_DIR / config["models_dir"]

    # Loaded first so a missing or empty models directory fails before anything else runs.
    models = load_models(config)

    # Single source of truth for preprocessing and patch settings, so training and
    # inference stay in sync.
    settings_path = models_dir / "inference_patch_config.json"
    print(f"\nLoading inference settings from: {settings_path}")
    patch_config = PatchConfig(**load_patch_variables(settings_path))
    print(f"  Patch size: {patch_config.patch_size}")
    print(f"  Reorder: {patch_config.apply_reorder}")
    print(f"  Resample: {patch_config.target_spacing}")

    software_versions = build_software_versions(models_dir, model_type, len(models))

    engine = PatchInferenceEngine(models, patch_config, sw_batch_size=SW_BATCH_SIZE)

    # predict() returns [n_classes, *spatial], but the reorientation path adds a leading
    # singleton ([1, n_classes, *spatial]); squeeze(0) normalizes both, since it only
    # drops dim 0 when its size is 1 and n_classes here is 2.
    print(f"\nRunning patch-based inference "
          f"(patch={patch_config.patch_size}, overlap={patch_config.patch_overlap}, "
          f"agg={patch_config.aggregation_mode}, "
          f"TTA={'on' if use_tta else 'off'})...")
    prob = engine.predict(datafolder, return_probabilities=True, tta=use_tta).squeeze(0)
    tumor_prob = prob[1]
    segmentation = keep_largest(prob.argmax(0).float())

    print(f"\nSaving outputs to: {output}")

    mask_output_dir = output + '/mask'
    create_dicom_mask(segmentation, datafolder, mask_output_dir,
                      uid_suffix=config["seg_uid"], software_versions=software_versions)
    print(f"  Saved segmentation mask to: {mask_output_dir}")

    prob_output_dir = output + '/vote_map'
    create_dicom_prob_mask(tumor_prob, datafolder, prob_output_dir, uid_suffix=config["prob_uid"])
    print(f"  Saved probability mask to: {prob_output_dir}")

    # Optional description passthrough (guarded so a missing/foreign descr.json
    # never aborts inference after the masks have been written)
    description = {}
    input_description_path = Path(datafolder + "/descr.json")
    if input_description_path.is_file():
        try:
            with open(input_description_path, "r") as file:
                description = json.load(file)
            if isinstance(description, list) and description and isinstance(description[0], dict):
                description[0]["ProbabilityMask"] = prob_output_dir
        except (json.JSONDecodeError, OSError) as err:
            print(f"  Warning: could not read descr.json ({err})")

    seg_array = segmentation.cpu().numpy().squeeze()
    tumor_voxels = np.sum(seg_array > 0)
    total_voxels = seg_array.size

    print("\n" + "=" * 60)
    print(f"{display_name.upper()} INFERENCE COMPLETE")
    print("=" * 60)
    print(f"Folds ensembled: {len(models)}")
    print(f"TTA: {'on' if use_tta else 'off'}")
    print(f"Tumor voxels: {int(tumor_voxels)}")
    print(f"Total voxels: {total_voxels}")
    print(f"Tumor percentage: {100 * tumor_voxels / total_voxels:.4f}%")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        prog='VestibularSchwanomaSegmentation',
        description='Vestibular schwanoma 5-fold ensemble segmentation script.'
    )
    parser.add_argument('fn', type=str, help='Directory name of the input folder')
    parser.add_argument('on', type=str, help='Directory name for the output')
    parser.add_argument('--model-type', type=str, choices=['unet'],
                        default='unet', help='Model type to use (default: unet)')
    parser.add_argument('--tta', action='store_true',
                        help='Enable 8-flip mirror test-time augmentation '
                             '(default: off; much slower on CPU, ~7-9x).')
    args = parser.parse_args()

    datafolder = args.fn + '/input'
    output = args.on

    run_inference(datafolder, output, args.model_type, use_tta=args.tta)


if __name__ == "__main__":
    main()
