#!/usr/bin/env python3
'''
Bundle exported cross-validation fold learners into vs5f_unet_models/ so the
UNet 5-fold ensemble container can be built.

Each exported best_learner.pkl is copied as-is: no loading, no re-export, and no
torch.compile unwrap. The raw exported learners load and run in the container,
matching the single-model vs_seg build that already runs in production.

Run once in the fastmonai-dev env, for example:
    python prepare_fold_models.py --experiment vs5f_unet
'''

import argparse
import glob
import shutil
from pathlib import Path

from fastMONAI.vision_all import *


def run_id_from_path(path):
    """Parse the MLflow run-dir hash from an mlruns-style artifact path.

    Expects a path of the form .../<run_id>/artifacts/... and returns <run_id>,
    or None when the path does not follow that layout.
    """
    parts = Path(path).parts
    if "artifacts" in parts:
        idx = parts.index("artifacts")
        if idx > 0:
            return parts[idx - 1]
    return None


def discover_learners(args):
    """Return an ordered list of (src_path, run_id) for the fold learners.

    Explicit --pkl-paths win and are used in the given order. Otherwise try
    find_fold_learners against a live MLflow store, then fall back to the on-disk
    mlruns glob. The MLflow attempt is guarded so a missing store degrades to the
    glob instead of crashing.
    """
    if args.pkl_paths:
        return [(p, run_id_from_path(p) or "unknown") for p in args.pkl_paths]

    try:
        found = find_fold_learners(args.experiment)
    except Exception as err:
        print(f"WARNING: MLflow discovery failed ({err}); using on-disk mlruns glob.")
        found = {}

    if found:
        return [(src, run_id_from_path(src) or str(fold))
                for fold, src in sorted(found.items())]

    matches = sorted(glob.glob(args.mlruns_glob))
    if matches:
        print("WARNING: no MLflow fold learners found; falling back to on-disk "
              "mlruns glob. On-disk mlruns is tag-stripped, so fold order follows "
              "path order and may not match the training fold numbers. Prefer "
              "--pkl-paths (explicit fold order) or a live MLflow store.")
    return [(src, run_id_from_path(src) or "unknown") for src in matches]


def main():
    parser = argparse.ArgumentParser(
        prog="PrepareFoldModels",
        description="Copy exported cross-validation fold learners into the "
                    "ensemble models directory (no cleaning or re-export)."
    )
    parser.add_argument("--experiment", type=str, default="vs5f_unet",
                        help="MLflow experiment to discover fold learners from "
                             "(default: vs5f_unet).")
    parser.add_argument("--out", type=Path,
                        default=Path(__file__).parent / "vs5f_unet_models",
                        help="Output models directory (default: ./vs5f_unet_models).")
    parser.add_argument("--pkl-paths", nargs="*", default=None,
                        help="Explicit exported best_learner.pkl paths in fold "
                             "order; overrides discovery.")
    parser.add_argument("--mlruns-glob", type=str,
                        default="mlruns/1/*/artifacts/model/best_learner.pkl",
                        help="Fallback glob for on-disk mlruns learners.")
    parser.add_argument("--patch-size", nargs=3, type=int, default=[192, 192, 48],
                        help="Fallback patch size if --patch-config is missing "
                             "(default: 192 192 48).")
    parser.add_argument("--target-spacing", nargs=3, type=float,
                        default=[0.4102, 0.4102, 1.5],
                        help="Fallback voxel spacing if --patch-config is missing "
                             "(default: 0.4102 0.4102 1.5).")
    parser.add_argument("--reorder", default=True, action=argparse.BooleanOptionalAction,
                        help="Reorder to RAS+ canonical orientation (default: True).")
    parser.add_argument("--patch-config", type=Path,
                        default=Path(__file__).parent.parent / "inference_patch_config.json",
                        help="Canonical training patch config to bundle for inference "
                             "(default: ../inference_patch_config.json). When present it is "
                             "the single source of truth; the --patch-size / --target-spacing "
                             "/ --reorder args are only the fallback used if it is missing.")
    args = parser.parse_args()

    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    folds = discover_learners(args)
    if not folds:
        raise SystemExit(
            "No fold learners found. Pass --pkl-paths, run against a live MLflow "
            "store, or point --mlruns-glob at exported best_learner.pkl files.")

    # Copy each exported learner as-is (no loading, no re-export, no cleaning).
    run_ids = []
    for i, (src, run_id) in enumerate(folds, start=1):
        dst = out / ("fold_" + str(i) + ".pkl")
        shutil.copy(src, dst)
        run_ids.append(run_id)
        print(f"Copied fold {i}: {src} -> {dst} (run {run_id})")

    # Ensemble provenance, one run id per line in fold_i.pkl order.
    run_ids_path = out / "mlflow_run_ids.txt"
    run_ids_path.write_text("\n".join(run_ids) + "\n")

    # Full patch-inference config, read back at inference with load_patch_variables
    # into PatchConfig(**config). Prefer the canonical training config so every
    # patch/preprocessing/postprocessing parameter (overlap, aggregation,
    # normalization, keep_largest_component, ...) stays coupled to training; fall
    # back to the CLI args + known-training defaults only if it is missing.
    settings_path = out / "inference_patch_config.json"
    if args.patch_config and args.patch_config.is_file():
        config = load_patch_variables(args.patch_config)
        print(f"Sourced patch config from {args.patch_config}")
    else:
        print(f"WARNING: {args.patch_config} not found; falling back to CLI/default "
              "patch config. Verify these match training.")
        config = dict(
            patch_size=args.patch_size,
            patch_overlap=0.5,
            aggregation_mode="hann",
            apply_reorder=args.reorder,
            target_spacing=args.target_spacing,
            keep_largest_component=True,
            normalization=[{"name": "ZNormalization",
                            "masking_method": "foreground",
                            "channel_wise": True}],
        )
    store_patch_variables(str(settings_path), **config)

    print("\n" + "=" * 60)
    print(f"Bundled {len(folds)} fold learners into {out}")
    print(f"Run ids: {run_ids}")
    print(f"Settings: {settings_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
