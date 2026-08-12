"""Cross-validation result collection and summary helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_METRICS = (
    "dsc",
    "sensitivity",
    "precision",
    "ldr",
    "rve",
    "assd_mm",
    "hd95_mm",
    "nsd_tau0.5_mm",
    "nsd_tau1.0_mm",
    "nsd_tau2.0_mm",
)

KEY_METRICS = (
    "dsc",
    "sensitivity",
    "precision",
    "ldr",
    "rve",
    "assd_mm",
    "hd95_mm",
    "nsd_tau1.0_mm",
)


def load_fold_results(results_dir: str | Path) -> dict[int, pd.DataFrame]:
    """Load every ``fold_<n>/results.csv`` below a model results directory."""

    root = Path(results_dir)
    frames = {}
    for fold_dir in sorted(root.glob("fold_*")):
        try:
            fold = int(fold_dir.name.removeprefix("fold_"))
        except ValueError:
            continue
        csv_path = fold_dir / "results.csv"
        if csv_path.is_file():
            frames[fold] = pd.read_csv(csv_path)
    return frames


def combine_fold_results(
    fold_results: Mapping[int, pd.DataFrame],
    *,
    expected_folds: Sequence[int],
    dataset_df: pd.DataFrame,
    fold_col: str = "fold",
    case_id_col: str = "case_id",
) -> pd.DataFrame:
    """Combine held-out results only after validating fold and case coverage."""

    expected = sorted(map(int, expected_folds))
    completed = sorted(map(int, fold_results))
    if completed != expected:
        raise ValueError(
            f"Incomplete cross-validation results: expected {expected}, found {completed}"
        )

    frames = []
    for fold in expected:
        frame = fold_results[fold].copy()
        if case_id_col not in frame:
            raise ValueError(f"Fold {fold} results have no {case_id_col!r} column")
        frame[fold_col] = fold
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)

    if combined[case_id_col].duplicated().any():
        duplicates = sorted(
            combined.loc[combined[case_id_col].duplicated(False), case_id_col]
            .astype(str)
            .unique()
        )
        raise ValueError(f"Duplicate validation cases found: {duplicates}")

    required_dataset_columns = {case_id_col, fold_col}
    missing_columns = sorted(required_dataset_columns - set(dataset_df.columns))
    if missing_columns:
        raise ValueError(f"Dataset is missing columns: {missing_columns}")
    expected_cases = set(
        dataset_df.loc[dataset_df[fold_col].isin(expected), case_id_col]
    )
    observed_cases = set(combined[case_id_col])
    if observed_cases != expected_cases:
        missing = sorted(expected_cases - observed_cases)
        extra = sorted(observed_cases - expected_cases)
        raise ValueError(f"Validation case mismatch; missing={missing}, extra={extra}")
    return combined


def summarize_metrics(
    combined: pd.DataFrame, metrics: Sequence[str] = DEFAULT_METRICS
) -> pd.DataFrame:
    """Return finite mean/std/count values for each available metric."""

    rows = []
    for metric in metrics:
        if metric not in combined:
            continue
        values = pd.to_numeric(combined[metric], errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )
        rows.append(
            {
                "metric": metric,
                "mean": float(values.mean()),
                "std": float(values.std()),
                "n_finite": int(values.notna().sum()),
                "n_excluded": int(values.isna().sum()),
            }
        )
    return pd.DataFrame(rows)


def aggregate_results(
    results_dir: str | Path,
    expected_folds: Sequence[int],
    dataset_df: pd.DataFrame,
) -> pd.DataFrame | None:
    """Load, validate, persist, and print one model's pooled CV results."""

    results_dir = Path(results_dir)
    fold_results = load_fold_results(results_dir)
    if not fold_results:
        print(f"No fold results found in {results_dir}. Run training first.")
        return None
    try:
        combined = combine_fold_results(
            fold_results,
            expected_folds=expected_folds,
            dataset_df=dataset_df,
        )
    except ValueError as error:
        if str(error).startswith("Incomplete cross-validation results"):
            print(f"{error}. Summary skipped.")
            return None
        raise

    results_dir.mkdir(parents=True, exist_ok=True)
    combined.to_csv(results_dir / "cv_summary.csv", index=False)
    summary = summarize_metrics(combined)

    print(f"\n{'=' * 60}")
    print(f"  CROSS-VALIDATION SUMMARY: {results_dir.name}")
    print(f"{'=' * 60}")
    print(f"  Folds completed: {sorted(combined['fold'].unique().tolist())}")
    print(f"  Total subjects:  {len(combined)}\n")
    print(f"  {'Metric':<15} {'Mean':>10} {'Std':>10}")
    print(f"  {'-' * 35}")
    for row in summary.to_dict("records"):
        note = (
            f"  ({row['n_excluded']} non-finite excl.)"
            if row["n_excluded"]
            else ""
        )
        print(
            f"  {row['metric']:<15} {row['mean']:>10.4f} "
            f"{row['std']:>10.4f}{note}"
        )

    if "surface_status" in combined:
        print("\n  Surface-metric status counts:")
        for status, count in combined["surface_status"].value_counts().items():
            print(f"    {status:<12} {count}")

    print("\n  Per-fold DSC:")
    for fold, group in combined.groupby("fold"):
        print(f"    Fold {fold}: {group['dsc'].mean():.4f} +/- {group['dsc'].std():.4f}")
    print(f"\n  Results saved to {results_dir / 'cv_summary.csv'}")
    return combined


def build_model_comparison(
    model_results: Mapping[str, pd.DataFrame],
    metrics: Sequence[str] = KEY_METRICS,
) -> pd.DataFrame:
    """Build the machine-readable cross-model mean/std comparison table."""

    rows = []
    for model_key, combined in model_results.items():
        row = {
            "model": model_key,
            "folds": sorted(combined["fold"].unique().tolist()),
            "n_cases": int(len(combined)),
        }
        for metric in metrics:
            if metric not in combined:
                continue
            finite = pd.to_numeric(combined[metric], errors="coerce").replace(
                [np.inf, -np.inf], np.nan
            )
            row[f"{metric}_mean"] = float(finite.mean())
            row[f"{metric}_std"] = float(finite.std())
        rows.append(row)
    return pd.DataFrame(rows).set_index("model") if rows else pd.DataFrame()


def format_model_comparison(
    comparison: pd.DataFrame, metrics: Sequence[str] = KEY_METRICS
) -> pd.DataFrame:
    """Format a machine-readable comparison for compact notebook display."""

    pretty = pd.DataFrame(index=comparison.index)
    for metric in metrics:
        mean_col, std_col = f"{metric}_mean", f"{metric}_std"
        if mean_col in comparison:
            pretty[metric] = [
                f"{mean:.4f} +/- {std:.4f}"
                for mean, std in zip(comparison[mean_col], comparison[std_col])
            ]
    return pretty


def select_qualitative_cases(combined: pd.DataFrame) -> list[tuple[str, pd.Series]]:
    """Select distinct median- and worst-DSC rows for qualitative review."""

    if combined.empty or "dsc" not in combined:
        return []
    ordered = combined.sort_values("dsc").reset_index(drop=True)
    candidates = [
        ("median DSC", ordered.iloc[len(ordered) // 2]),
        ("worst DSC", ordered.iloc[0]),
    ]
    selected = []
    seen = set()
    for label, row in candidates:
        case_id = row["case_id"]
        if case_id not in seen:
            selected.append((label, row))
            seen.add(case_id)
    return selected
