#!/usr/bin/env python3
"""
summary_sachs_um.py - Summarize ATE error from simulations/Sachs_UM.py.

Reads:
  - <results_dir>/ate_estimates.csv
  - <data_dir>/ate_true.csv

Computes per-row ate_abs_error = |ate_hat - ate_true| and writes:
  1) By-outcome summary: method, n_size, outcome (and k_adapt if present)
  2) Overall summary across outcomes using seed-level MAE
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


EXPECTED_OUTCOMES = ("erk", "akt", "raf", "jnk", "p38")


def _resolve_path(base: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base / path


def _canonicalize_outcome(series: pd.Series) -> pd.Series:
    clean = series.astype(str).str.strip().str.lower()
    clean = clean.str.replace(r"^ate_", "", regex=True)
    return clean


def _load_ate_estimates(path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        raise RuntimeError(f"Failed to read {path}: {exc}") from exc

    required = {"seed", "n_size", "method", "outcome", "ate_hat"}
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise RuntimeError(f"{path} missing columns {missing}")

    cols = ["seed", "n_size", "method", "outcome", "ate_hat"]
    if "k_adapt" in df.columns:
        cols.append("k_adapt")

    df = df[cols].copy()
    df["seed"] = pd.to_numeric(df["seed"], errors="coerce").astype("Int64")
    df["n_size"] = pd.to_numeric(df["n_size"], errors="coerce").astype("Int64")
    df["ate_hat"] = pd.to_numeric(df["ate_hat"], errors="coerce")
    df["method"] = df["method"].astype(str)
    df["outcome"] = _canonicalize_outcome(df["outcome"])
    if "k_adapt" in df.columns:
        df["k_adapt"] = pd.to_numeric(df["k_adapt"], errors="coerce").astype("Int64")
    return df


def _load_ate_true_long(path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        raise RuntimeError(f"Failed to read {path}: {exc}") from exc

    ate_cols = [c for c in df.columns if str(c).startswith("ate_")]
    if not ate_cols:
        raise RuntimeError(f"{path} has no columns starting with 'ate_'")
    if df.shape[0] < 1:
        raise RuntimeError(f"{path} has no rows")

    row = df.iloc[0][ate_cols]
    truth = (
        row.rename_axis("outcome_col")
        .reset_index(name="ate_true")
        .assign(outcome=lambda x: _canonicalize_outcome(x["outcome_col"]))
        .drop(columns=["outcome_col"])
    )
    truth["ate_true"] = pd.to_numeric(truth["ate_true"], errors="coerce")
    return truth


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize Sachs_UM ATE absolute error from Sachs_UM.py outputs."
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results/Sachs_UM",
        help="Results directory from Sachs_UM.py (contains ate_estimates.csv).",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/sachs_um",
        help="Data directory from DataGen_Sachs_UM.py (contains ate_true.csv).",
    )
    parser.add_argument(
        "--results",
        type=str,
        default=None,
        help="Optional direct path to ate_estimates.csv (overrides --results_dir).",
    )
    parser.add_argument(
        "--truth",
        type=str,
        default=None,
        help="Optional direct path to ate_true.csv (overrides --data_dir).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output path for by-outcome summary CSV.",
    )
    parser.add_argument(
        "--output_overall",
        type=str,
        default=None,
        help="Optional output path for overall summary CSV.",
    )
    args = parser.parse_args()

    acee_root = Path(__file__).resolve().parents[1]
    results_dir = _resolve_path(acee_root, args.results_dir)
    data_dir = _resolve_path(acee_root, args.data_dir)

    results_path = (
        _resolve_path(acee_root, args.results)
        if args.results
        else results_dir / "ate_estimates.csv"
    )
    truth_path = (
        _resolve_path(acee_root, args.truth)
        if args.truth
        else data_dir / "ate_true.csv"
    )
    output_path = (
        _resolve_path(acee_root, args.output)
        if args.output
        else results_dir / "summary_sachs_um_ate_abs_error.csv"
    )
    output_overall_path = (
        _resolve_path(acee_root, args.output_overall)
        if args.output_overall
        else results_dir / "summary_sachs_um_overall_ate_abs_error.csv"
    )

    estimates = _load_ate_estimates(results_path)
    truth = _load_ate_true_long(truth_path)

    estimate_outcomes = set(estimates["outcome"].dropna().tolist())
    truth_outcomes = set(truth["outcome"].dropna().tolist())
    expected_outcomes = set(EXPECTED_OUTCOMES)

    unexpected_estimate_outcomes = sorted(estimate_outcomes - expected_outcomes)
    if unexpected_estimate_outcomes:
        print(
            "Warning: unexpected outcome names in estimates: "
            + ", ".join(unexpected_estimate_outcomes)
        )

    missing_truth_outcomes = sorted(expected_outcomes - truth_outcomes)
    if missing_truth_outcomes:
        print(
            "Warning: missing expected outcomes in truth: "
            + ", ".join(missing_truth_outcomes)
        )

    merged = estimates.merge(truth, on="outcome", how="left")
    missing_truth = int(merged["ate_true"].isna().sum())
    if missing_truth:
        print(f"Warning: {missing_truth} rows missing ate_true after merge.")

    merged["ate_abs_error"] = (merged["ate_hat"] - merged["ate_true"]).abs()

    group_cols = ["method", "n_size", "outcome"]
    if "k_adapt" in merged.columns:
        group_cols.append("k_adapt")

    summary_sort_cols = ["n_size", "method", "outcome"]
    if "k_adapt" in merged.columns:
        summary_sort_cols.append("k_adapt")

    summary = (
        merged.dropna(subset=["n_size", "outcome", "ate_abs_error"])
        .groupby(group_cols, as_index=False, dropna=False)
        .agg(
            ate_abs_error_mean=("ate_abs_error", "mean"),
            ate_abs_error_std=("ate_abs_error", "std"),
            n=("ate_abs_error", "count"),
        )
        .sort_values(summary_sort_cols)
    )
    summary["ate_abs_error_se"] = np.where(
        summary["n"] > 1,
        summary["ate_abs_error_std"] / summary["n"].pow(0.5),
        np.nan,
    )
    summary = summary.drop(columns=["ate_abs_error_std"])

    # Overall across outcomes: first compute per-seed MAE over outcomes,
    # then summarize mean/SE across seeds.
    seed_group_cols = ["method", "n_size", "seed"]
    overall_group_cols = ["method", "n_size"]
    if "k_adapt" in merged.columns:
        seed_group_cols.append("k_adapt")
        overall_group_cols.append("k_adapt")

    seed_level = (
        merged.dropna(subset=["seed", "n_size", "outcome", "ate_abs_error"])
        .groupby(seed_group_cols, as_index=False, dropna=False)
        .agg(seed_mae=("ate_abs_error", "mean"))
    )

    overall_sort_cols = ["n_size", "method"]
    if "k_adapt" in merged.columns:
        overall_sort_cols.append("k_adapt")

    overall = (
        seed_level.groupby(overall_group_cols, as_index=False, dropna=False)
        .agg(
            ate_abs_error_mean=("seed_mae", "mean"),
            ate_abs_error_std=("seed_mae", "std"),
            n=("seed_mae", "count"),
        )
        .sort_values(overall_sort_cols)
    )
    overall["ate_abs_error_se"] = np.where(
        overall["n"] > 1,
        overall["ate_abs_error_std"] / overall["n"].pow(0.5),
        np.nan,
    )
    overall = overall.drop(columns=["ate_abs_error_std"])

    print()
    print(f"Loaded estimates: {results_path}")
    print(f"Loaded truth: {truth_path}")
    print("Summary (mean ATE abs error):")
    print(summary.to_string(index=False))
    print()
    print("Summary (overall across outcomes):")
    print(overall.to_string(index=False))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False)
    output_overall_path.parent.mkdir(parents=True, exist_ok=True)
    overall.to_csv(output_overall_path, index=False)
    print(f"\nSaved by-outcome summary CSV to: {output_path}")
    print(f"Saved overall summary CSV to: {output_overall_path}")


if __name__ == "__main__":
    main()
