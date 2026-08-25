#!/usr/bin/env python3
"""
summary_symprod_simpson_um.py - Summarize ATE error from symprod_simpson_UM.py.

Scenarios:
  - additive: symprod_simpson_um
  - nonadditive: symprod_simpson_nonadditive_um

Reads (per scenario):
  - <results_root>/<scenario_results_dir>/ate_estimates.csv
  - <data_root>/<scenario_data_dir>/ate_true.csv

Writes:
  1) By-scenario summary: scenario, method, n_size (and k_adapt if present)
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


SCENARIOS = {
    "additive": {
        "results_dir": "symprod_simpson_um",
        "data_dir": "symprod_simpson_um",
    },
    "nonadditive": {
        "results_dir": "symprod_simpson_nonadditive_um",
        "data_dir": "symprod_simpson_nonadditive_um",
    },
}


def parse_scenarios(value: str):
    items = [item.strip().lower() for item in value.split(",") if item.strip()]
    unknown = [item for item in items if item not in SCENARIOS]
    if unknown:
        raise argparse.ArgumentTypeError(
            f"Unknown scenarios: {', '.join(unknown)}. Available: {', '.join(SCENARIOS)}."
        )
    return items


def _resolve_path(base: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base / path


def _load_ate_estimates(path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        raise RuntimeError(f"Failed to read {path}: {exc}") from exc

    required = {"seed", "n_size", "method", "ate_hat"}
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise RuntimeError(f"{path} missing columns {missing}")

    cols = ["seed", "n_size", "method", "ate_hat"]
    if "k_adapt" in df.columns:
        cols.append("k_adapt")

    df = df[cols].copy()
    df["seed"] = pd.to_numeric(df["seed"], errors="coerce").astype("Int64")
    df["n_size"] = pd.to_numeric(df["n_size"], errors="coerce").astype("Int64")
    df["ate_hat"] = pd.to_numeric(df["ate_hat"], errors="coerce")
    df["method"] = df["method"].astype(str)
    if "k_adapt" in df.columns:
        df["k_adapt"] = pd.to_numeric(df["k_adapt"], errors="coerce").astype("Int64")
    return df


def _load_ate_true_scalar(path: Path) -> float:
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        raise RuntimeError(f"Failed to read {path}: {exc}") from exc

    if "ate_true" not in df.columns:
        raise RuntimeError(f"{path} missing column 'ate_true'")
    if df.shape[0] < 1:
        raise RuntimeError(f"{path} has no rows")

    value = pd.to_numeric(df["ate_true"].iloc[0], errors="coerce")
    if pd.isna(value):
        raise RuntimeError(f"{path} has non-numeric ate_true in first row")
    return float(value)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize symprod_simpson_UM ATE absolute error across additive and nonadditive scenarios."
    )
    parser.add_argument(
        "--scenarios",
        type=parse_scenarios,
        default=parse_scenarios("additive,nonadditive"),
        help="Comma-separated scenarios: additive, nonadditive.",
    )
    parser.add_argument(
        "--results_root",
        type=str,
        default="results",
        help="Root directory containing scenario result folders.",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="data",
        help="Root directory containing scenario data folders.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output path for by-scenario summary CSV.",
    )
    parser.add_argument(
        "--output_overall",
        type=str,
        default=None,
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()

    acee_root = Path(__file__).resolve().parents[1]
    results_root = _resolve_path(acee_root, args.results_root)
    data_root = _resolve_path(acee_root, args.data_root)

    output_path = (
        _resolve_path(acee_root, args.output)
        if args.output
        else results_root / "summary_symprod_simpson_um_ate_abs_error.csv"
    )
    scenario_frames = []
    for scenario in args.scenarios:
        cfg = SCENARIOS[scenario]
        estimates_path = results_root / cfg["results_dir"] / "ate_estimates.csv"
        truth_path = data_root / cfg["data_dir"] / "ate_true.csv"

        if not estimates_path.exists():
            print(f"Warning: missing estimates for {scenario}: {estimates_path}")
            continue
        if not truth_path.exists():
            print(f"Warning: missing truth for {scenario}: {truth_path}")
            continue

        try:
            estimates = _load_ate_estimates(estimates_path)
            ate_true = _load_ate_true_scalar(truth_path)
        except Exception as exc:
            print(f"Warning: skipping {scenario}: {exc}")
            continue

        estimates["scenario"] = scenario
        estimates["ate_true"] = ate_true
        scenario_frames.append(estimates)

    if not scenario_frames:
        raise SystemExit("No valid scenario inputs found.")

    merged = pd.concat(scenario_frames, ignore_index=True)
    merged["ate_abs_error"] = (merged["ate_hat"] - merged["ate_true"]).abs()

    group_cols = ["scenario", "method", "n_size"]
    if "k_adapt" in merged.columns:
        group_cols.append("k_adapt")

    summary_sort_cols = ["scenario", "n_size", "method"]
    if "k_adapt" in merged.columns:
        summary_sort_cols.append("k_adapt")

    summary = (
        merged.dropna(subset=["scenario", "n_size", "ate_abs_error"])
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

    print()
    print("Summary (mean ATE abs error by scenario):")
    print(summary.to_string(index=False))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False)

    print(f"\nSaved by-scenario summary CSV to: {output_path}")


if __name__ == "__main__":
    main()
