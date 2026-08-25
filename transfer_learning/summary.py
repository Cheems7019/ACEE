#!/usr/bin/env python3
"""
Summarize mean ATE absolute error across dim_phi, n1, and methods.
"""

import argparse
from pathlib import Path

import pandas as pd


def _load_results(path: Path, method: str):
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        print(f"Warning: failed to read {path}: {exc}")
        return None

    if "ate_abs_error" not in df.columns:
        if "ate_hat" in df.columns and "ate_true" in df.columns:
            df["ate_abs_error"] = (df["ate_hat"] - df["ate_true"]).abs()
        elif "ate_hat_bc" in df.columns and "ate_true" in df.columns:
            df["ate_abs_error"] = (df["ate_hat_bc"] - df["ate_true"]).abs()
        else:
            print(f"Warning: {path} missing ate_abs_error and ATE columns")
            return None

    required = {"n1", "dim_phi", "ate_abs_error"}
    missing = [col for col in required if col not in df.columns]
    if missing:
        print(f"Warning: {path} missing columns {missing}")
        return None

    df = df[["n1", "dim_phi", "ate_abs_error"]].copy()
    df["method"] = method
    return df


def _resolve_path(base: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base / path


def main():
    parser = argparse.ArgumentParser(
        description="Summarize ATE absolute error for transfer-learning results."
    )
    parser.add_argument(
        "--results_root",
        type=str,
        default="results",
        help="Root results directory (default: transfer_learning/results).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="summary_transfer_learning.csv",
        help="Summary CSV filename.",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).parent.resolve()
    results_root = _resolve_path(script_dir, args.results_root)
    output_path = _resolve_path(script_dir, args.output)

    method_files = {
        "TM_positive": results_root / "TM_positive" / "ate_estimates_dimphi_20.csv",
        "TM_negative": results_root / "TM_negative" / "ate_estimates_dimphi_20.csv",
        "TM_positive_bc": results_root / "TM_positive_bc" / "ate_estimates_bc_dimphi_20.csv",
        "TM_negative_bc": results_root / "TM_negative_bc" / "ate_estimates_bc_dimphi_20.csv",
    }

    frames = []
    for method, path in method_files.items():
        if not path.is_file():
            print(f"Warning: missing results file {path}")
            continue
        df = _load_results(path, method)
        if df is not None:
            frames.append(df)

    if not frames:
        print("No valid result files found.")
        return

    all_df = pd.concat(frames, ignore_index=True)
    all_df["n1"] = pd.to_numeric(all_df["n1"], errors="coerce")
    all_df["dim_phi"] = pd.to_numeric(all_df["dim_phi"], errors="coerce")
    all_df["ate_abs_error"] = pd.to_numeric(all_df["ate_abs_error"], errors="coerce")

    summary = (
        all_df.dropna(subset=["n1", "dim_phi", "ate_abs_error"])
        .groupby(["method", "n1", "dim_phi"], as_index=False)
        .agg(
            ate_abs_error_mean=("ate_abs_error", "mean"),
            ate_abs_error_std=("ate_abs_error", "std"),
            n=("ate_abs_error", "count"),
        )
        .sort_values(["method", "n1", "dim_phi"])
    )
    summary["ate_abs_error_se"] = summary["ate_abs_error_std"] / summary["n"].pow(0.5)
    summary = summary.drop(columns=["ate_abs_error_std"])

    print()
    print("Summary (mean ATE abs error):")
    print(summary.to_string(index=False))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False)
    print(f"\nSaved summary CSV to: {output_path}")


if __name__ == "__main__":
    main()
