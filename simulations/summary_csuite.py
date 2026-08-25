#!/usr/bin/env python3
"""
summary_csuite.py - Summarize csuite ATE results for ACEE and DECI.

Computes MAE per seed (then averages) and standard errors for:
  - ACEE (acee_* simulation outputs)
  - DECI Gaussian
  - DECI Spline

Scenarios: nonlin_simpson, nonlin_simpson_nonadditive, symprod_simpson,
           symprod_simpson_nonadditive, sachs
Sample sizes: 500, 1000
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

SCENARIOS = [
    "nonlin_simpson",
    "nonlin_simpson_nonadditive",
    "symprod_simpson",
    "symprod_simpson_nonadditive",
    "sachs",
]

SAMPLE_SIZES = [500, 1000]

SACHS_OUTCOMES = ["ate_erk", "ate_akt", "ate_raf", "ate_jnk", "ate_p38"]

METHODS = [
    ("acee", "ACEE"),
    ("mlp", "MLP"),
    ("deci_gaussian", "DECI Gaussian"),
    ("deci_spline", "DECI Spline"),
]


def load_results(path: Path, outcome_cols: list[str]):
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        print(f"Warning: failed to read {path}: {exc}")
        return None

    required = {"seed", "n_samples"} | set(outcome_cols)
    if not required.issubset(df.columns):
        print(f"Warning: {path} missing columns {sorted(required)}")
        return None

    cols = ["seed", "n_samples"] + outcome_cols
    df = df[cols].copy()
    df["seed"] = df["seed"].astype(int)
    df["n_samples"] = df["n_samples"].astype(int)
    return df


def load_true_ate(path: Path, outcome_cols: list[str]):
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        print(f"Warning: failed to read {path}: {exc}")
        return None

    if outcome_cols == ["ATE"]:
        if "ate_true" not in df.columns:
            print(f"Warning: {path} missing ate_true column")
            return None
        return np.array([float(df["ate_true"].iloc[0])], dtype=float)

    missing = [col for col in outcome_cols if col not in df.columns]
    if missing:
        print(f"Warning: {path} missing columns {missing}")
        return None
    return df[outcome_cols].iloc[0].to_numpy(dtype=float)


def compute_seed_metrics(df: pd.DataFrame, outcome_cols: list[str], true_values: np.ndarray, n_samples: int):
    df_size = df[df["n_samples"] == n_samples]
    if df_size.empty:
        return np.nan, np.nan, 0

    seed_mae = []
    for seed, group in df_size.groupby("seed"):
        estimates = group[outcome_cols].mean().to_numpy(dtype=float)
        mask = np.isfinite(estimates) & np.isfinite(true_values)
        if not np.any(mask):
            continue
        errors = estimates[mask] - true_values[mask]
        mae = float(np.mean(np.abs(errors)))
        seed_mae.append(mae)

    if not seed_mae:
        return np.nan, np.nan, 0

    seed_mae = np.array(seed_mae, dtype=float)
    n_seeds = int(seed_mae.size)
    mae_mean = float(np.mean(seed_mae))
    mae_se = float(np.std(seed_mae, ddof=1) / np.sqrt(n_seeds)) if n_seeds > 1 else np.nan
    return mae_mean, mae_se, n_seeds


def main():
    parser = argparse.ArgumentParser(description="Summarize csuite results for ACEE and DECI.")
    parser.add_argument(
        "--data_root",
        type=str,
        default="data",
        help="Root directory for csuite data (ate_true.csv).",
    )
    parser.add_argument(
        "--acee_results_root",
        type=str,
        default="results",
        help="Root directory for ACEE results.",
    )
    parser.add_argument(
        "--deci_results_root",
        type=str,
        default="competitors/deci/result/csuite",
        help="Root directory for DECI results.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Directory to save summary CSV.",
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default="summary_csuite.csv",
        help="Summary CSV filename.",
    )
    args = parser.parse_args()

    repository_root = Path(__file__).resolve().parents[1]
    data_root = repository_root / args.data_root
    acee_root = repository_root / args.acee_results_root
    deci_root = repository_root / args.deci_results_root
    output_dir = repository_root / args.output_dir

    rows = []

    print("=" * 80)
    print("CSUITE Summary: ACEE vs DECI (Gaussian/Spline)")
    print("=" * 80)

    for scenario in SCENARIOS:
        outcome_cols = ["ATE"] if scenario != "sachs" else SACHS_OUTCOMES

        ate_true_path = data_root / f"csuite_{scenario}" / "ate_true.csv"
        true_values = load_true_ate(ate_true_path, outcome_cols)
        if true_values is None:
            true_values = np.full(len(outcome_cols), np.nan, dtype=float)

        acee_path = acee_root / scenario / "acee_ate_all_seeds.csv"
        mlp_path = acee_root / scenario / "mlp_ate_all_seeds.csv"
        deci_gaussian_path = deci_root / scenario / "deci_gaussian_effect_all_seeds.csv"
        deci_spline_path = deci_root / scenario / "deci_spline_effect_all_seeds.csv"

        data_by_method = {
            "acee": load_results(acee_path, outcome_cols),
            "mlp": load_results(mlp_path, outcome_cols),
            "deci_gaussian": load_results(deci_gaussian_path, outcome_cols),
            "deci_spline": load_results(deci_spline_path, outcome_cols),
        }

        for n_samples in SAMPLE_SIZES:
            row = {"scenario": scenario, "n_samples": int(n_samples)}

            for method_key, _ in METHODS:
                df_method = data_by_method.get(method_key)
                if df_method is None:
                    mae = np.nan
                    mae_se = np.nan
                    n_seeds = 0
                else:
                    mae, mae_se, n_seeds = compute_seed_metrics(
                        df_method, outcome_cols, true_values, n_samples
                    )
                row[f"{method_key}_mae"] = mae
                row[f"{method_key}_mae_se"] = mae_se
                row[f"{method_key}_n"] = n_seeds

            rows.append(row)

    df_summary = pd.DataFrame(rows)

    print("\nSummary Table (MAE/SE):")
    print(df_summary.to_string(index=False))

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / args.output_name
    df_summary.to_csv(output_path, index=False)
    print(f"\nSaved summary CSV to: {output_path}")


if __name__ == "__main__":
    main()
