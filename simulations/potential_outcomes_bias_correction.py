#!/usr/bin/env python3
import argparse
import csv
import os
import sys
import warnings

import numpy as np
import pandas as pd

# Add parent directory to path to import utils
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.mbc import mbc


def parse_n_sizes(value: str):
    return [int(v) for v in value.split(",") if v.strip()]


def parse_scenarios(value: str):
    return [v.strip() for v in value.split(",") if v.strip()]


def load_train_ate_true(truth_file: str):
    if not os.path.exists(truth_file):
        warnings.warn(f"Missing train truth file: {truth_file}")
        return None

    try:
        value = np.loadtxt(truth_file, delimiter=",", comments="#")
    except Exception as exc:
        warnings.warn(f"Failed to read train truth file {truth_file}: {exc}")
        return None

    if np.ndim(value) == 0:
        return float(value)
    if value.size == 0:
        warnings.warn(f"Train truth file is empty: {truth_file}")
        return None
    return float(np.ravel(value)[0])


def main():
    parser = argparse.ArgumentParser(
        description="Bias-corrected ATE estimation for M1-M4 using saved potential outcomes."
    )
    parser.add_argument(
        "--scenarios",
        type=parse_scenarios,
        default=parse_scenarios("M1,M2,M3,M4"),
        help="Comma-separated list of scenarios (M1,M2,M3,M4).",
    )
    parser.add_argument("--n_seeds", type=int, default=50)
    parser.add_argument(
        "--n_sizes",
        type=parse_n_sizes,
        default=parse_n_sizes("1000,2000"),
        help="Comma-separated training sizes.",
    )
    parser.add_argument("--mbc_m", type=int, default=5)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    acee_root = os.path.dirname(script_dir)

    parser.add_argument(
        "--data_root",
        type=str,
        default=os.path.join(acee_root, "data"),
        help="Root directory containing EM data folders.",
    )
    parser.add_argument(
        "--results_root",
        type=str,
        default=os.path.join(acee_root, "results"),
        help="Root directory containing EM results folders with mu files.",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default=os.path.join(acee_root, "results", "EM_bc"),
        help="Directory to write BC results.",
    )
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)

    print()
    print("=" * 70)
    print("BC ESTIMATOR CONFIGURATION (M1-M4)")
    print("=" * 70)
    print(f"Scenarios: {args.scenarios}")
    print(f"Seeds: {args.n_seeds}")
    print(f"Training sizes: {args.n_sizes}")
    print(f"mbc_m: {args.mbc_m}")
    print(f"Data root: {args.data_root}")
    print(f"Results root: {args.results_root}")
    print(f"Results dir: {args.results_dir}")
    print("=" * 70)
    print()

    results_file = os.path.join(args.results_dir, "ate_estimates_bc.csv")
    if os.path.exists(results_file):
        os.remove(results_file)

    with open(results_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "scenario",
                "seed",
                "n_size",
                "mbc_m",
                "ate_hat_bc",
                "ate_true",
                "ate_abs_error",
            ]
        )

    for scenario in args.scenarios:
        scenario_data_dir = os.path.join(args.data_root, scenario)
        scenario_results_dir = os.path.join(args.results_root, scenario)

        if not os.path.isdir(scenario_data_dir):
            warnings.warn(f"Missing data dir for scenario {scenario}: {scenario_data_dir}")
            continue
        if not os.path.isdir(scenario_results_dir):
            warnings.warn(
                f"Missing results dir for scenario {scenario}: {scenario_results_dir}"
            )
            continue

        print()
        print("=" * 70)
        print(f"SCENARIO {scenario}")
        print("=" * 70)

        for seed in range(args.n_seeds):
            for n_size in args.n_sizes:
                mu_file = os.path.join(
                    scenario_results_dir, f"seed_{seed}_n_{n_size}_mu.csv"
                )
                if not os.path.exists(mu_file):
                    warnings.warn(f"Missing mu file: {mu_file}")
                    continue

                try:
                    data_mu = np.loadtxt(mu_file, delimiter=",")
                except Exception as exc:
                    warnings.warn(f"Failed to read {mu_file}: {exc}")
                    continue

                if data_mu.ndim == 1:
                    data_mu = data_mu.reshape(1, -1)

                x_dim = data_mu.shape[1] - 4
                if x_dim <= 0:
                    warnings.warn(
                        f"Unexpected column count in {mu_file}: {data_mu.shape[1]}"
                    )
                    continue

                X = data_mu[:, :x_dim]
                D = data_mu[:, x_dim]
                Y = data_mu[:, x_dim + 1]
                mu1 = data_mu[:, x_dim + 2]
                mu0 = data_mu[:, x_dim + 3]

                ate_hat_bc = mbc(
                    X=X,
                    Y=Y,
                    Tr=D,
                    M=args.mbc_m,
                    Model1=mu1,
                    Model0=mu0,
                )

                train_truth_file = os.path.join(
                    scenario_data_dir, f"n_{n_size}", f"seed_{seed}_train_truth.csv"
                )
                ate_true = load_train_ate_true(train_truth_file)
                if ate_true is not None:
                    ate_abs_error = abs(ate_hat_bc - ate_true)
                else:
                    ate_abs_error = np.nan

                with open(results_file, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(
                        [
                            scenario,
                            seed,
                            n_size,
                            args.mbc_m,
                            ate_hat_bc,
                            ate_true,
                            ate_abs_error,
                        ]
                    )

                print(
                    f"Seed {seed} | n={n_size} | ate_hat_bc={ate_hat_bc:.6f}"
                )

    print()
    print("=" * 70)
    print("CREATING FINAL SUMMARY STATISTICS")
    print("=" * 70)

    if os.path.exists(results_file):
        results_df = pd.read_csv(results_file)
        if len(results_df) == 0:
            print("No results found to summarize.")
            return

        grouped = results_df.groupby(["scenario", "n_size"])
        summary_rows = []
        for (scenario, n_size), group in grouped:
            ate_hat_mean = group["ate_hat_bc"].mean()
            ate_hat_std = group["ate_hat_bc"].std()
            ate_true_vals = group["ate_true"].dropna()
            if len(ate_true_vals) > 0:
                ate_true_mean = ate_true_vals.mean()
                ate_err = (group["ate_hat_bc"] - group["ate_true"]).abs().mean()
            else:
                ate_true_mean = np.nan
                ate_err = np.nan

            summary_rows.append(
                {
                    "scenario": scenario,
                    "n_size": n_size,
                    "ate_hat_bc_mean": ate_hat_mean,
                    "ate_hat_bc_std": ate_hat_std,
                    "ate_true_mean": ate_true_mean,
                    "ate_abs_error": ate_err,
                }
            )

        summary_df = pd.DataFrame(summary_rows)
        print()
        print("Summary by scenario and sample size:")
        print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
