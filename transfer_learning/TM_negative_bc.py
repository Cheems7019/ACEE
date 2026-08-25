#!/usr/bin/env python3
import argparse
import csv
import os
import warnings

import numpy as np
import pandas as pd

# Add parent directory to path to import utils
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.mbc import mbc


def parse_n1_list(value: str):
    return [int(v) for v in value.split(",") if v.strip()]


def parse_dim_phi_list(value: str):
    return [int(v) for v in value.split(",") if v.strip()]


def load_ate_true_map(data_dir: str):
    ate_true_file = os.path.join(data_dir, "ate_true.csv")
    if not os.path.exists(ate_true_file):
        return {}

    ate_true_map = {}
    try:
        with open(ate_true_file, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    seed = int(row["seed"])
                    if "ate_true_sate" in row and row["ate_true_sate"] not in (None, ""):
                        ate_true_map[seed] = float(row["ate_true_sate"])
                    elif "ate_true" in row and row["ate_true"] not in (None, ""):
                        ate_true_map[seed] = float(row["ate_true"])
                except (KeyError, ValueError):
                    continue
    except Exception as exc:
        warnings.warn(f"Failed to read {ate_true_file}: {exc}")
        return {}

    return ate_true_map


def main():
    parser = argparse.ArgumentParser(
        description="Bias-corrected ATE estimation for TM_negative using saved potential outcomes."
    )
    parser.add_argument("--n_seeds", type=int, default=50)
    parser.add_argument("--n0", type=int, default=200, help="Original dataset size.")
    parser.add_argument(
        "--n1",
        type=parse_n1_list,
        default=parse_n1_list("0,200,500,1000"),
        help="Auxiliary dataset sizes (comma-separated).",
    )
    parser.add_argument("--x_dim", type=int, default=15)
    parser.add_argument(
        "--dim_phi",
        type=parse_dim_phi_list,
        default=parse_dim_phi_list("20"),
        help="Shared representation dimensions (comma-separated).",
    )
    parser.add_argument("--mbc_m", type=int, default=5)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    acee_root = os.path.dirname(script_dir)

    parser.add_argument(
        "--data_dir",
        type=str,
        default=os.path.join(acee_root, "transfer_learning", "data", "TM_negative"),
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default=os.path.join(acee_root, "transfer_learning", "results", "TM_negative_bc"),
    )
    args = parser.parse_args()

    if not os.path.isdir(args.data_dir):
        raise SystemExit(f"Data dir not found: {args.data_dir}")

    os.makedirs(args.results_dir, exist_ok=True)

    print()
    print("=" * 70)
    print("BC ESTIMATOR CONFIGURATION")
    print("=" * 70)
    print(f"Seeds: {args.n_seeds}")
    print(f"Original size: {args.n0}")
    print(f"Auxiliary sizes: {args.n1}")
    print(f"X dimension: {args.x_dim}")
    print(f"dim_phi values: {args.dim_phi}")
    print(f"mbc_m: {args.mbc_m}")
    print(f"Data dir: {args.data_dir}")
    print(f"Results dir: {args.results_dir}")
    print("=" * 70)
    print()

    dim_phi_label = "_".join(str(v) for v in args.dim_phi)
    results_file = os.path.join(
        args.results_dir,
        f"ate_estimates_bc_dimphi_{dim_phi_label}.csv",
    )
    if os.path.exists(results_file):
        os.remove(results_file)

    with open(results_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["seed", "n0", "n1", "dim_phi", "mbc_m", "ate_hat_bc", "ate_true", "ate_abs_error"]
        )

    ate_true_map = load_ate_true_map(args.data_dir)

    for seed in range(args.n_seeds):
        print()
        print("=" * 70)
        print(f"SEED {seed}/{args.n_seeds - 1}")
        print("=" * 70)

        ate_true = ate_true_map.get(seed)
        if ate_true is not None:
            print(f"True ATE (from data): {ate_true:.6f}")
        else:
            print("True ATE not found for this seed.")

        for n1 in args.n1:
            data_subdir = os.path.join(args.data_dir, f"n0_{args.n0}_n1_{n1}")
            for dim_phi in args.dim_phi:
                mu_file = os.path.join(
                    data_subdir,
                    f"seed_{seed}_orig_mu_dimphi_{dim_phi}.csv",
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

                expected_cols = args.x_dim + 4
                x_dim = args.x_dim
                if data_mu.shape[1] != expected_cols:
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

                ate_abs_error = None
                if ate_true is not None:
                    ate_abs_error = abs(ate_hat_bc - ate_true)

                with open(results_file, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(
                        [
                            seed,
                            args.n0,
                            n1,
                            dim_phi,
                            args.mbc_m,
                            ate_hat_bc,
                            ate_true,
                            ate_abs_error,
                        ]
                    )

                print(
                    f"Seed {seed} | n1={n1} | dim_phi={dim_phi} | ate_hat_bc={ate_hat_bc:.6f}"
                )

    print()
    print("=" * 70)
    print("CREATING FINAL SUMMARY STATISTICS")
    print("=" * 70)

    if os.path.exists(results_file):
        results_df = pd.read_csv(results_file)
        if "ate_true" not in results_df.columns:
            results_df["ate_true"] = results_df["seed"].map(ate_true_map)
        else:
            results_df["ate_true"] = results_df["ate_true"].fillna(
                results_df["seed"].map(ate_true_map)
            )

        grouped = results_df.groupby(["n1", "dim_phi"])
        summary_rows = []
        for (n1, dim_phi), group in grouped:
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
                    "n1": n1,
                    "dim_phi": dim_phi,
                    "ate_hat_bc_mean": ate_hat_mean,
                    "ate_hat_bc_std": ate_hat_std,
                    "ate_true_mean": ate_true_mean,
                    "ate_abs_error": ate_err,
                }
            )

        summary_df = pd.DataFrame(summary_rows)
        print()
        print("Summary by auxiliary size and dim_phi:")
        print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
