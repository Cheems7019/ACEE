#!/usr/bin/env python3
import argparse
import csv
import os
import random
import sys
import warnings

# Add parent directory to path to import utils
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import torch

from sklearn.preprocessing import QuantileTransformer, StandardScaler

from utils.ddpm import (
    generate_conditional_samples,
    prepare_conditional_data,
    train_conditional_ddpm,
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def parse_hidden_dims(value: str):
    return [int(v) for v in value.split(",") if v.strip()]


def parse_n_sizes(value: str):
    return [int(v) for v in value.split(",") if v.strip()]


def load_tau_true(truth_file: str):
    if not os.path.exists(truth_file):
        warnings.warn(f"Missing truth file: {truth_file}")
        return None

    try:
        truth = np.loadtxt(truth_file, delimiter=",", comments="#")
    except Exception as exc:
        warnings.warn(f"Failed to read truth file {truth_file}: {exc}")
        return None

    if np.ndim(truth) == 0:
        return np.array([float(truth)])
    if truth.ndim == 1:
        return truth
    if truth.shape[1] < 1:
        warnings.warn(f"Truth file has no columns: {truth_file}")
        return None
    return truth[:, 0]


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


def estimate_ite_ddpm(
    ddpm,
    X: np.ndarray,
    X_orig: np.ndarray,
    scaler_y,
    scaler_xd,
    x_dim: int,
    monte_carlo_size: int,
    mc_batch_size: int,
    return_mu: bool = False,
):
    n = X.shape[0]

    # Create [X, D] pairs in original space and transform once.
    xd0_orig = np.column_stack([X_orig, np.zeros((n, 1))])
    xd1_orig = np.column_stack([X_orig, np.ones((n, 1))])

    # Transform to normalized space (preserves the joint scaling)
    cond0 = scaler_xd.transform(xd0_orig).astype(np.float32)
    cond1 = scaler_xd.transform(xd1_orig).astype(np.float32)

    if mc_batch_size is None or mc_batch_size <= 0:
        mc_batch_size = monte_carlo_size

    batch_sizes = [mc_batch_size] * (monte_carlo_size // mc_batch_size)
    if monte_carlo_size % mc_batch_size != 0:
        batch_sizes.append(monte_carlo_size % mc_batch_size)

    sum0 = np.zeros(n, dtype=np.float64)
    sum1 = np.zeros(n, dtype=np.float64)
    total_samples = 0

    # Set model to evaluation mode
    ddpm.eval()

    with torch.no_grad():
        for batch_idx, batch_size in enumerate(batch_sizes):
            if batch_idx % 10 == 0 and len(batch_sizes) > 1:
                print(f"    MC batch {batch_idx+1}/{len(batch_sizes)}")

            samples0 = generate_conditional_samples(ddpm, cond0, n_samples=batch_size)
            samples1 = generate_conditional_samples(ddpm, cond1, n_samples=batch_size)

            if samples0.shape[0] != batch_size:
                raise ValueError(
                    f"Expected {batch_size} samples, got {samples0.shape[0]}"
                )
            if samples1.shape[0] != batch_size:
                raise ValueError(
                    f"Expected {batch_size} samples, got {samples1.shape[0]}"
                )

            # Convert CUDA tensors to NumPy arrays for sklearn
            samples0_np = samples0.detach().cpu().numpy().reshape(batch_size * n, 1)
            samples1_np = samples1.detach().cpu().numpy().reshape(batch_size * n, 1)

            y0 = scaler_y.inverse_transform(samples0_np).reshape(batch_size, n)
            y1 = scaler_y.inverse_transform(samples1_np).reshape(batch_size, n)

            sum0 += y0.sum(axis=0)
            sum1 += y1.sum(axis=0)
            total_samples += batch_size

    mean0 = sum0 / total_samples
    mean1 = sum1 / total_samples
    tau_hat = mean1 - mean0
    ate_hat = float(np.mean(tau_hat))
    if return_mu:
        return tau_hat, ate_hat, mean0, mean1
    return tau_hat, ate_hat


def main():
    parser = argparse.ArgumentParser(
        description="Simulation for M3 ITE with conditional diffusion model"
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--n_seeds", type=int, default=50)
    parser.add_argument(
        "--n_sizes", type=parse_n_sizes, default=parse_n_sizes("1000,2000")
    )
    parser.add_argument("--x_dim", type=int, default=30)
    parser.add_argument(
        "--hidden_dims",
        type=parse_hidden_dims,
        default=parse_hidden_dims("512,256,256,256,128"),
        help="Hidden dims for diffusion model",
    )
    parser.add_argument("--dim_t", type=int, default=128)
    parser.add_argument("--n_steps", type=int, default=1000)
    parser.add_argument("--n_epochs", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument(
        "--batch_size",
        type=int,
        default=0,
        help="Training batch size (0 for full-batch)",
    )
    parser.add_argument(
        "--total_mc_ate",
        type=int,
        default=100000,
        help="Total MC samples for ATE (monte_carlo_size * n_size).",
    )
    parser.add_argument("--monte_carlo_size_ite", type=int, default=1000)
    parser.add_argument("--mc_batch_size", type=int, default=200)
    parser.add_argument("--use_standard_scaler", action="store_true")

    # Get script directory and set paths relative to ACEE root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    acee_root = os.path.dirname(script_dir)  # Go up from simulations to ACEE

    parser.add_argument(
        "--data_dir",
        type=str,
        default=os.path.join(acee_root, "data", "M3"),
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default=os.path.join(acee_root, "results", "M3"),
    )
    args = parser.parse_args()

    if not os.path.isdir(args.data_dir):
        raise SystemExit(f"Data dir not found: {args.data_dir}")

    os.makedirs(args.results_dir, exist_ok=True)

    print()
    print("=" * 70)
    print("EXPERIMENTAL CONFIGURATION")
    print("=" * 70)
    print(f"Device: {args.device}")
    print(f"Seeds: {args.n_seeds}")
    print(f"Training sizes: {args.n_sizes}")
    print(f"X dimension: {args.x_dim}")
    print(f"Diffusion hidden dims: {args.hidden_dims}")
    print(f"Dim t: {args.dim_t}")
    print(f"Diffusion steps: {args.n_steps}")
    print(f"Epochs: {args.n_epochs}")
    print(f"Learning rate: {args.lr}")
    print(
        f"Training batch size: {args.batch_size if args.batch_size > 0 else 'Full-batch'}"
    )
    print(f"ATE total MC: {args.total_mc_ate}")
    print(f"ITE Monte Carlo size: {args.monte_carlo_size_ite}")
    print(f"MC batch size: {args.mc_batch_size}")
    print(f"Scaler: {'StandardScaler' if args.use_standard_scaler else 'QuantileTransformer'}")
    print(f"Data dir: {args.data_dir}")
    print(f"Results dir: {args.results_dir}")
    print("=" * 70)
    print()

    results_file = os.path.join(args.results_dir, "ate_estimates.csv")
    if os.path.exists(results_file):
        os.remove(results_file)

    with open(results_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["seed", "n_size", "ate_hat", "ate_abs_error", "ite_mse", "ite_mae"])

    ate_true_map = {}

    for seed in range(args.n_seeds):
        print()
        print("=" * 70)
        print(f"SEED {seed}/{args.n_seeds - 1}")
        print("=" * 70)

        seed_everything(seed)

        for n_size in args.n_sizes:
            print()
            print("=" * 50)
            print(f"SEED {seed} | Training Size: {n_size}")
            print("=" * 50)

            data_subdir = os.path.join(args.data_dir, f"n_{n_size}")
            data_file = os.path.join(data_subdir, f"seed_{seed}_data.csv")
            if not os.path.exists(data_file):
                warnings.warn(f"Missing data file: {data_file}")
                continue

            try:
                data = np.loadtxt(data_file, delimiter=",")
            except Exception as exc:
                warnings.warn(f"Failed to read {data_file}: {exc}")
                continue

            if data.ndim == 1:
                data = data.reshape(1, -1)

            expected_cols = args.x_dim + 2
            x_dim = args.x_dim
            if data.shape[1] != expected_cols:
                warnings.warn(
                    f"Expected {expected_cols} columns in {data_file}, got {data.shape[1]}"
                )
                x_dim = data.shape[1] - 2

            X = data[:, :x_dim]
            D = data[:, x_dim]
            y = data[:, x_dim + 1]

            print(f"Loaded training data: X shape {X.shape}, y shape {y.shape}")
            print(f"Treatment balance: {D.mean():.3f}")
            print(f"y range: [{y.min():.3f}, {y.max():.3f}]")

            ite_data_file = os.path.join(data_subdir, f"seed_{seed}_ite_data.csv")
            X_ite = None
            D_ite = None
            y_ite = None
            if os.path.exists(ite_data_file):
                try:
                    ite_data = np.loadtxt(ite_data_file, delimiter=",")
                except Exception as exc:
                    warnings.warn(f"Failed to read {ite_data_file}: {exc}")
                    ite_data = None
                if ite_data is not None:
                    if ite_data.ndim == 1:
                        ite_data = ite_data.reshape(1, -1)
                    if ite_data.shape[1] != expected_cols:
                        warnings.warn(
                            f"Expected {expected_cols} columns in {ite_data_file}, got {ite_data.shape[1]}"
                        )
                    else:
                        X_ite = ite_data[:, :x_dim]
                        D_ite = ite_data[:, x_dim]
                        y_ite = ite_data[:, x_dim + 1]
                        print(
                            f"Loaded ITE data: X shape {X_ite.shape}, y shape {y_ite.shape}"
                        )
                        print(f"ITE treatment balance: {D_ite.mean():.3f}")
                        print(f"ITE y range: [{y_ite.min():.3f}, {y_ite.max():.3f}]")
            else:
                warnings.warn(f"Missing ITE data file: {ite_data_file}")

            train_truth_file = os.path.join(
                data_subdir, f"seed_{seed}_train_truth.csv"
            )
            ate_true = load_train_ate_true(train_truth_file)
            if ate_true is not None:
                ate_true_map[seed] = ate_true
            else:
                print("True ATE not found for this seed/sample size.")

            ite_truth_file = os.path.join(data_subdir, f"seed_{seed}_truth.csv")
            tau_true = load_tau_true(ite_truth_file)
            if tau_true is None:
                print("True ITE not found for this seed/sample size.")

            # Normalize data (including treatment for consistent scaling)
            X_with_d_for_norm = np.column_stack([X, D])

            if args.use_standard_scaler:
                scaler_xd = StandardScaler()
                scaler_y = StandardScaler()
            else:
                scaler_xd = QuantileTransformer(output_distribution="normal", random_state=seed)
                scaler_y = QuantileTransformer(output_distribution="normal", random_state=seed)

            scaler_xd.fit(X_with_d_for_norm)
            scaler_y.fit(y.reshape(-1, 1))

            X_with_d_norm = scaler_xd.transform(X_with_d_for_norm)
            y_norm = scaler_y.transform(y.reshape(-1, 1)).reshape(-1)

            X_norm = X_with_d_norm[:, :-1]
            D_norm = X_with_d_norm[:, -1]

            y_target, cond = prepare_conditional_data(X_norm, D_norm, y_norm)

            X_ite_norm = None
            if X_ite is not None and D_ite is not None:
                X_ite_with_d_norm = scaler_xd.transform(
                    np.column_stack([X_ite, D_ite])
                )
                X_ite_norm = X_ite_with_d_norm[:, :-1]

            print()
            print("Training conditional DDPM...")
            ddpm = train_conditional_ddpm(
                train_target=torch.tensor(y_target, dtype=torch.float32),
                train_cond=torch.tensor(cond, dtype=torch.float32),
                n_epochs=args.n_epochs,
                lr=args.lr,
                hidden_dims=args.hidden_dims,
                dim_t=args.dim_t,
                n_steps=args.n_steps,
                device=args.device,
                batch_size=args.batch_size,
                verbose=True,
            )

            mc_size_ate = max(1, int(args.total_mc_ate // n_size))
            total_mc_ate = mc_size_ate * n_size

            print()
            print(
                f"Estimating ATE with {mc_size_ate} Monte Carlo samples "
                f"(total {total_mc_ate})..."
            )
            _, ate_hat, mu0_hat, mu1_hat = estimate_ite_ddpm(
                ddpm=ddpm,
                X=X_norm.astype(np.float32),
                X_orig=X.astype(np.float32),
                scaler_y=scaler_y,
                scaler_xd=scaler_xd,
                x_dim=x_dim,
                monte_carlo_size=mc_size_ate,
                mc_batch_size=args.mc_batch_size,
                return_mu=True,
            )

            mu_out = np.column_stack(
                [
                    X,
                    D.reshape(-1, 1),
                    y.reshape(-1, 1),
                    mu1_hat.reshape(-1, 1),
                    mu0_hat.reshape(-1, 1),
                ]
            )
            mu_file = os.path.join(
                args.results_dir, f"seed_{seed}_n_{n_size}_mu.csv"
            )
            np.savetxt(mu_file, mu_out, delimiter=",")
            print(f"Saved potential outcomes to {mu_file}")

            ate_abs_error = (
                float(np.abs(ate_hat - ate_true)) if ate_true is not None else np.nan
            )

            ite_mse = np.nan
            ite_mae = np.nan
            tau_hat_ite = None
            if X_ite_norm is None or X_ite is None:
                warnings.warn("Missing ITE data; skipping ITE estimation.")
            elif tau_true is None:
                warnings.warn("Missing ITE truth; skipping ITE estimation.")
            else:
                print()
                print(
                    f"Estimating ITE with {args.monte_carlo_size_ite} Monte Carlo samples..."
                )
                tau_hat_ite, _ = estimate_ite_ddpm(
                    ddpm=ddpm,
                    X=X_ite_norm.astype(np.float32),
                    X_orig=X_ite.astype(np.float32),
                    scaler_y=scaler_y,
                    scaler_xd=scaler_xd,
                    x_dim=x_dim,
                    monte_carlo_size=args.monte_carlo_size_ite,
                    mc_batch_size=args.mc_batch_size,
                )

                if tau_true.shape[0] != tau_hat_ite.shape[0]:
                    warnings.warn(
                        "ITE length mismatch for seed "
                        f"{seed} n_size {n_size}: {tau_hat_ite.shape[0]} vs {tau_true.shape[0]}"
                    )
                else:
                    ite_mse = float(np.mean((tau_hat_ite - tau_true) ** 2))
                    ite_mae = float(np.mean(np.abs(tau_hat_ite - tau_true)))

                    ite_pred_file = os.path.join(
                        data_subdir, f"seed_{seed}_n_{n_size}_ite_predictions.csv"
                    )
                    ite_pred_out = np.column_stack([tau_hat_ite, tau_true])
                    np.savetxt(
                        ite_pred_file,
                        ite_pred_out,
                        delimiter=",",
                        header="tau_hat,tau_true",
                    )
                    print(f"Saved ITE predictions to {ite_pred_file}")

            with open(results_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([seed, n_size, ate_hat, ate_abs_error, ite_mse, ite_mae])

            print()
            print("=" * 50)
            print(f"RESULTS: Seed {seed}, n={n_size}")
            print("=" * 50)
            print(f"ATE estimated: {ate_hat:.6f}")
            if ate_true is not None:
                print(f"ATE true:      {ate_true:.6f}")
            if tau_true is not None:
                print(f"ITE MSE:       {ite_mse:.6f}")
                print(f"ITE MAE:       {ite_mae:.6f}")
            print("=" * 50)

        print()
        print("=" * 70)
        print(f"Saving results after completing seed {seed}...")
        print("=" * 70)

        if os.path.exists(results_file):
            results_df = pd.read_csv(results_file)
            seed_data = results_df[results_df["seed"] == seed]
            if len(seed_data) > 0:
                print()
                print(f"Seed {seed} summary:")
                print(seed_data[["n_size", "ate_hat", "ite_mse", "ite_mae"]].to_string(index=False))

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

        grouped = results_df.groupby("n_size")
        summary_rows = []
        for n_size, group in grouped:
            ate_hat_mean = group["ate_hat"].mean()
            ate_hat_std = group["ate_hat"].std()
            ate_true_vals = group["ate_true"].dropna()
            if len(ate_true_vals) > 0:
                ate_true_mean = ate_true_vals.mean()
                ate_err = (group["ate_hat"] - group["ate_true"]).abs().mean()
            else:
                ate_true_mean = np.nan
                ate_err = np.nan

            ite_mse_mean = group["ite_mse"].mean()
            ite_mae_mean = group["ite_mae"].mean()

            summary_rows.append(
                {
                    "n_size": n_size,
                    "ate_hat_mean": ate_hat_mean,
                    "ate_hat_std": ate_hat_std,
                    "ate_true_mean": ate_true_mean,
                    "ate_abs_error": ate_err,
                    "ite_mse_mean": ite_mse_mean,
                    "ite_mae_mean": ite_mae_mean,
                }
            )

        summary_df = pd.DataFrame(summary_rows)
        print()
        print("Summary by sample size:")
        print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
