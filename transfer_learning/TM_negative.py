#!/usr/bin/env python3
import argparse
import csv
import os
import random
import sys
import warnings

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import QuantileTransformer, StandardScaler

# Add parent directory to path to import transfer_learning utils
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from transfer_learning.utils.transfer_ddpm import (
    generate_transfer_samples,
    prepare_conditional_data,
    train_transfer_ddpm,
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


def parse_n1_list(value: str):
    return [int(v) for v in value.split(",") if v.strip()]


def parse_dim_phi_list(value: str):
    return [int(v) for v in value.split(",") if v.strip()]


def load_ate_true_map(data_dir: str):
    ate_true_file = os.path.join(data_dir, "ate_true.csv")
    if not os.path.exists(ate_true_file):
        return {}, {}

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


def estimate_potential_outcomes(
    ddpm,
    X_orig: np.ndarray,
    scaler_y,
    scaler_xd,
    monte_carlo_size: int,
    mc_batch_size: int,
):
    n = X_orig.shape[0]

    xd0_orig = np.column_stack([X_orig, np.zeros((n, 1))])
    xd1_orig = np.column_stack([X_orig, np.ones((n, 1))])

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

    ddpm.eval()
    with torch.no_grad():
        for batch_idx, batch_size in enumerate(batch_sizes):
            if batch_idx % 10 == 0 and len(batch_sizes) > 1:
                print(f"    MC batch {batch_idx+1}/{len(batch_sizes)}")

            samples0 = generate_transfer_samples(ddpm, cond0, n_samples=batch_size, domain="orig")
            samples1 = generate_transfer_samples(ddpm, cond1, n_samples=batch_size, domain="orig")

            if samples0.shape[0] != batch_size:
                raise ValueError(
                    f"Expected {batch_size} samples, got {samples0.shape[0]}"
                )
            if samples1.shape[0] != batch_size:
                raise ValueError(
                    f"Expected {batch_size} samples, got {samples1.shape[0]}"
                )

            samples0_np = samples0.detach().cpu().numpy().reshape(batch_size * n, 1)
            samples1_np = samples1.detach().cpu().numpy().reshape(batch_size * n, 1)

            y0 = scaler_y.inverse_transform(samples0_np).reshape(batch_size, n)
            y1 = scaler_y.inverse_transform(samples1_np).reshape(batch_size, n)

            sum0 += y0.sum(axis=0)
            sum1 += y1.sum(axis=0)
            total_samples += batch_size

    mu0 = sum0 / total_samples
    mu1 = sum1 / total_samples
    return mu0, mu1


def main():
    parser = argparse.ArgumentParser(
        description="Transfer-learning ATE estimation for TM_negative with conditional diffusion."
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--n_seeds", type=int, default=50)
    parser.add_argument("--n0", type=int, default=200, help="Original dataset size.")
    parser.add_argument(
        "--n1",
        type=parse_n1_list,
        default=parse_n1_list("0,200,500,1000"),
        help="Auxiliary dataset sizes (comma-separated). Use 0 for no auxiliary data.",
    )
    parser.add_argument("--x_dim", type=int, default=15)
    parser.add_argument(
        "--dim_phi",
        type=parse_dim_phi_list,
        default=parse_dim_phi_list("20"),
        help="Shared representation dimensions (comma-separated).",
    )
    parser.add_argument(
        "--shared_hidden_dims",
        type=parse_hidden_dims,
        default=parse_hidden_dims("512,256,256,256,128"),
    )
    parser.add_argument(
        "--head_hidden_dims",
        type=parse_hidden_dims,
        default=parse_hidden_dims("256,128"),
    )
    parser.add_argument("--dim_t", type=int, default=128)
    parser.add_argument("--n_steps", type=int, default=1000)
    parser.add_argument("--n_epochs", type=int, default=1500)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument(
        "--batch_size",
        type=int,
        default=0,
        help="Training batch size (0 for full-batch)",
    )
    parser.add_argument("--aux_batch_size", type=int, default=None)
    parser.add_argument("--pretrain_aux_epochs", type=int, default=1000)
    parser.add_argument("--pretrain_aux_lr", type=float, default=1e-5)
    parser.add_argument("--pretrain_aux_batch_size", type=int, default=None)
    parser.add_argument(
        "--use_pretrain",
        action="store_true",
        help="Enable auxiliary-only pretraining phase.",
    )
    parser.add_argument("--finetune_orig_epochs", type=int, default=100)
    parser.add_argument("--finetune_orig_lr", type=float, default=None)
    parser.add_argument("--finetune_orig_batch_size", type=int, default=None)
    parser.add_argument("--orig_weight", type=float, default=1.0)
    parser.add_argument(
        "--aux_weight",
        type=float,
        default=None,
        help="Aux loss weight; if unset, uses n0/n1.",
    )
    parser.add_argument("--monte_carlo_size", type=int, default=200)
    parser.add_argument("--mc_batch_size", type=int, default=200)
    parser.add_argument("--use_standard_scaler", action="store_true")

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
        default=os.path.join(acee_root, "transfer_learning", "results", "TM_negative"),
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
    print(f"Original size: {args.n0}")
    print(f"Auxiliary sizes: {args.n1}")
    print(f"X dimension: {args.x_dim}")
    print(f"dim_phi values: {args.dim_phi}")
    print(f"Shared hidden dims: {args.shared_hidden_dims}")
    print(f"Head hidden dims: {args.head_hidden_dims}")
    print(f"Dim t: {args.dim_t}")
    print(f"Diffusion steps: {args.n_steps}")
    print(f"Epochs: {args.n_epochs}")
    print(f"Learning rate: {args.lr}")
    print(
        f"Training batch size: {args.batch_size if args.batch_size > 0 else 'Full-batch'}"
    )
    print(f"Pretrain aux epochs: {args.pretrain_aux_epochs}")
    print(f"Pretrain aux lr: {args.pretrain_aux_lr}")
    print(f"Use pretrain: {args.use_pretrain}")
    print(f"Finetune orig epochs: {args.finetune_orig_epochs}")
    print(f"Finetune orig lr: {args.finetune_orig_lr if args.finetune_orig_lr is not None else args.lr}")
    print(f"Monte Carlo size: {args.monte_carlo_size}")
    print(f"MC batch size: {args.mc_batch_size}")
    print(f"Scaler: {'StandardScaler' if args.use_standard_scaler else 'QuantileTransformer'}")
    print(f"Data dir: {args.data_dir}")
    print(f"Results dir: {args.results_dir}")
    print("=" * 70)
    print()

    dim_phi_label = "_".join(str(v) for v in args.dim_phi)
    results_file = os.path.join(
        args.results_dir,
        f"ate_estimates_dimphi_{dim_phi_label}.csv",
    )
    if os.path.exists(results_file):
        os.remove(results_file)

    with open(results_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["seed", "n0", "n1", "dim_phi", "ate_hat", "ate_true", "ate_abs_error"])

    ate_true_map = load_ate_true_map(args.data_dir)

    for seed in range(args.n_seeds):
        print()
        print("=" * 70)
        print(f"SEED {seed}/{args.n_seeds - 1}")
        print("=" * 70)

        seed_everything(seed)
        ate_true = ate_true_map.get(seed)
        if ate_true is not None:
            print(f"True ATE (from data): {ate_true:.6f}")
        else:
            print("True ATE not found for this seed.")

        for dim_phi in args.dim_phi:
            print()
            print("=" * 50)
            print(f"SEED {seed} | dim_phi={dim_phi}")
            print("=" * 50)

            for n1 in args.n1:
                print()
                print("-" * 50)
                print(f"SEED {seed} | n0={args.n0}, n1={n1}, dim_phi={dim_phi}")
                print("-" * 50)

                data_subdir = os.path.join(args.data_dir, f"n0_{args.n0}_n1_{n1}")
                orig_file = os.path.join(data_subdir, f"seed_{seed}_orig.csv")
                aux_file = os.path.join(data_subdir, f"seed_{seed}_aux.csv")

                if not os.path.exists(orig_file):
                    warnings.warn(f"Missing original data file for seed={seed}, n1={n1}")
                    continue

                try:
                    data_orig = np.loadtxt(orig_file, delimiter=",")
                except Exception as exc:
                    warnings.warn(f"Failed to read original data for seed={seed}, n1={n1}: {exc}")
                    continue

                if data_orig.ndim == 1:
                    data_orig = data_orig.reshape(1, -1)

                expected_cols = args.x_dim + 2
                x_dim = args.x_dim
                if data_orig.shape[1] != expected_cols:
                    warnings.warn(
                        f"Expected {expected_cols} columns in {orig_file}, got {data_orig.shape[1]}"
                    )
                    x_dim = data_orig.shape[1] - 2

                aux_available = n1 > 0
                data_aux = None
                if aux_available:
                    if not os.path.exists(aux_file):
                        warnings.warn(f"Missing auxiliary data file for seed={seed}, n1={n1}")
                        continue
                    try:
                        data_aux = np.loadtxt(aux_file, delimiter=",")
                    except Exception as exc:
                        warnings.warn(f"Failed to read auxiliary data for seed={seed}, n1={n1}: {exc}")
                        continue

                    if data_aux.ndim == 1:
                        data_aux = data_aux.reshape(1, -1)

                    if data_aux.shape[1] != x_dim + 2:
                        warnings.warn(
                            f"Expected {x_dim + 2} columns in {aux_file}, got {data_aux.shape[1]}"
                        )
                        continue

                X_orig = data_orig[:, :x_dim]
                D_orig = data_orig[:, x_dim]
                y_orig = data_orig[:, x_dim + 1]

                print(f"Loaded original data: X shape {X_orig.shape}, y shape {y_orig.shape}")
                print(f"Original treatment balance: {D_orig.mean():.3f}")
                if aux_available:
                    X_aux = data_aux[:, :x_dim]
                    D_aux = data_aux[:, x_dim]
                    y_aux = data_aux[:, x_dim + 1]
                    print(f"Loaded auxiliary data: X shape {X_aux.shape}, y shape {y_aux.shape}")
                    print(f"Auxiliary treatment balance: {D_aux.mean():.3f}")
                    X_with_d_aux = np.column_stack([X_aux, D_aux])

                X_with_d_orig = np.column_stack([X_orig, D_orig])

                if args.use_standard_scaler:
                    scaler_xd = StandardScaler()
                    scaler_y = StandardScaler()
                else:
                    scaler_xd = QuantileTransformer(output_distribution="normal", random_state=seed)
                    scaler_y = QuantileTransformer(output_distribution="normal", random_state=seed)

                scaler_xd.fit(X_with_d_orig)
                scaler_y.fit(y_orig.reshape(-1, 1))

                X_with_d_orig_norm = scaler_xd.transform(X_with_d_orig)
                y_orig_norm = scaler_y.transform(y_orig.reshape(-1, 1)).reshape(-1)

                X_orig_norm = X_with_d_orig_norm[:, :-1]
                D_orig_norm = X_with_d_orig_norm[:, -1]

                y_target_orig, cond_orig = prepare_conditional_data(
                    X_orig_norm, D_orig_norm, y_orig_norm
                )

                aux_weight = args.aux_weight
                if aux_weight is None:
                    if n1 == 0:
                        aux_weight = 0.0
                    elif args.batch_size == 0:
                        aux_weight = 1.0
                    else:
                        aux_weight = float(args.n0) / float(n1)

                print(f"Aux weight: {aux_weight:.6f}")
                print()
                print("Training transfer DDPM...")
                print(f"\nTraining with dim_phi={dim_phi}...")
                y_target_aux = None
                cond_aux = None
                if aux_available:
                    X_with_d_aux_norm = scaler_xd.transform(X_with_d_aux)
                    y_aux_norm = scaler_y.transform(y_aux.reshape(-1, 1)).reshape(-1)
                    X_aux_norm = X_with_d_aux_norm[:, :-1]
                    D_aux_norm = X_with_d_aux_norm[:, -1]
                    y_target_aux, cond_aux = prepare_conditional_data(
                        X_aux_norm, D_aux_norm, y_aux_norm
                    )

                ddpm = train_transfer_ddpm(
                    orig_target=torch.tensor(y_target_orig, dtype=torch.float32),
                    orig_cond=torch.tensor(cond_orig, dtype=torch.float32),
                    aux_target=torch.tensor(y_target_aux, dtype=torch.float32) if aux_available else None,
                    aux_cond=torch.tensor(cond_aux, dtype=torch.float32) if aux_available else None,
                    n_epochs=args.n_epochs,
                    lr=args.lr,
                    shared_hidden_dims=args.shared_hidden_dims,
                    dim_phi=dim_phi,
                    head_hidden_dims_orig=args.head_hidden_dims,
                    head_hidden_dims_aux=args.head_hidden_dims,
                    dim_t=args.dim_t,
                    n_steps=args.n_steps,
                    device=args.device,
                    batch_size=args.batch_size,
                    aux_batch_size=args.aux_batch_size,
                    pretrain_aux_epochs=args.pretrain_aux_epochs if (args.use_pretrain and aux_available) else 0,
                    pretrain_aux_lr=args.pretrain_aux_lr,
                    pretrain_aux_batch_size=args.pretrain_aux_batch_size,
                    finetune_orig_epochs=args.finetune_orig_epochs,
                    finetune_orig_lr=args.finetune_orig_lr,
                    finetune_orig_batch_size=args.finetune_orig_batch_size,
                    orig_weight=args.orig_weight,
                    aux_weight=aux_weight,
                    verbose=True,
                    use_domain_heads=False,
                )

                print()
                print(f"Estimating potential outcomes with {args.monte_carlo_size} Monte Carlo samples...")
                mu0_hat, mu1_hat = estimate_potential_outcomes(
                    ddpm=ddpm,
                    X_orig=X_orig.astype(np.float32),
                    scaler_y=scaler_y,
                    scaler_xd=scaler_xd,
                    monte_carlo_size=args.monte_carlo_size,
                    mc_batch_size=args.mc_batch_size,
                )

                mu_out = np.column_stack(
                    [
                        X_orig,
                        D_orig.reshape(-1, 1),
                        y_orig.reshape(-1, 1),
                        mu1_hat.reshape(-1, 1),
                        mu0_hat.reshape(-1, 1),
                    ]
                )
                mu_file = os.path.join(
                    data_subdir,
                    f"seed_{seed}_orig_mu_dimphi_{dim_phi}.csv",
                )
                np.savetxt(mu_file, mu_out, delimiter=",")
                print(f"Saved potential outcomes to {mu_file}")

                ate_hat = float(np.mean(mu1_hat - mu0_hat))

                ate_abs_error = None
                if ate_true is not None:
                    ate_abs_error = abs(ate_hat - ate_true)

                with open(results_file, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(
                        [seed, args.n0, n1, dim_phi, ate_hat, ate_true, ate_abs_error]
                    )

                print()
                print("=" * 50)
                print(f"RESULTS: Seed {seed}, n0={args.n0}, n1={n1}, dim_phi={dim_phi}")
                print("=" * 50)
                print(f"ATE estimated: {ate_hat:.6f}")
                if ate_true is not None:
                    print(f"ATE true:      {ate_true:.6f}")
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
                print(seed_data[["n1", "dim_phi", "ate_hat"]].to_string(index=False))

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
            ate_hat_mean = group["ate_hat"].mean()
            ate_hat_std = group["ate_hat"].std()
            ate_true_vals = group["ate_true"].dropna()
            if len(ate_true_vals) > 0:
                ate_true_mean = ate_true_vals.mean()
                ate_err = (group["ate_hat"] - group["ate_true"]).abs().mean()
            else:
                ate_true_mean = np.nan
                ate_err = np.nan

            summary_rows.append(
                {
                    "n1": n1,
                    "dim_phi": dim_phi,
                    "ate_hat_mean": ate_hat_mean,
                    "ate_hat_std": ate_hat_std,
                    "ate_true_mean": ate_true_mean,
                    "ate_abs_error": ate_err,
                }
            )

        summary_df = pd.DataFrame(summary_rows)
        print()
        print("Summary by auxiliary size:")
        print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
