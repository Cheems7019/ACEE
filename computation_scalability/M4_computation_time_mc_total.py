#!/usr/bin/env python3
import argparse
import csv
import os
import random
import sys
import time
import warnings

# Add parent directory to path to import utils
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
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


def parse_int_list(value: str):
    return [int(v) for v in value.split(",") if v.strip()]


def parse_hidden_dims(value: str):
    return [int(v) for v in value.split(",") if v.strip()]


def estimate_ate_ddpm(
    ddpm,
    X: np.ndarray,
    X_orig: np.ndarray,
    scaler_y,
    scaler_xd,
    x_dim: int,
    monte_carlo_size: int,
    mc_batch_size: int,
):
    n = X.shape[0]

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
        for batch_size in batch_sizes:
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

            samples0_np = samples0.detach().cpu().numpy().reshape(batch_size * n, 1)
            samples1_np = samples1.detach().cpu().numpy().reshape(batch_size * n, 1)

            y0 = scaler_y.inverse_transform(samples0_np).reshape(batch_size, n)
            y1 = scaler_y.inverse_transform(samples1_np).reshape(batch_size, n)

            sum0 += y0.sum(axis=0)
            sum1 += y1.sum(axis=0)
            total_samples += batch_size

    mean0 = sum0 / total_samples
    mean1 = sum1 / total_samples
    ate_hat = float(np.mean(mean1 - mean0))
    return ate_hat


def run_for_mc_total(mc_total: int, data_dir: str, args) -> None:
    os.makedirs(args.results_root, exist_ok=True)
    results_file = os.path.join(args.results_root, f"M4_time_mc_total_{mc_total}.csv")
    if os.path.exists(results_file):
        os.remove(results_file)

    with open(results_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "seed",
                "x_dim",
                "n_size",
                "mc_total",
                "total_time_s",
                "train_time_s",
                "sample_time_s",
            ]
        )

    n_size = args.n_size
    data_subdir = os.path.join(data_dir, f"n_{n_size}")

    for seed in range(args.n_seeds):
        print(f"\n{'='*70}")
        print(
            f"MC_TOTAL {mc_total} | N_SIZE {n_size} | "
            f"SEED {seed}/{args.n_seeds - 1}"
        )
        print(f"{'='*70}")

        seed_everything(seed)
        total_start = time.perf_counter()

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

        x_dim_used = args.x_dim
        expected_cols = args.x_dim + 2
        if data.shape[1] != expected_cols:
            warnings.warn(
                f"Expected {expected_cols} columns in {data_file}, got {data.shape[1]}"
            )
            x_dim_used = data.shape[1] - 2

        X = data[:, :x_dim_used]
        D = data[:, x_dim_used]
        y = data[:, x_dim_used + 1]

        X_with_d_for_norm = np.column_stack([X, D])

        if args.use_standard_scaler:
            scaler_xd = StandardScaler()
            scaler_y = StandardScaler()
        else:
            scaler_xd = QuantileTransformer(
                output_distribution="normal", random_state=seed
            )
            scaler_y = QuantileTransformer(
                output_distribution="normal", random_state=seed
            )

        scaler_xd.fit(X_with_d_for_norm)
        scaler_y.fit(y.reshape(-1, 1))

        X_with_d_norm = scaler_xd.transform(X_with_d_for_norm)
        y_norm = scaler_y.transform(y.reshape(-1, 1)).reshape(-1)

        X_norm = X_with_d_norm[:, :-1]
        D_norm = X_with_d_norm[:, -1]

        y_target, cond = prepare_conditional_data(X_norm, D_norm, y_norm)

        train_start = time.perf_counter()
        ddpm = train_conditional_ddpm(
            train_target=torch.tensor(y_target, dtype=torch.float32),
            train_cond=torch.tensor(cond, dtype=torch.float32),
            n_epochs=args.n_epochs,
            lr=args.lr,
            hidden_dims=args.hidden_dims,
            dim_t=args.dim_t,
            n_steps=args.n_steps,
            device=args.device,
            batch_size=0,
            verbose=True,
        )
        train_time = time.perf_counter() - train_start

        mc_samples = mc_total // n_size
        if mc_samples < 1:
            mc_samples = 1
        if mc_total % n_size != 0:
            print(
                f"Warning: mc_total {mc_total} not divisible by n_size {n_size}. "
                f"Using mc_samples={mc_samples}."
            )

        sample_start = time.perf_counter()
        ate_hat = estimate_ate_ddpm(
            ddpm=ddpm,
            X=X_norm.astype(np.float32),
            X_orig=X.astype(np.float32),
            scaler_y=scaler_y,
            scaler_xd=scaler_xd,
            x_dim=x_dim_used,
            monte_carlo_size=mc_samples,
            mc_batch_size=mc_samples,
        )
        sample_time = time.perf_counter() - sample_start

        total_time = time.perf_counter() - total_start

        with open(results_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    seed,
                    x_dim_used,
                    n_size,
                    mc_total,
                    total_time,
                    train_time,
                    sample_time,
                ]
            )

        print(
            f"x_dim={x_dim_used} | mc_total={mc_total} | total={total_time:.2f}s | "
            f"train={train_time:.2f}s | sample={sample_time:.2f}s"
        )

    print(f"\nSaved timing results to: {results_file}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Computation scalability for M4 (vary mc_total with fixed n_size)."
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--n_seeds", type=int, default=5)
    parser.add_argument("--x_dim", type=int, default=30)
    parser.add_argument("--n_size", type=int, default=1000)
    parser.add_argument(
        "--mc_totals",
        type=parse_int_list,
        default=parse_int_list("200000,300000,400000"),
        help="Comma-separated total n_size * mc_samples budgets.",
    )
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
    parser.add_argument("--use_standard_scaler", action="store_true")
    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="Base directory with generated data (defaults to computation_scalability/data).",
    )
    parser.add_argument(
        "--results_root",
        type=str,
        default=None,
        help="Directory to save timing CSVs (defaults to computation_scalability/results).",
    )
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    args.data_root = args.data_root or os.path.join(script_dir, "data")
    args.results_root = args.results_root or os.path.join(script_dir, "results")

    data_dir = os.path.join(args.data_root, f"M4_xdim_{args.x_dim}")
    if not os.path.isdir(data_dir):
        raise SystemExit(f"Data dir not found: {data_dir}")

    for mc_total in args.mc_totals:
        run_for_mc_total(mc_total, data_dir, args)


if __name__ == "__main__":
    main()
