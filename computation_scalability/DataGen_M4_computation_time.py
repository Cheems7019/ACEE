#!/usr/bin/env python3
import argparse
import csv
import os
import random
import sys

# Add parent directory to path to import utils
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from utils.utils_data import Sampler_M4


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)


def parse_int_list(value: str):
    return [int(v) for v in value.split(",") if v.strip()]


def generate_data_for_x_dim(
    x_dim: int,
    n_sizes,
    sampler_hidden_dims,
    sigma: float,
    correlation: float,
    weight_scale: float,
    n_seeds: int,
    ate_mc_samples: int,
    data_root: str,
):
    scenario_name = f"M4_xdim_{x_dim}"
    data_dir = os.path.join(data_root, scenario_name)
    os.makedirs(data_dir, exist_ok=True)

    for n_size in n_sizes:
        os.makedirs(os.path.join(data_dir, f"n_{n_size}"), exist_ok=True)

    print("\n" + "=" * 70)
    print(f"DATA GENERATION: M4 (x_dim={x_dim})")
    print("=" * 70)
    print(f"Seeds: {n_seeds}")
    print(f"Training sizes: {n_sizes}")
    print(f"X dimension: {x_dim}")
    print(f"Sigma: {sigma}")
    print(f"Correlation: {correlation}")
    print(f"Weight scale: {weight_scale}")
    print(f"Sampler hidden dims: {sampler_hidden_dims}")
    print(f"ATE MC samples: {ate_mc_samples}")
    print(f"Data dir: {data_dir}")
    print("=" * 70 + "\n")

    ate_true_file = os.path.join(data_dir, "ate_true.csv")
    if os.path.exists(ate_true_file):
        os.remove(ate_true_file)

    with open(ate_true_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["seed", "ate_true"])

    for seed in range(n_seeds):
        print(f"\n{'='*70}")
        print(f"SEED {seed}/{n_seeds - 1}")
        print(f"{'='*70}")

        seed_everything(seed)

        sampler = Sampler_M4(
            sigma=sigma,
            correlation=correlation,
            x_dim=x_dim,
            seed=seed,
            hidden_dims=sampler_hidden_dims,
            weight_scale=weight_scale,
        )

        print(
            f"\nComputing true ATE for seed {seed}'s DGP with {ate_mc_samples} Monte Carlo samples..."
        )
        ate_true = sampler.estimate_ate(n_mc=ate_mc_samples)
        print(f"True ATE for this DGP: {ate_true:.6f}")

        with open(ate_true_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([seed, ate_true])

        for n_size in n_sizes:
            print(f"\n{'='*50}")
            print(f"SEED {seed} | Training Size: {n_size}")
            print(f"{'='*50}")

            try:
                print("\nGenerating training data...")
                X_with_d, y = sampler.sample(n=n_size)

                if X_with_d.shape != (n_size, x_dim + 1):
                    raise ValueError(
                        f"Expected X_with_d shape ({n_size}, {x_dim + 1}), got {X_with_d.shape}"
                    )
                if y.shape != (n_size,):
                    raise ValueError(f"Expected y shape ({n_size},), got {y.shape}")

                X = X_with_d[:, :-1]
                D = X_with_d[:, -1]

                print(f"Generated training data: X shape {X.shape}, y shape {y.shape}")
                print(f"Treatment balance: {D.mean():.3f}")
                print(f"y range: [{y.min():.3f}, {y.max():.3f}]")

                data_subdir = os.path.join(data_dir, f"n_{n_size}")
                data_file = os.path.join(data_subdir, f"seed_{seed}_data.csv")
                data_out = np.concatenate([X_with_d, y.reshape(-1, 1)], axis=1)
                np.savetxt(data_file, data_out, delimiter=",")
                print(f"Saved data to {data_file}")

            except Exception as exc:
                print(f"\nERROR processing seed={seed}, n_size={n_size}: {exc}")
                continue


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate data for M4 computation scalability experiments."
    )
    parser.add_argument(
        "--x_dims",
        type=parse_int_list,
        default=parse_int_list("30"),
        help="Comma-separated list of x_dim values.",
    )
    parser.add_argument(
        "--n_sizes",
        type=parse_int_list,
        default=parse_int_list("1000"),
        help="Comma-separated list of training sizes.",
    )
    parser.add_argument("--n_seeds", type=int, default=5)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--correlation", type=float, default=0.5)
    parser.add_argument(
        "--weight_scale",
        type=float,
        default=0.75,
        help="Weight scale for neural networks in sampler",
    )
    parser.add_argument(
        "--sampler_hidden_dims",
        type=parse_int_list,
        default=parse_int_list("50,50"),
    )
    parser.add_argument("--ate_mc_samples", type=int, default=100000)
    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="Base directory to write data; defaults to computation_scalability/data",
    )

    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_root = args.data_root or os.path.join(script_dir, "data")

    for x_dim in args.x_dims:
        generate_data_for_x_dim(
            x_dim=x_dim,
            n_sizes=args.n_sizes,
            sampler_hidden_dims=args.sampler_hidden_dims,
            sigma=args.sigma,
            correlation=args.correlation,
            weight_scale=args.weight_scale,
            n_seeds=args.n_seeds,
            ate_mc_samples=args.ate_mc_samples,
            data_root=data_root,
        )


if __name__ == "__main__":
    main()
