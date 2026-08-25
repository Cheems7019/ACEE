#!/usr/bin/env python3
import argparse
import os
import random
import sys

# Add parent directory to path to import utils
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from utils.utils_data import Sampler_M1, Sampler_M2, Sampler_M3, Sampler_M4


SCENARIOS = {
    "M1": {
        "sampler": Sampler_M1,
        "description": "M1",
        "n_sizes": [1000, 2000],
        "x_dim": 30,
        "sampler_hidden_dims": [30, 30],
        "sigma": 1.0,
    },
    "M2": {
        "sampler": Sampler_M2,
        "description": "M2 (heteroscedastic)",
        "n_sizes": [1000, 2000],
        "x_dim": 30,
        "sampler_hidden_dims": [30, 30],
        "sigma": 1.0,
    },
    "M3": {
        "sampler": Sampler_M3,
        "description": "M3",
        "n_sizes": [1000, 2000],
        "x_dim": 30,
        "sampler_hidden_dims": [30, 30],
        "sigma": 1.0,
    },
    "M4": {
        "sampler": Sampler_M4,
        "description": "M4",
        "n_sizes": [1000, 2000],
        "x_dim": 30,
        "sampler_hidden_dims": [30, 30],
        "sigma": 1.0,
    },
}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)


def parse_hidden_dims(value: str):
    return [int(v) for v in value.split(",") if v.strip()]


def parse_n_sizes(value: str):
    return [int(v) for v in value.split(",") if v.strip()]


def parse_scenarios(value: str):
    items = [item.strip().upper() for item in value.split(",") if item.strip()]
    unknown = [item for item in items if item not in SCENARIOS]
    if unknown:
        raise argparse.ArgumentTypeError(
            f"Unknown scenarios: {', '.join(unknown)}. Use M1,M2,M3,M4."
        )
    return items


def resolve_value(arg_value, default_value):
    return default_value if arg_value is None else arg_value


def generate_data_for_scenario(
    name: str,
    sampler_cls,
    description: str,
    n_sizes,
    n_size_ite: int,
    x_dim: int,
    sampler_hidden_dims,
    sigma: float,
    correlation: float,
    weight_scale: float,
    x_scale: float,
    n_seeds: int,
    ate_mc_samples: int,
    data_root: str,
):
    data_dir = os.path.join(data_root, name)
    os.makedirs(data_dir, exist_ok=True)

    for n_size in n_sizes:
        os.makedirs(os.path.join(data_dir, f"n_{n_size}"), exist_ok=True)

    print("\n" + "=" * 70)
    print(f"DATA GENERATION: {description}")
    print("=" * 70)
    print(f"Seeds: {n_seeds}")
    print(f"Training sizes: {n_sizes}")
    print(f"ITE size: {n_size_ite}")
    print(f"X dimension: {x_dim}")
    print(f"X scale (ITE): {x_scale}")
    print(f"Sigma: {sigma}")
    print(f"Correlation: {correlation}")
    print(f"Weight scale: {weight_scale}")
    print(f"Sampler hidden dims: {sampler_hidden_dims}")
    print(f"ATE MC samples: {ate_mc_samples}")
    print(f"Data dir: {data_dir}")
    print("=" * 70 + "\n")

    for seed in range(n_seeds):
        print(f"\n{'='*70}")
        print(f"SEED {seed}/{n_seeds - 1}")
        print(f"{'='*70}")

        seed_everything(seed)

        sampler = sampler_cls(
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

        for n_size in n_sizes:
            print(f"\n{'='*50}")
            print(f"SEED {seed} | Training Size: {n_size}")
            print(f"{'='*50}")

            try:
                print("\nGenerating training data...")
                X_with_d, y = sampler.sample(n=n_size, return_potential_outcomes=False)

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

                train_truth_file = os.path.join(
                    data_subdir, f"seed_{seed}_train_truth.csv"
                )
                np.savetxt(train_truth_file, np.array([ate_true]), delimiter=",")
                print(f"Saved training truth to {train_truth_file}")

                print("\nGenerating ITE data...")
                X_ite_with_d, y_ite, mu0_ite, mu1_ite = sampler.sample(
                    n=n_size_ite,
                    return_potential_outcomes=True,
                    x_scale=x_scale,
                )

                if X_ite_with_d.shape != (n_size_ite, x_dim + 1):
                    raise ValueError(
                        f"Expected X_with_d shape ({n_size_ite}, {x_dim + 1}), got {X_ite_with_d.shape}"
                    )
                if y_ite.shape != (n_size_ite,):
                    raise ValueError(
                        f"Expected y shape ({n_size_ite},), got {y_ite.shape}"
                    )

                X_ite = X_ite_with_d[:, :-1]
                D_ite = X_ite_with_d[:, -1]

                print(
                    f"Generated ITE data: X shape {X_ite.shape}, y shape {y_ite.shape}"
                )
                print(f"ITE treatment balance: {D_ite.mean():.3f}")
                print(f"ITE y range: [{y_ite.min():.3f}, {y_ite.max():.3f}]")

                ite_data_file = os.path.join(data_subdir, f"seed_{seed}_ite_data.csv")
                ite_out = np.concatenate([X_ite_with_d, y_ite.reshape(-1, 1)], axis=1)
                np.savetxt(ite_data_file, ite_out, delimiter=",")
                print(f"Saved ITE data to {ite_data_file}")

                tau_true = mu1_ite - mu0_ite
                truth_file = os.path.join(data_subdir, f"seed_{seed}_truth.csv")
                np.savetxt(
                    truth_file,
                    tau_true,
                    delimiter=",",
                    header="tau_true",
                )
                print(f"Saved ITE truth to {truth_file}")

            except Exception as exc:
                print(f"\nERROR processing seed={seed}, n_size={n_size}: {exc}")
                continue


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate data for manuscript scenarios M1-M4 with ITE truth files."
    )
    parser.add_argument(
        "--scenarios",
        type=parse_scenarios,
        default=parse_scenarios("M1,M2,M3,M4"),
        help="Comma-separated list of scenarios (M1,M2,M3,M4).",
    )
    parser.add_argument("--n_seeds", type=int, default=50)
    parser.add_argument("--n_sizes", type=parse_n_sizes, default=None)
    parser.add_argument("--n_size_ite", type=int, default=100)
    parser.add_argument("--x_dim", type=int, default=None)
    parser.add_argument("--sigma", type=float, default=None)
    parser.add_argument("--correlation", type=float, default=0.5)
    parser.add_argument(
        "--x_scale",
        type=float,
        default=0.2,
        help="Scale factor for X in the ITE dataset.",
    )
    parser.add_argument(
        "--weight_scale",
        type=float,
        default=0.5,
        help="Weight scale for neural networks in sampler",
    )
    parser.add_argument("--sampler_hidden_dims", type=parse_hidden_dims, default=None)
    parser.add_argument("--ate_mc_samples", type=int, default=100000)
    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="Base directory to write data; defaults to ACEE/data",
    )

    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    acee_root = os.path.dirname(script_dir)
    data_root = args.data_root or os.path.join(acee_root, "data")

    for scenario_name in args.scenarios:
        cfg = SCENARIOS[scenario_name]
        n_sizes = resolve_value(args.n_sizes, cfg["n_sizes"])
        x_dim = resolve_value(args.x_dim, cfg["x_dim"])
        sigma = resolve_value(args.sigma, cfg.get("sigma", 1.0))
        sampler_hidden_dims = resolve_value(
            args.sampler_hidden_dims, cfg["sampler_hidden_dims"]
        )

        generate_data_for_scenario(
            name=scenario_name,
            sampler_cls=cfg["sampler"],
            description=cfg["description"],
            n_sizes=n_sizes,
            n_size_ite=args.n_size_ite,
            x_dim=x_dim,
            sampler_hidden_dims=sampler_hidden_dims,
            sigma=sigma,
            correlation=args.correlation,
            weight_scale=args.weight_scale,
            x_scale=args.x_scale,
            n_seeds=args.n_seeds,
            ate_mc_samples=args.ate_mc_samples,
            data_root=data_root,
        )


if __name__ == "__main__":
    main()
