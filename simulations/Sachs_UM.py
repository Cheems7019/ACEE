#!/usr/bin/env python3
import argparse
import csv
import os
import random
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import QuantileTransformer, StandardScaler

# Add parent directory to path to import utils
ACEE_ROOT = Path(__file__).resolve().parents[1]
if str(ACEE_ROOT) not in sys.path:
    sys.path.insert(0, str(ACEE_ROOT))

from utils.conditional_ddpm import (
    train_all_conditional_ddpms,
    generate_conditional_samples,
)


SACHS_CFG = {
    "intervention_idxs": [2, 8],  # PKA, Mek
    "intervention_low": -1.0,
    "intervention_high": 1.0,
    "outcomes": [
        ("erk", 9),
        ("akt", 10),
        ("raf", 4),
        ("jnk", 5),
        ("p38", 6),
    ],
}


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


def parse_n_sizes(value: str):
    return [int(v) for v in value.split(",") if v.strip()]


def parse_components(value: str):
    return [int(v) for v in value.split(",") if v.strip()]


def discover_seeds(data_dir: str, n_sizes):
    seeds = set()
    for n_size in n_sizes:
        subdir = os.path.join(data_dir, f"n_{n_size}")
        if not os.path.isdir(subdir):
            continue
        for fname in os.listdir(subdir):
            if not (fname.startswith("train_") and fname.endswith(".csv")):
                continue
            seed_str = fname[len("train_") : -len(".csv")]
            if seed_str.isdigit():
                seeds.add(int(seed_str))
    return sorted(seeds)


def load_data(data_file: str, x_dim_expected: int | None = None):
    data = np.loadtxt(data_file, delimiter=",")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 1:
        raise ValueError(f"Expected at least 1 column, got {data.shape[1]}")

    x_dim = data.shape[1]
    if x_dim_expected is not None and x_dim_expected != x_dim:
        warnings.warn(
            f"Expected x_dim={x_dim_expected}, but file has x_dim={x_dim}. Using file value."
        )
    return data


def compute_adaptive_k(X: np.ndarray) -> int:
    if X.ndim != 2:
        raise ValueError("Expected X to be 2D.")
    if X.shape[1] < 2:
        return 1

    _, s, _ = np.linalg.svd(X, full_matrices=False)
    if s.size < 2:
        return 1

    max_k = min(X.shape[1] - 1, s.size - 1)
    if max_k <= 0:
        return 1

    ratios = []
    for k in range(max_k):
        denom = s[k + 1]
        ratios.append(float("inf") if denom == 0 else s[k] / denom)
    return int(np.argmax(ratios)) + 1


def fit_scaler(train_data: np.ndarray, seed: int, use_standard_scaler: bool):
    if use_standard_scaler:
        scaler = StandardScaler()
        train_data_norm = scaler.fit_transform(train_data)
        return scaler, train_data_norm
    n_quantiles = min(1000, train_data.shape[0])
    qt = QuantileTransformer(
        output_distribution="normal",
        random_state=seed,
        n_quantiles=n_quantiles,
        subsample=min(100000, train_data.shape[0]),
    )
    train_data_norm = qt.fit_transform(train_data)
    return qt, train_data_norm


def scale_column_value(scaler, value: float, col_idx: int, d_in: int) -> float:
    dummy = np.zeros((1, d_in))
    dummy[0, col_idx] = value
    return float(scaler.transform(dummy)[0, col_idx])


def inverse_scale_column_values(scaler, values: np.ndarray, col_idx: int, d_in: int) -> np.ndarray:
    values = np.asarray(values).reshape(-1)
    dummy = np.zeros((values.shape[0], d_in))
    dummy[:, col_idx] = values
    return scaler.inverse_transform(dummy)[:, col_idx]


def find_ancestors(adj: np.ndarray, target_idx: int) -> list[int]:
    n = adj.shape[0]
    ancestors = set()
    stack = [target_idx]
    while stack:
        node = stack.pop()
        parents = [i for i in range(n) if adj[i, node] != 0]
        for parent in parents:
            if parent not in ancestors:
                ancestors.add(parent)
                stack.append(parent)
    return sorted(ancestors)


def find_descendants(adj: np.ndarray, source_idx: int) -> list[int]:
    n = adj.shape[0]
    descendants = set()
    stack = [source_idx]
    while stack:
        node = stack.pop()
        children = [i for i in range(n) if adj[node, i] != 0]
        for child in children:
            if child not in descendants:
                descendants.add(child)
                stack.append(child)
    return sorted(descendants)


def group_outcomes_by_observed(cfg: dict, adj: np.ndarray, descendants_interventions: set[int]):
    grouped = {}
    for outcome_name, outcome_idx in cfg["outcomes"]:
        ancestors_outcome = set(find_ancestors(adj, outcome_idx))
        observed_idxs = tuple(
            sorted(
                idx
                for idx in ancestors_outcome
                if idx not in descendants_interventions
                and idx not in cfg["intervention_idxs"]
            )
        )
        grouped.setdefault(observed_idxs, []).append((outcome_name, outcome_idx))
    return grouped


def estimate_ate_joint_multi(
    models,
    groups: torch.Tensor,
    A: torch.Tensor,
    scaler,
    d_in: int,
    n_samples_train: int,
    intervention_idxs: list[int],
    outcome_idxs: list[int],
    x0_low: float,
    x0_high: float,
    mc_samples: int,
    mc_batch: int,
    observed_vars: dict[int, np.ndarray],
) -> dict[int, float]:
    if mc_samples <= 0:
        mc_samples = 1
    if not outcome_idxs:
        return {}

    low_values = {}
    high_values = {}
    for idx in intervention_idxs:
        low_norm = scale_column_value(scaler, x0_low, idx, d_in)
        high_norm = scale_column_value(scaler, x0_high, idx, d_in)
        low_values[idx] = np.full(n_samples_train, low_norm, dtype=np.float32)
        high_values[idx] = np.full(n_samples_train, high_norm, dtype=np.float32)

    do_low = {**observed_vars, **low_values}
    do_high = {**observed_vars, **high_values}

    if mc_batch <= 0 or mc_batch > mc_samples:
        mc_batch = mc_samples
    batch_sizes = [mc_batch] * (mc_samples // mc_batch)
    if mc_samples % mc_batch != 0:
        batch_sizes.append(mc_samples % mc_batch)

    n_outcomes = len(outcome_idxs)
    low_sum = np.zeros((n_outcomes, n_samples_train), dtype=np.float64)
    high_sum = np.zeros((n_outcomes, n_samples_train), dtype=np.float64)

    for mc_batch_size in batch_sizes:
        samples_low_norm = generate_conditional_samples(
            models=models,
            groups=groups,
            A=A,
            do_vars=do_low,
            sample_vars=outcome_idxs,
            n_samples=mc_batch_size,
        )
        samples_high_norm = generate_conditional_samples(
            models=models,
            groups=groups,
            A=A,
            do_vars=do_high,
            sample_vars=outcome_idxs,
            n_samples=mc_batch_size,
        )

        low_norm_vals = samples_low_norm.detach().cpu().numpy().reshape(-1, n_outcomes)
        high_norm_vals = samples_high_norm.detach().cpu().numpy().reshape(-1, n_outcomes)

        for out_pos, outcome_idx in enumerate(outcome_idxs):
            low_vals = inverse_scale_column_values(
                scaler, low_norm_vals[:, out_pos], outcome_idx, d_in
            )
            high_vals = inverse_scale_column_values(
                scaler, high_norm_vals[:, out_pos], outcome_idx, d_in
            )

            low_sum[out_pos] += low_vals.reshape(mc_batch_size, n_samples_train).sum(axis=0)
            high_sum[out_pos] += high_vals.reshape(mc_batch_size, n_samples_train).sum(axis=0)

    low_mean = low_sum / mc_samples
    high_mean = high_sum / mc_samples
    ate_values = {}
    for out_pos, outcome_idx in enumerate(outcome_idxs):
        ate_values[outcome_idx] = float(high_mean[out_pos].mean() - low_mean[out_pos].mean())
    return ate_values


def build_sachs_adj():
    adj = np.zeros((11, 11), dtype=int)
    adj[0, 2] = 1  # PKC -> PKA
    adj[0, 4] = 1  # PKC -> Raf
    adj[0, 5] = 1  # PKC -> Jnk
    adj[0, 6] = 1  # PKC -> P38
    adj[0, 8] = 1  # PKC -> Mek
    adj[1, 3] = 1  # Plcg -> PIP3
    adj[1, 7] = 1  # Plcg -> PIP2
    adj[2, 4] = 1  # PKA -> Raf
    adj[2, 5] = 1  # PKA -> Jnk
    adj[2, 6] = 1  # PKA -> P38
    adj[2, 8] = 1  # PKA -> Mek
    adj[2, 9] = 1  # PKA -> Erk
    adj[2, 10] = 1  # PKA -> Akt
    adj[3, 7] = 1  # PIP3 -> PIP2
    adj[4, 8] = 1  # Raf -> Mek
    adj[8, 9] = 1  # Mek -> Erk
    adj[9, 10] = 1  # Erk -> Akt
    return adj


def build_factor_adj(base_adj: np.ndarray) -> np.ndarray:
    n = base_adj.shape[0]
    adj = np.zeros((n + 1, n + 1), dtype=int)
    adj[1:, 1:] = base_adj
    adj[0, 1:] = 1  # factor group -> all observed variables
    return adj


def load_factors(path: str, n_expected: int, z_dim_expected: int | None = None):
    Z = np.loadtxt(path, delimiter=",")
    if Z.ndim == 1:
        Z = Z.reshape(-1, 1)
    if Z.shape[0] != n_expected:
        raise ValueError(f"Expected Z rows {n_expected}, got {Z.shape[0]}")
    if z_dim_expected is not None and Z.shape[1] != z_dim_expected:
        raise ValueError(f"Expected Z dim {z_dim_expected}, got {Z.shape[1]}")
    return Z


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sachs_UM with conditional diffusion model using Z factors."
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--n_seeds", type=int, default=None)
    parser.add_argument(
        "--n_sizes",
        type=parse_n_sizes,
        default=parse_n_sizes("500"),
        help="Training sizes to process (only 500 is supported).",
    )
    parser.add_argument("--x_dim", type=int, default=None)

    parser.add_argument(
        "--pca_components",
        type=parse_components,
        default=parse_components("1,2,3"),
        help="Comma-separated PCA components to load for Z.",
    )
    parser.add_argument(
        "--pca_adaptive",
        action="store_true",
        default=True,
        help="Load adaptive PCA components based on singular value ratios.",
    )
    parser.add_argument(
        "--no_pca_adaptive",
        action="store_false",
        dest="pca_adaptive",
        help="Disable adaptive PCA.",
    )
    parser.add_argument(
        "--vae_components",
        type=parse_components,
        default=parse_components("1,2,3"),
        help="Comma-separated VAE latent dimensions to load for Z.",
    )

    parser.add_argument(
        "--hidden_dims",
        type=parse_components,
        default=parse_components("512,256,256,256,128"),
        help="Hidden dims for conditional DDPM",
    )
    parser.add_argument("--dim_t", type=int, default=128)
    parser.add_argument("--n_steps", type=int, default=1000)
    parser.add_argument("--n_epochs", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1024,
        help="Training batch size (0 for full-batch)",
    )
    parser.add_argument("--monte_carlo_size", type=int, default=400)
    parser.add_argument("--mc_batch_size", type=int, default=200)
    parser.add_argument("--use_standard_scaler", action="store_true")

    parser.add_argument(
        "--data_dir",
        type=str,
        default=str(ACEE_ROOT / "data" / "sachs_um"),
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default=str(ACEE_ROOT / "results" / "Sachs_UM"),
    )

    args = parser.parse_args()

    data_dir = args.data_dir
    results_dir = args.results_dir

    if not os.path.isdir(data_dir):
        raise SystemExit(f"Data dir not found: {data_dir}")

    n_sizes = sorted(set(args.n_sizes))
    if n_sizes != [500]:
        warnings.warn(
            f"Sachs_UM supports only n_500. Using [500] instead of {n_sizes}."
        )
        n_sizes = [500]

    if args.n_seeds is None:
        seeds = discover_seeds(data_dir, n_sizes)
    else:
        seeds = list(range(args.n_seeds))
    if not seeds:
        raise SystemExit("No seeds found to process.")

    os.makedirs(results_dir, exist_ok=True)

    print("\n" + "=" * 70)
    print("SACHS_UM EXPERIMENT")
    print("=" * 70)
    print(f"Data dir: {data_dir}")
    print(f"Results dir: {results_dir}")
    print(f"Seeds: {seeds if args.n_seeds is None else f'0..{args.n_seeds-1}'}")
    print(f"Training sizes: {n_sizes}")
    print(f"PCA components: {args.pca_components}")
    print(f"PCA adaptive: {args.pca_adaptive}")
    print(f"VAE components: {args.vae_components}")
    print(f"Hidden dims: {args.hidden_dims}")
    print(f"Dim t: {args.dim_t}")
    print(f"Diffusion steps: {args.n_steps}")
    print(f"Epochs: {args.n_epochs}")
    print(f"Learning rate: {args.lr}")
    print(
        f"Training batch size: {args.batch_size if args.batch_size > 0 else 'Full-batch'}"
    )
    print(f"Monte Carlo size: {args.monte_carlo_size}")
    print(f"MC batch size: {args.mc_batch_size}")
    print(f"Scaler: {'StandardScaler' if args.use_standard_scaler else 'QuantileTransformer'}")
    print("=" * 70 + "\n")

    results_file = os.path.join(results_dir, "ate_estimates.csv")
    if os.path.exists(results_file):
        os.remove(results_file)
    with open(results_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["seed", "n_size", "method", "outcome", "ate_hat", "k_adapt"])

    base_adj = build_sachs_adj()
    base_A = torch.tensor(base_adj, dtype=torch.float32)

    descendants_interventions = set()
    for idx in SACHS_CFG["intervention_idxs"]:
        descendants_interventions.update(find_descendants(base_adj, idx))

    outcome_groups_base = group_outcomes_by_observed(
        SACHS_CFG, base_adj, descendants_interventions
    )

    for seed in seeds:
        print(f"\n{'='*70}")
        print(f"SEED {seed}")
        print(f"{'='*70}")

        seed_everything(seed)

        for n_size in n_sizes:
            print(f"\n{'='*50}")
            print(f"SEED {seed} | Training Size: {n_size}")
            print(f"{'='*50}")

            data_subdir = os.path.join(data_dir, f"n_{n_size}")
            data_file = os.path.join(data_subdir, f"train_{seed}.csv")
            if not os.path.exists(data_file):
                warnings.warn(f"Missing data file: {data_file}")
                continue

            try:
                X_obs = load_data(data_file, x_dim_expected=args.x_dim)
            except Exception as exc:
                warnings.warn(f"Failed to read {data_file}: {exc}")
                continue

            n_samples_train = X_obs.shape[0]
            print(f"Loaded data: X shape {X_obs.shape}")

            method_specs = [("no_z", None, None)]
            for k in args.pca_components:
                if k <= 0:
                    warnings.warn(f"Skipping PCA with non-positive n_components={k}")
                    continue
                z_file = os.path.join(data_subdir, f"seed_{seed}_pca_k{k}_z.csv")
                method_specs.append((f"pca_k{k}", z_file, k))

            if args.pca_adaptive:
                try:
                    k_adapt = compute_adaptive_k(X_obs)
                    z_file = os.path.join(
                        data_subdir, f"seed_{seed}_pca_adaptive_k{k_adapt}_z.csv"
                    )
                    method_specs.append(("pca_adaptive", z_file, k_adapt))
                except Exception as exc:
                    warnings.warn(f"Failed to compute adaptive PCA k: {exc}")

            for k in args.vae_components:
                if k <= 0:
                    warnings.warn(f"Skipping VAE with non-positive latent_dim={k}")
                    continue
                z_file = os.path.join(data_subdir, f"seed_{seed}_vae_k{k}_z.csv")
                method_specs.append((f"vae_k{k}", z_file, k))

            for method, z_file, z_dim in method_specs:
                if z_file is None:
                    Z = None
                else:
                    if not os.path.exists(z_file):
                        warnings.warn(f"Missing {method} Z file: {z_file}")
                        continue
                    try:
                        Z = load_factors(z_file, n_expected=n_samples_train, z_dim_expected=z_dim)
                    except Exception as exc:
                        warnings.warn(f"Failed to read Z from {z_file}: {exc}")
                        continue

                if Z is None:
                    train_data = X_obs
                    groups = torch.arange(train_data.shape[1], dtype=torch.long)
                    A = base_A
                    var_offset = 0
                else:
                    train_data = np.column_stack([Z, X_obs])
                    groups = torch.zeros(train_data.shape[1], dtype=torch.long)
                    groups[Z.shape[1]:] = torch.arange(1, 1 + X_obs.shape[1], dtype=torch.long)
                    A = torch.tensor(build_factor_adj(base_adj), dtype=torch.float32)
                    var_offset = Z.shape[1]

                scaler, train_data_norm = fit_scaler(
                    train_data, seed=seed, use_standard_scaler=args.use_standard_scaler
                )
                train_data_norm_tensor = torch.tensor(train_data_norm, dtype=torch.float32)

                print()
                print(f"Training conditional DDPMs ({method})...")
                models = train_all_conditional_ddpms(
                    train_data=train_data_norm_tensor,
                    groups=groups,
                    A=A,
                    n_epochs=args.n_epochs,
                    lr=args.lr,
                    hidden_dims=args.hidden_dims,
                    dim_t=args.dim_t,
                    n_steps=args.n_steps,
                    device=args.device,
                    verbose=True,
                )

                ate_values_map = {}
                for observed_idxs_base, outcomes in outcome_groups_base.items():
                    observed_idxs = [idx + var_offset for idx in observed_idxs_base]
                    observed_vars = {
                        idx: train_data_norm[:, idx].astype(np.float32)
                        for idx in observed_idxs
                    }
                    if var_offset > 0:
                        for z_idx in range(var_offset):
                            observed_vars[z_idx] = train_data_norm[:, z_idx].astype(np.float32)

                    outcome_idxs = [outcome_idx + var_offset for _, outcome_idx in outcomes]

                    group_ate_values = estimate_ate_joint_multi(
                        models=models,
                        groups=groups,
                        A=A,
                        scaler=scaler,
                        d_in=train_data.shape[1],
                        n_samples_train=n_samples_train,
                        intervention_idxs=[
                            idx + var_offset for idx in SACHS_CFG["intervention_idxs"]
                        ],
                        outcome_idxs=outcome_idxs,
                        x0_low=SACHS_CFG["intervention_low"],
                        x0_high=SACHS_CFG["intervention_high"],
                        mc_samples=args.monte_carlo_size,
                        mc_batch=args.mc_batch_size,
                        observed_vars=observed_vars,
                    )
                    ate_values_map.update(group_ate_values)

                outcome_name_map = {
                    idx: name for name, idx in SACHS_CFG["outcomes"]
                }
                for outcome_idx, ate_hat in ate_values_map.items():
                    base_idx = outcome_idx - var_offset
                    outcome_name = outcome_name_map.get(base_idx, f"outcome_{base_idx}")
                    with open(results_file, "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow(
                            [
                                seed,
                                n_size,
                                method,
                                outcome_name,
                                ate_hat,
                                "" if method != "pca_adaptive" else z_dim,
                            ]
                        )


if __name__ == "__main__":
    main()
