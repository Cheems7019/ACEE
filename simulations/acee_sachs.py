#!/usr/bin/env python3
import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import QuantileTransformer, StandardScaler


ACEE_ROOT = Path(__file__).resolve().parents[1]
if str(ACEE_ROOT) not in sys.path:
    sys.path.insert(0, str(ACEE_ROOT))

from utils.conditional_ddpm import train_all_conditional_ddpms, generate_conditional_samples


DATASETS = {
    "sachs": {
        "data_dir": "csuite_sachs",
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
    },
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


def get_parents(adj: np.ndarray, target_idx: int) -> list[int]:
    return [i for i in range(adj.shape[0]) if adj[i, target_idx] != 0]


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


def collect_training_sets(dataset_dir: Path):
    size_dirs = sorted(path for path in dataset_dir.glob("n_*") if path.is_dir())
    training_sets = []
    if size_dirs:
        for size_dir in size_dirs:
            try:
                n_samples = int(size_dir.name.split("_", 1)[1])
            except (IndexError, ValueError):
                print(f"Warning: Skipping directory with unexpected name: {size_dir.name}")
                continue
            train_files = sorted(size_dir.glob("train_*.csv"))
            if not train_files:
                print(f"Warning: No train_*.csv files found in {size_dir}")
                continue
            train_seeds = []
            for train_file in train_files:
                try:
                    seed_num = int(train_file.stem.split("_")[1])
                    train_seeds.append((seed_num, train_file))
                except (IndexError, ValueError):
                    print(f"Warning: Skipping file with unexpected name: {train_file.name}")
            train_seeds.sort()
            training_sets.append((n_samples, train_seeds))
        return training_sets

    train_files = sorted(dataset_dir.glob("train_*.csv"))
    if not train_files:
        return []
    train_seeds = []
    for train_file in train_files:
        try:
            seed_num = int(train_file.stem.split("_")[1])
            train_seeds.append((seed_num, train_file))
        except (IndexError, ValueError):
            print(f"Warning: Skipping file with unexpected name: {train_file.name}")
    train_seeds.sort()
    return [(None, train_seeds)]


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
        ate_values[outcome_idx] = float((high_mean[out_pos] - low_mean[out_pos]).mean())
    return ate_values


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Estimate ATEs for csuite_sachs using conditional diffusion."
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--monte_carlo_size",
        type=int,
        default=200,
        help="Number of outcome samples per training sample for each intervention",
    )
    parser.add_argument(
        "--monte_carlo_batch_size",
        type=int,
        default=200,
        help="Batch size for Monte Carlo sampling",
    )
    parser.add_argument("--n_epochs", type=int, default=2000, help="Training epochs per group")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--n_steps", type=int, default=1000, help="Diffusion steps")
    parser.add_argument(
        "--hidden_dims",
        type=str,
        default="512,256,256,256,128",
        help="Comma-separated hidden layer dimensions",
    )
    parser.add_argument(
        "--use_standard_scaler",
        action="store_true",
        help="Use StandardScaler instead of QuantileTransformer",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="sachs",
        help="Comma-separated dataset keys",
    )
    args = parser.parse_args()

    hidden_dims = [int(d.strip()) for d in args.hidden_dims.split(",")]
    datasets = [name.strip() for name in args.datasets.split(",") if name.strip()]

    acee_root = Path(__file__).resolve().parents[1]
    data_root = acee_root / "data"
    results_root = acee_root / "results"

    for dataset_key in datasets:
        if dataset_key not in DATASETS:
            raise ValueError(f"Unknown dataset key: {dataset_key}")
        cfg = DATASETS[dataset_key]

        dataset_dir = data_root / cfg["data_dir"]
        adj_path = dataset_dir / "adj_matrix.csv"

        training_sets = collect_training_sets(dataset_dir)
        if not training_sets:
            raise ValueError(f"No train_*.csv files found in {dataset_dir}")

        adj = np.loadtxt(adj_path, delimiter=",", dtype=float)
        A = torch.tensor(adj, dtype=torch.float32)

        descendants_interventions = set()
        for idx in cfg["intervention_idxs"]:
            descendants_interventions.update(find_descendants(adj, idx))

        outcome_groups = group_outcomes_by_observed(
            cfg, adj, descendants_interventions
        )

        result_dir = results_root / dataset_key
        result_dir.mkdir(parents=True, exist_ok=True)
        summary_path = result_dir / "acee_ate_all_seeds.csv"
        if summary_path.exists():
            summary_path.unlink()

        for n_samples, train_seeds in training_sets:
            for seed, train_path in train_seeds:
                seed_everything(seed)

                train_data = pd.read_csv(train_path, header=None).to_numpy(dtype=float)
                d_in = train_data.shape[1]
                groups = torch.arange(d_in, dtype=torch.long)

                scaler, train_data_norm = fit_scaler(
                    train_data, seed=seed, use_standard_scaler=args.use_standard_scaler
                )
                train_data_norm_tensor = torch.tensor(train_data_norm, dtype=torch.float32)

                models = train_all_conditional_ddpms(
                    train_data=train_data_norm_tensor,
                    groups=groups,
                    A=A,
                    n_epochs=args.n_epochs,
                    lr=args.lr,
                    hidden_dims=hidden_dims,
                    dim_t=128,
                    n_steps=args.n_steps,
                    device=args.device,
                    verbose=True,
                )

                ate_values_map = {}
                for observed_idxs, outcomes in outcome_groups.items():
                    observed_vars = {
                        idx: train_data_norm[:, idx].astype(np.float32)
                        for idx in observed_idxs
                    }
                    outcome_idxs = [outcome_idx for _, outcome_idx in outcomes]

                    group_ate_values = estimate_ate_joint_multi(
                        models=models,
                        groups=groups,
                        A=A,
                        scaler=scaler,
                        d_in=d_in,
                        n_samples_train=train_data.shape[0],
                        intervention_idxs=cfg["intervention_idxs"],
                        outcome_idxs=outcome_idxs,
                        x0_low=cfg["intervention_low"],
                        x0_high=cfg["intervention_high"],
                        mc_samples=args.monte_carlo_size,
                        mc_batch=args.monte_carlo_batch_size,
                        observed_vars=observed_vars,
                    )
                    ate_values_map.update(group_ate_values)

                current_n_samples = n_samples or train_data.shape[0]
                ate_row = {"seed": seed, "n_samples": current_n_samples}
                for outcome_name, outcome_idx in cfg["outcomes"]:
                    ate_value = ate_values_map[outcome_idx]
                    ate_row[f"ate_{outcome_name}"] = ate_value
                    print(
                        f"{dataset_key} n={current_n_samples} seed {seed}: "
                        f"{outcome_name} ATE={ate_value:.6f}"
                    )

                ate_df = pd.DataFrame([ate_row])
                ate_df.to_csv(
                    summary_path,
                    mode="a",
                    index=False,
                    header=not summary_path.exists(),
                )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
