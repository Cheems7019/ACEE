#!/usr/bin/env python3
import argparse
import csv
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import QuantileTransformer, StandardScaler


ACEE_ROOT = Path(__file__).resolve().parents[1]
if str(ACEE_ROOT) not in sys.path:
    sys.path.insert(0, str(ACEE_ROOT))

from utils.conditional_ddpm import train_all_conditional_ddpms, generate_conditional_samples


DATASET_CFG = {
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


def parse_int_list(value: str):
    return [int(v) for v in value.split(",") if v.strip()]


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
    for _, outcome_idx in cfg["outcomes"]:
        parents_outcome = set(get_parents(adj, outcome_idx))
        observed_idxs = tuple(
            sorted(
                idx
                for idx in parents_outcome
                if idx not in descendants_interventions
                and idx not in cfg["intervention_idxs"]
            )
        )
        grouped.setdefault(observed_idxs, []).append(outcome_idx)
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


def load_training_data(train_path: Path) -> np.ndarray:
    data = np.loadtxt(train_path, delimiter=",")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return data


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Computation scalability for csuite_sachs with conditional diffusion."
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--n_seeds", type=int, default=5)
    parser.add_argument(
        "--n_samples",
        type=parse_int_list,
        default=parse_int_list("1000,2000,5000,10000"),
        help="Comma-separated list of training sample sizes.",
    )
    parser.add_argument(
        "--mc_total",
        type=int,
        default=100000,
        help="Total n_samples * mc_samples budget (mc_samples = mc_total / n_samples).",
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

    hidden_dims = [int(d.strip()) for d in args.hidden_dims.split(",")]

    script_dir = Path(__file__).resolve().parent
    data_root = Path(args.data_root) if args.data_root else (script_dir / "data")
    results_root = Path(args.results_root) if args.results_root else (script_dir / "results")
    results_root.mkdir(parents=True, exist_ok=True)

    dataset_dir = data_root / DATASET_CFG["data_dir"]
    adj_path = dataset_dir / "adj_matrix.csv"
    if not adj_path.exists():
        raise SystemExit(f"Missing adj_matrix.csv: {adj_path}")

    adj = np.loadtxt(adj_path, delimiter=",", dtype=float)
    A = torch.tensor(adj, dtype=torch.float32)

    descendants_interventions = set()
    for idx in DATASET_CFG["intervention_idxs"]:
        descendants_interventions.update(find_descendants(adj, idx))

    outcome_groups = group_outcomes_by_observed(
        DATASET_CFG, adj, descendants_interventions
    )

    for n_samples in args.n_samples:
        mc_samples = args.mc_total // n_samples
        if mc_samples < 1:
            mc_samples = 1
        if args.mc_total % n_samples != 0:
            print(
                f"Warning: mc_total {args.mc_total} not divisible by n_samples {n_samples}. "
                f"Using mc_samples={mc_samples}."
            )

        results_path = results_root / f"sachs_time_n{n_samples}.csv"
        if results_path.exists():
            results_path.unlink()

        with open(results_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "seed",
                    "n_samples",
                    "total_time_s",
                    "train_time_s",
                    "sample_time_s",
                ]
            )

        size_dir = dataset_dir / f"n_{n_samples}"
        if not size_dir.exists():
            raise SystemExit(f"Missing data dir: {size_dir}")

        for seed in range(args.n_seeds):
            train_path = size_dir / f"train_{seed}.csv"
            if not train_path.exists():
                print(f"Warning: missing training file {train_path}")
                continue

            seed_everything(seed)

            total_start = time.perf_counter()
            train_data = load_training_data(train_path)
            d_in = train_data.shape[1]
            groups = torch.arange(d_in, dtype=torch.long)

            scaler, train_data_norm = fit_scaler(
                train_data, seed=seed, use_standard_scaler=args.use_standard_scaler
            )
            train_data_norm_tensor = torch.tensor(train_data_norm, dtype=torch.float32)

            train_start = time.perf_counter()
            # Full-batch training (conditional_ddpm uses all samples each epoch).
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
            train_time = time.perf_counter() - train_start

            sample_start = time.perf_counter()
            for observed_idxs, outcome_idxs in outcome_groups.items():
                observed_vars = {
                    idx: train_data_norm[:, idx].astype(np.float32)
                    for idx in observed_idxs
                }
                estimate_ate_joint_multi(
                    models=models,
                    groups=groups,
                    A=A,
                    scaler=scaler,
                    d_in=d_in,
                    n_samples_train=train_data.shape[0],
                    intervention_idxs=DATASET_CFG["intervention_idxs"],
                    outcome_idxs=outcome_idxs,
                    x0_low=DATASET_CFG["intervention_low"],
                    x0_high=DATASET_CFG["intervention_high"],
                    mc_samples=mc_samples,
                    mc_batch=mc_samples,
                    observed_vars=observed_vars,
                )
            sample_time = time.perf_counter() - sample_start

            total_time = time.perf_counter() - total_start

            with open(results_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        seed,
                        n_samples,
                        total_time,
                        train_time,
                        sample_time,
                    ]
                )

            print(
                f"n={n_samples} seed={seed} | total={total_time:.2f}s | "
                f"train={train_time:.2f}s | sample={sample_time:.2f}s"
            )

        print(f"\nSaved timing results to: {results_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
