#!/usr/bin/env python3
"""
Generate nonlin_simpson data with unmeasured confounding.

Scenarios:
  - additive
  - nonadditive

For each scenario, outputs (to data_root/dataset):
  - n_500/train_{seed}.csv
  - n_500/seed_{seed}_H.csv
  - adj_matrix.csv
  - variables.json
  - ate_true.csv (single scalar ATE for do(x0=1)-do(x0=-1) on x2)
"""

import argparse
import csv
import json
import os

import numpy as np


FIXED_N_SAMPLES = 500

SCENARIOS = {
    "additive": {
        "dataset": "nonlin_simpson_um",
        "additive": True,
    },
    "nonadditive": {
        "dataset": "nonlin_simpson_nonadditive_um",
        "additive": False,
    },
}


def parse_scenarios(value: str):
    items = [item.strip().lower() for item in value.split(",") if item.strip()]
    unknown = [item for item in items if item not in SCENARIOS]
    if unknown:
        raise argparse.ArgumentTypeError(
            f"Unknown scenarios: {', '.join(unknown)}. Available: {', '.join(SCENARIOS)}."
        )
    return items


def softplus(x):
    x = np.asarray(x)
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)


def _sample_unmeasured(n_samples: int, rng: np.random.Generator, std: float):
    return rng.normal(0.0, std, size=n_samples)


def nonlin_simpson_um(
    n_samples: int,
    rng: np.random.Generator,
    sigma_unmeasured_confounding: float = 1.0,
    additive: bool = True,
):
    h = _sample_unmeasured(n_samples, rng, sigma_unmeasured_confounding)
    x0 = rng.normal(0.0, 1.5, size=n_samples) + h

    if additive:
        x1_noise = rng.laplace(loc=0.0, scale=0.4, size=n_samples)
        x1 = 1.5 * (softplus(1.2 - x0) - 1.25) + x1_noise + h

        x2_noise = rng.laplace(loc=0.0, scale=1.0, size=n_samples)
        x2 = 1.8 * np.tanh(1.8 * x1) + 1.5 * np.sin(0.8 * x0) + 1.0 * np.tanh(x2_noise) + h

        x3_noise = rng.normal(0.0, 0.6, size=n_samples)
        x3 = 5.5 * np.tanh((x2 - 2.5) / 3) + 0.6 * np.sin(x2) + x3_noise + h
    else:
        x1_noise = rng.normal(0.0, 1.0, size=n_samples)
        x1 = (
            1.5 * (softplus(1.2 - x0 + 0.75 * np.sin(x1_noise)) - 1.25)
            * (1 + 0.25 * np.tanh(x1_noise))
            + h
        )

        x2_noise = rng.normal(0.0, 1.2, size=n_samples)
        x2 = (
            (
                1.8 * np.tanh(1.8 * x1 + 0.4 * x2_noise)
                + 1.5 * np.sin(0.8 * x0)
            )
            * (1 + 0.35 * np.tanh(x2_noise))
            + h
        )

        x3_noise = rng.standard_t(df=5, size=n_samples)
        x3 = (
            (
                5.5 * np.tanh((x2 + 0.5 * np.sin(x3_noise) - 2.5) / 3)
                + 0.6 * np.sin(x2)
            )
            * (1 + 0.28 * np.tanh(x3_noise))
            + h
        )

    return np.column_stack((x0, x1, x2, x3)), h


def nonlin_simpson_um_ate(
    n_samples: int,
    rng: np.random.Generator,
    sigma_unmeasured_confounding: float = 1.0,
    additive: bool = True,
    intervention_idx: int = 0,
    outcome_idx: int = 2,
    intervention_low: float = -1.0,
    intervention_high: float = 1.0,
):
    if intervention_idx not in (0, 1, 2, 3) or outcome_idx not in (0, 1, 2, 3):
        raise ValueError("intervention_idx and outcome_idx must be between 0 and 3.")

    def sample_under_do(value):
        local_rng = np.random.default_rng(rng.integers(0, 2**32 - 1))
        h = _sample_unmeasured(n_samples, local_rng, sigma_unmeasured_confounding)

        x0 = (
            np.full(n_samples, value)
            if intervention_idx == 0
            else local_rng.normal(0.0, 1.5, size=n_samples) + h
        )

        if additive:
            x1 = (
                np.full(n_samples, value)
                if intervention_idx == 1
                else 1.5 * (softplus(1.2 - x0) - 1.25)
                + local_rng.laplace(loc=0.0, scale=0.4, size=n_samples)
                + h
            )

            x2 = (
                np.full(n_samples, value)
                if intervention_idx == 2
                else 1.8 * np.tanh(1.8 * x1)
                + 1.5 * np.sin(0.8 * x0)
                + 1.0 * np.tanh(local_rng.laplace(loc=0.0, scale=1.0, size=n_samples))
                + h
            )

            x3 = (
                np.full(n_samples, value)
                if intervention_idx == 3
                else 5.5 * np.tanh((x2 - 2.5) / 3)
                + 0.6 * np.sin(x2)
                + local_rng.normal(0.0, 0.6, size=n_samples)
                + h
            )
        else:
            x1_noise = local_rng.normal(0.0, 1.0, size=n_samples)
            x1 = (
                np.full(n_samples, value)
                if intervention_idx == 1
                else 1.5 * (softplus(1.2 - x0 + 0.75 * np.sin(x1_noise)) - 1.25)
                * (1 + 0.25 * np.tanh(x1_noise))
                + h
            )

            x2_noise = local_rng.normal(0.0, 1.2, size=n_samples)
            x2 = (
                np.full(n_samples, value)
                if intervention_idx == 2
                else (
                    1.8 * np.tanh(1.8 * x1 + 0.4 * x2_noise)
                    + 1.5 * np.sin(0.8 * x0)
                )
                * (1 + 0.35 * np.tanh(x2_noise))
                + h
            )

            x3_noise = local_rng.standard_t(df=5, size=n_samples)
            x3 = (
                np.full(n_samples, value)
                if intervention_idx == 3
                else (
                    5.5 * np.tanh((x2 + 0.5 * np.sin(x3_noise) - 2.5) / 3)
                    + 0.6 * np.sin(x2)
                )
                * (1 + 0.28 * np.tanh(x3_noise))
                + h
            )

        return np.column_stack((x0, x1, x2, x3))[:, outcome_idx]

    low_vals = sample_under_do(intervention_low)
    high_vals = sample_under_do(intervention_high)
    return float(high_vals.mean() - low_vals.mean())


def adj_nonlin_simpson():
    adj = np.zeros((4, 4), dtype=int)
    adj[0, 1] = 1
    adj[0, 2] = 1
    adj[1, 2] = 1
    adj[2, 3] = 1
    return adj


def build_variables(data, num_nodes, dim):
    variables = []
    col_idx = 0
    for node_idx in range(num_nodes):
        for dim_idx in range(dim):
            col = data[:, col_idx]
            variables.append(
                {
                    "always_observed": True,
                    "group_name": f"x{node_idx}",
                    "lower": float(np.min(col)),
                    "name": f"x{node_idx}_{dim_idx}",
                    "query": True,
                    "target": False,
                    "type": "continuous",
                    "upper": float(np.max(col)),
                }
            )
            col_idx += 1
    return {"metadata_variables": [], "variables": variables}


def write_variables(path, data, num_nodes, dim):
    variables = build_variables(data, num_nodes, dim)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(variables, f, indent=4)


def write_ate(path, n_seeds, ate_value):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["n_seeds", "ate_true"])
        writer.writerow([n_seeds, ate_value])


def generate_scenario(
    scenario_name: str,
    cfg: dict,
    n_seeds: int,
    ate_mc_samples: int,
    sigma_unmeasured_confounding: float,
    data_root: str,
):
    dataset_dir = os.path.join(data_root, cfg["dataset"])
    os.makedirs(dataset_dir, exist_ok=True)

    n_samples = FIXED_N_SAMPLES
    size_dir = os.path.join(dataset_dir, f"n_{n_samples}")
    os.makedirs(size_dir, exist_ok=True)

    all_train_data = []
    for seed in range(n_seeds):
        rng_train = np.random.default_rng(seed)
        train, h = nonlin_simpson_um(
            n_samples=n_samples,
            rng=rng_train,
            sigma_unmeasured_confounding=sigma_unmeasured_confounding,
            additive=cfg["additive"],
        )

        train_path = os.path.join(size_dir, f"train_{seed}.csv")
        np.savetxt(train_path, train, delimiter=",")
        all_train_data.append(train)

        h_path = os.path.join(size_dir, f"seed_{seed}_H.csv")
        np.savetxt(h_path, h.reshape(-1, 1), delimiter=",")

        if train.shape[1] != 4:
            raise ValueError(f"{scenario_name}: expected 4 columns, got {train.shape[1]}")

    combined_data = np.vstack(all_train_data)

    adj = adj_nonlin_simpson()
    np.savetxt(os.path.join(dataset_dir, "adj_matrix.csv"), adj, delimiter=",", fmt="%i")

    write_variables(
        os.path.join(dataset_dir, "variables.json"),
        combined_data,
        num_nodes=4,
        dim=1,
    )

    rng_ate = np.random.default_rng(1000)
    ate_value = nonlin_simpson_um_ate(
        n_samples=ate_mc_samples,
        rng=rng_ate,
        sigma_unmeasured_confounding=sigma_unmeasured_confounding,
        additive=cfg["additive"],
        intervention_idx=0,
        outcome_idx=2,
        intervention_low=-1.0,
        intervention_high=1.0,
    )
    write_ate(os.path.join(dataset_dir, "ate_true.csv"), n_seeds, ate_value)

    print(
        f"[{scenario_name}] saved to {dataset_dir} "
        f"({n_seeds} training files; size={n_samples}, ATE={ate_value:.6f})"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Generate nonlin_simpson_UM data (additive/nonadditive) with fixed n_samples=500."
    )
    parser.add_argument(
        "--scenarios",
        type=parse_scenarios,
        default=parse_scenarios("additive,nonadditive"),
        help="Comma-separated list: additive, nonadditive.",
    )
    parser.add_argument("--n_seeds", type=int, default=20)
    parser.add_argument("--ate_mc_samples", type=int, default=1000000)
    parser.add_argument(
        "--sigma_unmeasured_confounding",
        type=float,
        default=1.0,
        help="Std dev for H (Normal with mean 0 and this std).",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="Base directory to write data; defaults to ACEE/data.",
    )
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    acee_root = os.path.dirname(script_dir)
    data_root = args.data_root or os.path.join(acee_root, "data")

    for scenario_name in args.scenarios:
        cfg = SCENARIOS[scenario_name]
        generate_scenario(
            scenario_name=scenario_name,
            cfg=cfg,
            n_seeds=args.n_seeds,
            ate_mc_samples=args.ate_mc_samples,
            sigma_unmeasured_confounding=args.sigma_unmeasured_confounding,
            data_root=data_root,
        )


if __name__ == "__main__":
    main()
