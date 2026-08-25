#!/usr/bin/env python3
"""
Generate nonlinear Sachs data with unmeasured confounding.

Outputs (to data_root/dataset):
  - n_500/train_{seed}.csv
  - n_500/seed_{seed}_H.csv
  - adj_matrix.csv
  - variables.json
  - ate_true.csv (ATEs for standard Sachs interventions/outcomes)
"""

import argparse
import csv
import json
import os

import numpy as np


FIXED_N_SAMPLES = 500


def _sample_unmeasured(n_samples: int, rng: np.random.Generator, std: float):
    return rng.normal(0.0, std, size=n_samples)


def sachs_um(
    n_samples: int,
    rng: np.random.Generator,
    sigma: float = 1.0,
    sigma_unmeasured_confounding: float = 1.0,
):
    h = _sample_unmeasured(n_samples, rng, sigma_unmeasured_confounding)

    x = np.zeros((n_samples, 11))
    x[:, 0] = rng.normal(0.0, sigma, size=n_samples) + h  # PKC
    x[:, 1] = rng.normal(0.0, sigma, size=n_samples) + h  # Plcg
    x[:, 2] = (
        0.5 * np.tanh(x[:, 0] ** 2)
        + 0.4 * x[:, 0]
        + 0.3 * np.sin(x[:, 0]) ** 2
        + 0.2 * x[:, 0] ** 2
        + rng.normal(0.0, sigma, size=n_samples)
        + h
    )  # PKA
    x[:, 3] = (
        0.5 * x[:, 1] ** 2
        + 0.4 * np.sin(x[:, 1]) * np.cos(x[:, 1])
        + 0.3 * np.tanh(x[:, 1] ** 2)
        + 0.2 * x[:, 1]
        + rng.normal(0.0, sigma, size=n_samples)
        + h
    )  # PIP3
    x[:, 4] = (
        0.5 * np.tanh(x[:, 0] ** 2 + x[:, 2] ** 2)
        + 0.4 * x[:, 0] * np.sin(x[:, 2])
        + 0.3 * np.cos(x[:, 0]) * np.tanh(x[:, 2])
        + 0.2 * x[:, 0] * x[:, 2]
        + rng.normal(0.0, sigma, size=n_samples)
        + h
    )  # Raf
    x[:, 5] = (
        0.5 * np.tanh(x[:, 0] ** 2 + x[:, 2] ** 2)
        + 0.4 * np.sin(x[:, 0] * x[:, 2]) * np.cos(x[:, 2])
        + 0.3 * np.tanh(x[:, 0] ** 2)
        + 0.2 * x[:, 0] * x[:, 2]
        + rng.normal(0.0, sigma, size=n_samples)
        + h
    )  # Jnk
    x[:, 6] = (
        0.5 * np.tanh(x[:, 0] ** 2)
        + 0.4 * np.sin(x[:, 2] ** 2) * np.cos(x[:, 0])
        + 0.3 * np.tanh(x[:, 0] * x[:, 2]) * np.tanh(x[:, 2])
        + 0.2 * np.sin(x[:, 0]) ** 2
        + rng.normal(0.0, sigma, size=n_samples)
        + h
    )  # P38
    x[:, 7] = (
        0.5 * np.tanh(x[:, 1] ** 2 + x[:, 3] ** 2)
        + 0.4 * np.sin(x[:, 1]) * np.tanh(x[:, 3])
        + 0.3 * np.cos(x[:, 3]) * np.tanh(x[:, 1])
        + 0.2 * x[:, 1] * x[:, 3]
        + rng.normal(0.0, sigma, size=n_samples)
        + h
    )  # PIP2
    x[:, 8] = (
        0.4 * np.tanh(x[:, 0] ** 2 + x[:, 4] ** 2)
        + 0.4 * np.tanh(x[:, 2] * x[:, 4])
        + 0.3 * np.sin(x[:, 0]) * np.tanh(x[:, 4] * x[:, 2])
        + 0.2 * x[:, 0] * x[:, 2]
        + rng.normal(0.0, sigma, size=n_samples)
        + h
    )  # Mek
    x[:, 9] = (
        0.5 * np.tanh(x[:, 8] ** 2 + x[:, 2] ** 2)
        + 0.4 * np.sin(x[:, 2]) ** 2 * np.tanh(x[:, 8])
        + 0.3 * np.tanh(x[:, 8] * x[:, 2]) * np.cos(x[:, 2])
        + 0.2 * np.tanh(x[:, 8]) * x[:, 2]
        + rng.normal(0.0, sigma, size=n_samples)
        + h
    )  # Erk
    x[:, 10] = (
        0.5 * np.tanh(x[:, 9] ** 2 + x[:, 2] ** 2)
        + 0.4 * np.cos(x[:, 2]) ** 2 * np.tanh(x[:, 9])
        + 0.3 * np.tanh(x[:, 9] * x[:, 2]) * np.sin(x[:, 2]) ** 2
        + 0.2 * np.tanh(x[:, 9]) * np.cos(x[:, 2])
        + rng.normal(0.0, sigma, size=n_samples)
        + h
    )  # Akt
    return x, h


def sachs_um_ate(
    n_samples: int,
    rng: np.random.Generator,
    intervention_low: float,
    intervention_high: float,
    sigma: float = 1.0,
    sigma_unmeasured_confounding: float = 1.0,
):
    def sample_under_do(pka_value, mek_value):
        local_rng = np.random.default_rng(rng.integers(0, 2**32 - 1))
        h = _sample_unmeasured(n_samples, local_rng, sigma_unmeasured_confounding)

        pkc = local_rng.normal(0.0, sigma, size=n_samples) + h
        plcg = local_rng.normal(0.0, sigma, size=n_samples) + h
        pka = np.full(n_samples, pka_value)

        raf = (
            0.5 * np.tanh(pkc**2 + pka**2)
            + 0.4 * pkc * np.sin(pka)
            + 0.3 * np.cos(pkc) * np.tanh(pka)
            + 0.2 * pkc * pka
            + local_rng.normal(0.0, sigma, size=n_samples)
            + h
        )
        jnk = (
            0.5 * np.tanh(pkc**2 + pka**2)
            + 0.4 * np.sin(pkc * pka) * np.cos(pka)
            + 0.3 * np.tanh(pkc**2)
            + 0.2 * pkc * pka
            + local_rng.normal(0.0, sigma, size=n_samples)
            + h
        )
        p38 = (
            0.5 * np.tanh(pkc**2)
            + 0.4 * np.sin(pka**2) * np.cos(pkc)
            + 0.3 * np.tanh(pkc * pka) * np.tanh(pka)
            + 0.2 * np.sin(pkc) ** 2
            + local_rng.normal(0.0, sigma, size=n_samples)
            + h
        )

        mek = np.full(n_samples, mek_value)
        erk = (
            0.5 * np.tanh(mek**2 + pka**2)
            + 0.4 * np.sin(pka) ** 2 * np.tanh(mek)
            + 0.3 * np.tanh(mek * pka) * np.cos(pka)
            + 0.2 * np.tanh(mek) * pka
            + local_rng.normal(0.0, sigma, size=n_samples)
            + h
        )
        akt = (
            0.5 * np.tanh(erk**2 + pka**2)
            + 0.4 * np.cos(pka) ** 2 * np.tanh(erk)
            + 0.3 * np.tanh(erk * pka) * np.sin(pka) ** 2
            + 0.2 * np.tanh(erk) * np.cos(pka)
            + local_rng.normal(0.0, sigma, size=n_samples)
            + h
        )

        return {
            "raf": raf,
            "jnk": jnk,
            "p38": p38,
            "erk": erk,
            "akt": akt,
        }

    low_vals = sample_under_do(intervention_low, intervention_low)
    high_vals = sample_under_do(intervention_high, intervention_high)
    return {
        "ate_erk": float(high_vals["erk"].mean() - low_vals["erk"].mean()),
        "ate_akt": float(high_vals["akt"].mean() - low_vals["akt"].mean()),
        "ate_raf": float(high_vals["raf"].mean() - low_vals["raf"].mean()),
        "ate_jnk": float(high_vals["jnk"].mean() - low_vals["jnk"].mean()),
        "ate_p38": float(high_vals["p38"].mean() - low_vals["p38"].mean()),
    }


def adj_sachs():
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


def write_ate(path, ate_value):
    keys = list(ate_value.keys())
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(keys)
        writer.writerow([ate_value[key] for key in keys])


def generate_dataset(
    n_seeds: int,
    ate_mc_samples: int,
    sigma: float,
    sigma_unmeasured_confounding: float,
    data_root: str,
    dataset: str,
):
    dataset_dir = os.path.join(data_root, dataset)
    os.makedirs(dataset_dir, exist_ok=True)

    n_samples = FIXED_N_SAMPLES
    size_dir = os.path.join(dataset_dir, f"n_{n_samples}")
    os.makedirs(size_dir, exist_ok=True)

    all_train_data = []
    for seed in range(n_seeds):
        rng_train = np.random.default_rng(seed)
        train, h = sachs_um(
            n_samples,
            rng_train,
            sigma=sigma,
            sigma_unmeasured_confounding=sigma_unmeasured_confounding,
        )

        train_path = os.path.join(size_dir, f"train_{seed}.csv")
        np.savetxt(train_path, train, delimiter=",")
        all_train_data.append(train)

        h_path = os.path.join(size_dir, f"seed_{seed}_H.csv")
        np.savetxt(h_path, h.reshape(-1, 1), delimiter=",")

        expected_cols = 11
        if train.shape[1] != expected_cols:
            raise ValueError(
                f"sachs_um: expected {expected_cols} columns, got {train.shape[1]}"
            )

    combined_data = np.vstack(all_train_data)

    adj = adj_sachs()
    np.savetxt(os.path.join(dataset_dir, "adj_matrix.csv"), adj, delimiter=",", fmt="%i")

    write_variables(
        os.path.join(dataset_dir, "variables.json"),
        combined_data,
        num_nodes=11,
        dim=1,
    )

    rng_ate = np.random.default_rng(1000)
    ate_value = sachs_um_ate(
        ate_mc_samples,
        rng_ate,
        intervention_low=-1,
        intervention_high=1,
        sigma=sigma,
        sigma_unmeasured_confounding=sigma_unmeasured_confounding,
    )
    write_ate(os.path.join(dataset_dir, "ate_true.csv"), ate_value)

    ate_summary = ", ".join(f"{key}={val:.6f}" for key, val in ate_value.items())
    print(
        f"[sachs_um] saved to {dataset_dir} "
        f"({n_seeds} training files; size={n_samples}, ATE={ate_summary})"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Generate nonlinear Sachs-UM data with fixed n_samples=500."
    )
    parser.add_argument("--n_seeds", type=int, default=20)
    parser.add_argument("--ate_mc_samples", type=int, default=1000000)
    parser.add_argument("--sigma", type=float, default=1.0)
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
    parser.add_argument(
        "--dataset",
        type=str,
        default="sachs_um",
        help="Dataset directory name under data_root.",
    )
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    acee_root = os.path.dirname(script_dir)
    data_root = args.data_root or os.path.join(acee_root, "data")

    generate_dataset(
        n_seeds=args.n_seeds,
        ate_mc_samples=args.ate_mc_samples,
        sigma=args.sigma,
        sigma_unmeasured_confounding=args.sigma_unmeasured_confounding,
        data_root=data_root,
        dataset=args.dataset,
    )


if __name__ == "__main__":
    main()
