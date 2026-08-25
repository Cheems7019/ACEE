#!/usr/bin/env python3
import argparse
import csv
import json
import os

import numpy as np


def softplus(x):
    x = np.asarray(x)
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)


def nonlin_simpson(n_samples, rng, additive=True):
    """Generate nonlin_simpson observations with additive or non-additive noise."""
    x0 = rng.normal(0.0, 1.5, size=n_samples)

    if additive:
        x1_noise = rng.laplace(loc=0.0, scale=0.4, size=n_samples)
        x1 = 1.5 * (softplus(1.2 - x0) - 1.25) + x1_noise

        x2_noise = rng.laplace(loc=0.0, scale=1.0, size=n_samples)
        x2 = 1.8 * np.tanh(1.8 * x1) + 1.5 * np.sin(0.8 * x0) + 1.0 * np.tanh(x2_noise)

        x3_noise = rng.normal(0.0, 0.6, size=n_samples)
        x3 = 5.5 * np.tanh((x2 - 2.5) / 3) + 0.6 * np.sin(x2) + x3_noise
    else:
        x1_noise = rng.normal(0.0, 1.0, size=n_samples)
        x1 = 1.5 * (softplus(1.2 - x0 + 0.75 * np.sin(x1_noise)) - 1.25) * (
            1 + 0.25 * np.tanh(x1_noise)
        )

        x2_noise = rng.normal(0.0, 1.2, size=n_samples)
        x2 = (
            1.8 * np.tanh(1.8 * x1 + 0.4 * x2_noise)
            + 1.5 * np.sin(0.8 * x0)
        ) * (1 + 0.35 * np.tanh(x2_noise))

        x3_noise = rng.standard_t(df=5, size=n_samples)
        x3 = (
            5.5 * np.tanh((x2 + 0.5 * np.sin(x3_noise) - 2.5) / 3)
            + 0.6 * np.sin(x2)
        ) * (1 + 0.28 * np.tanh(x3_noise))

    return np.column_stack((x0, x1, x2, x3))

    
def nonlin_simpson_ate(
    n_samples,
    rng,
    intervention_idx,
    outcome_idx,
    intervention_low,
    intervention_high,
    additive=True,
):
    """Monte Carlo ATE for do(X_intervention=high) - do(X_intervention=low)."""
    if intervention_idx not in (0, 1, 2, 3) or outcome_idx not in (0, 1, 2, 3):
        raise ValueError("intervention_idx and outcome_idx must be between 0 and 3 for nonlin_simpson.")

    def sample_under_do(value):
        local_rng = np.random.default_rng(rng.integers(0, 2**32 - 1))

        x0 = (np.full(n_samples, value)
              if intervention_idx == 0
              else local_rng.normal(0.0, 1.5, size=n_samples))

        if additive:
            x1 = (np.full(n_samples, value)
                  if intervention_idx == 1
                  else 1.5 * (softplus(1.2 - x0) - 1.25) + local_rng.laplace(loc=0.0, scale=0.4, size=n_samples))

            x2 = (np.full(n_samples, value)
                  if intervention_idx == 2
                  else 1.8 * np.tanh(1.8 * x1) + 1.5 * np.sin(0.8 * x0) +
                  1.0 * np.tanh(local_rng.laplace(loc=0.0, scale=1.0, size=n_samples)))

            x3 = (np.full(n_samples, value)
                  if intervention_idx == 3
                  else 5.5 * np.tanh((x2 - 2.5) / 3) + 0.6 * np.sin(x2) +
                  local_rng.normal(0.0, 0.6, size=n_samples))
        else:
            x1_noise = local_rng.normal(0.0, 1.0, size=n_samples)
            x1 = (np.full(n_samples, value)
                  if intervention_idx == 1
                  else 1.5 * (softplus(1.2 - x0 + 0.75 * np.sin(x1_noise)) - 1.25) * (
                      1 + 0.25 * np.tanh(x1_noise)
                  ))

            x2_noise = local_rng.normal(0.0, 1.2, size=n_samples)
            x2 = (np.full(n_samples, value)
                  if intervention_idx == 2
                  else (
                      1.8 * np.tanh(1.8 * x1 + 0.4 * x2_noise)
                      + 1.5 * np.sin(0.8 * x0)
                  ) * (1 + 0.35 * np.tanh(x2_noise)))

            x3_noise = local_rng.standard_t(df=5, size=n_samples)
            x3 = (np.full(n_samples, value)
                  if intervention_idx == 3
                  else (
                      5.5 * np.tanh((x2 + 0.5 * np.sin(x3_noise) - 2.5) / 3)
                      + 0.6 * np.sin(x2)
                  ) * (1 + 0.28 * np.tanh(x3_noise)))

        return np.column_stack((x0, x1, x2, x3))[:, outcome_idx]

    low_vals = sample_under_do(intervention_low)
    high_vals = sample_under_do(intervention_high)
    return float(high_vals.mean() - low_vals.mean())


def symprod_simpson(n_samples, rng, additive=True):
    x0 = rng.normal(0.0, 1.5, size=n_samples)
    if additive:
        x1 = 2.5 * np.tanh(1.6 * x0) + 0.3 * rng.standard_t(df=3, size=n_samples)
        x2 = 0.6 * x0 * x1 + 0.5 * np.sin(x0) + rng.laplace(loc=0.0, scale=0.6, size=n_samples)
        x3 = 1.5 * np.tanh(1.4 * x0) + rng.normal(loc=0.0, scale=0.6, size=n_samples)
    else:
        x1_noise = rng.normal(0.0, 1.0, size=n_samples)
        x1 = (2.5 * np.tanh(1.6 * x0 + 0.4 * np.sin(x1_noise))) * (1 + 0.3 * np.tanh(x1_noise))

        x2_noise = rng.normal(0.0, 1.0, size=n_samples)
        x2 = (0.6 * x0 * x1 + 0.5 * np.sin(x0 + 0.3 * x2_noise)) * (
            1 + 0.25 * np.tanh(x2_noise)
        )

        x3_noise = rng.standard_t(df=4, size=n_samples)
        x3 = (1.5 * np.tanh(1.4 * x0 + 0.4 * np.sin(x3_noise))) * (1 + 0.25 * np.tanh(x3_noise))
    return np.column_stack((x0, x1, x2, x3))

def symprod_simpson_ate(
    n_samples,
    rng,
    intervention_idx,
    outcome_idx,
    intervention_low,
    intervention_high,
    additive=True,
):
    if intervention_idx not in (0, 1, 2, 3) or outcome_idx not in (0, 1, 2, 3):
        raise ValueError("intervention_idx and outcome_idx must be between 0 and 3 for symprod_simpson.")

    def sample_under_do(value):
        local_rng = np.random.default_rng(rng.integers(0, 2**32 - 1))

        x0 = (np.full(n_samples, value)
              if intervention_idx == 0
              else local_rng.normal(0.0, 1.5, size=n_samples))

        if additive:
            x1 = (np.full(n_samples, value)
                  if intervention_idx == 1
                  else 2.5 * np.tanh(1.6 * x0) + 0.3 * local_rng.standard_t(df=3, size=n_samples))

            x2 = (np.full(n_samples, value)
                  if intervention_idx == 2
                  else 0.6 * x0 * x1 + 0.5 * np.sin(x0) +
                  local_rng.laplace(loc=0.0, scale=0.6, size=n_samples))

            x3 = (np.full(n_samples, value)
                  if intervention_idx == 3
                  else 1.5 * np.tanh(1.4 * x0) +
                  local_rng.normal(loc=0.0, scale=0.6, size=n_samples))
        else:
            x1_noise = local_rng.normal(0.0, 1.0, size=n_samples)
            x1 = (np.full(n_samples, value)
                  if intervention_idx == 1
                  else (2.5 * np.tanh(1.6 * x0 + 0.4 * np.sin(x1_noise))) * (
                      1 + 0.3 * np.tanh(x1_noise)
                  ))

            x2_noise = local_rng.normal(0.0, 1.0, size=n_samples)
            x2 = (np.full(n_samples, value)
                  if intervention_idx == 2
                  else (0.6 * x0 * x1 + 0.5 * np.sin(x0 + 0.3 * x2_noise)) * (
                      1 + 0.25 * np.tanh(x2_noise)
                  ))

            x3_noise = local_rng.standard_t(df=4, size=n_samples)
            x3 = (np.full(n_samples, value)
                  if intervention_idx == 3
                  else (1.5 * np.tanh(1.4 * x0 + 0.4 * np.sin(x3_noise))) * (
                      1 + 0.25 * np.tanh(x3_noise)
                  ))

        return np.column_stack((x0, x1, x2, x3))[:, outcome_idx]

    low_vals = sample_under_do(intervention_low)
    high_vals = sample_under_do(intervention_high)
    return float(high_vals.mean() - low_vals.mean())


def sachs(n_samples, rng, sigma=1.0):
    x = np.zeros((n_samples, 11))
    x[:, 0] = rng.normal(0.0, sigma, size=n_samples)  # PKC
    x[:, 1] = rng.normal(0.0, sigma, size=n_samples)  # Plcg
    x[:, 2] = (
        0.5 * np.tanh(x[:, 0]**2)
        + 0.4 * x[:, 0]
        + 0.3 * np.sin(x[:, 0])**2
        + 0.2 * x[:, 0]**2
        + rng.normal(0.0, sigma, size=n_samples)
    )  # PKA
    x[:, 3] = (
        0.5 * x[:, 1]**2
        + 0.4 * np.sin(x[:, 1]) * np.cos(x[:, 1])
        + 0.3 * np.tanh(x[:, 1]**2)
        + 0.2 * x[:, 1]
        + rng.normal(0.0, sigma, size=n_samples)
    )  # PIP3
    x[:, 4] = (
        0.5 * np.tanh(x[:, 0]**2 + x[:, 2]**2)
        + 0.4 * x[:, 0] * np.sin(x[:, 2])
        + 0.3 * np.cos(x[:, 0]) * np.tanh(x[:, 2])
        + 0.2 * x[:, 0] * x[:, 2]
        + rng.normal(0.0, sigma, size=n_samples)
    )  # Raf
    x[:, 5] = (
        0.5 * np.tanh(x[:, 0]**2 + x[:, 2]**2)
        + 0.4 * np.sin(x[:, 0] * x[:, 2]) * np.cos(x[:, 2])
        + 0.3 * np.tanh(x[:, 0]**2)
        + 0.2 * x[:, 0] * x[:, 2]
        + rng.normal(0.0, sigma, size=n_samples)
    )  # Jnk
    x[:, 6] = (
        0.5 * np.tanh(x[:, 0]**2)
        + 0.4 * np.sin(x[:, 2]**2) * np.cos(x[:, 0])
        + 0.3 * np.tanh(x[:, 0] * x[:, 2]) * np.tanh(x[:, 2])
        + 0.2 * np.sin(x[:, 0])**2
        + rng.normal(0.0, sigma, size=n_samples)
    )  # P38
    x[:, 7] = (
        0.5 * np.tanh(x[:, 1]**2 + x[:, 3]**2)
        + 0.4 * np.sin(x[:, 1]) * np.tanh(x[:, 3])
        + 0.3 * np.cos(x[:, 3]) * np.tanh(x[:, 1])
        + 0.2 * x[:, 1] * x[:, 3]
        + rng.normal(0.0, sigma, size=n_samples)
    )  # PIP2
    x[:, 8] = (
        0.4 * np.tanh(x[:, 0]**2 + x[:, 4]**2)
        + 0.4 * np.tanh(x[:, 2] * x[:, 4])
        + 0.3 * np.sin(x[:, 0]) * np.tanh(x[:, 4] * x[:, 2])
        + 0.2 * x[:, 0] * x[:, 2]
        + rng.normal(0.0, sigma, size=n_samples)
    )  # Mek
    x[:, 9] = (
        0.5 * np.tanh(x[:, 8]**2 + x[:, 2]**2)
        + 0.4 * np.sin(x[:, 2])**2 * np.tanh(x[:, 8])
        + 0.3 * np.tanh(x[:, 8] * x[:, 2]) * np.cos(x[:, 2])
        + 0.2 * np.tanh(x[:, 8]) * x[:, 2]
        + rng.normal(0.0, sigma, size=n_samples)
    )  # Erk
    x[:, 10] = (
        0.5 * np.tanh(x[:, 9]**2 + x[:, 2]**2)
        + 0.4 * np.cos(x[:, 2])**2 * np.tanh(x[:, 9])
        + 0.3 * np.tanh(x[:, 9] * x[:, 2]) * np.sin(x[:, 2])**2
        + 0.2 * np.tanh(x[:, 9]) * np.cos(x[:, 2])
        + rng.normal(0.0, sigma, size=n_samples)
    )  # Akt
    return x


def sachs_ate(
    n_samples,
    rng,
    intervention_low,
    intervention_high,
    sigma=1.0,
):
    def sample_under_do(pka_value, mek_value):
        local_rng = np.random.default_rng(rng.integers(0, 2**32 - 1))
        pkc = local_rng.normal(0.0, sigma, size=n_samples)
        plcg = local_rng.normal(0.0, sigma, size=n_samples)
        pka = np.full(n_samples, pka_value)

        raf = (
            0.5 * np.tanh(pkc**2 + pka**2)
            + 0.4 * pkc * np.sin(pka)
            + 0.3 * np.cos(pkc) * np.tanh(pka)
            + 0.2 * pkc * pka
            + local_rng.normal(0.0, sigma, size=n_samples)
        )
        jnk = (
            0.5 * np.tanh(pkc**2 + pka**2)
            + 0.4 * np.sin(pkc * pka) * np.cos(pka)
            + 0.3 * np.tanh(pkc**2)
            + 0.2 * pkc * pka
            + local_rng.normal(0.0, sigma, size=n_samples)
        )
        p38 = (
            0.5 * np.tanh(pkc**2)
            + 0.4 * np.sin(pka**2) * np.cos(pkc)
            + 0.3 * np.tanh(pkc * pka) * np.tanh(pka)
            + 0.2 * np.sin(pkc)**2
            + local_rng.normal(0.0, sigma, size=n_samples)
        )

        mek = np.full(n_samples, mek_value)
        erk = (
            0.5 * np.tanh(mek**2 + pka**2)
            + 0.4 * np.sin(pka)**2 * np.tanh(mek)
            + 0.3 * np.tanh(mek * pka) * np.cos(pka)
            + 0.2 * np.tanh(mek) * pka
            + local_rng.normal(0.0, sigma, size=n_samples)
        )
        akt = (
            0.5 * np.tanh(erk**2 + pka**2)
            + 0.4 * np.cos(pka)**2 * np.tanh(erk)
            + 0.3 * np.tanh(erk * pka) * np.sin(pka)**2
            + 0.2 * np.tanh(erk) * np.cos(pka)
            + local_rng.normal(0.0, sigma, size=n_samples)
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

def adj_nonlin_simpson():
    adj = np.zeros((4, 4), dtype=int)
    adj[0, 1] = 1
    adj[0, 2] = 1
    adj[1, 2] = 1
    adj[2, 3] = 1
    return adj


def adj_symprod_simpson():
    adj = np.zeros((4, 4), dtype=int)
    adj[0, 1] = 1
    adj[1, 2] = 1
    adj[0, 2] = 1
    adj[0, 3] = 1
    return adj


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


SCENARIOS = {
    "nonlin_simpson": {
        "dataset": "csuite_nonlin_simpson",
        "generator": lambda n, rng: nonlin_simpson(n, rng, additive=True),
        "ate_fn": lambda n, rng: nonlin_simpson_ate(
            n,
            rng,
            intervention_idx=0,
            outcome_idx=2,
            intervention_low=-1,
            intervention_high=1,
            additive=True,
        ),
        "adj_fn": adj_nonlin_simpson,
        "num_nodes": 4,
        "dim": 1,
    },
    "nonlin_simpson_nonadditive": {
        "dataset": "csuite_nonlin_simpson_nonadditive",
        "generator": lambda n, rng: nonlin_simpson(n, rng, additive=False),
        "ate_fn": lambda n, rng: nonlin_simpson_ate(
            n,
            rng,
            intervention_idx=0,
            outcome_idx=2,
            intervention_low=-1,
            intervention_high=1,
            additive=False,
        ),
        "adj_fn": adj_nonlin_simpson,
        "num_nodes": 4,
        "dim": 1,
    },
    "symprod_simpson": {
        "dataset": "csuite_symprod_simpson",
        "generator": lambda n, rng: symprod_simpson(n, rng, additive=True),
        "ate_fn": lambda n, rng: symprod_simpson_ate(
            n,
            rng,
            intervention_idx=1,
            outcome_idx=2,
            intervention_low=-1,
            intervention_high=1,
            additive=True,
        ),
        "adj_fn": adj_symprod_simpson,
        "num_nodes": 4,
        "dim": 1,
    },
    "symprod_simpson_nonadditive": {
        "dataset": "csuite_symprod_simpson_nonadditive",
        "generator": lambda n, rng: symprod_simpson(n, rng, additive=False),
        "ate_fn": lambda n, rng: symprod_simpson_ate(
            n,
            rng,
            intervention_idx=1,
            outcome_idx=2,
            intervention_low=-1,
            intervention_high=1,
            additive=False,
        ),
        "adj_fn": adj_symprod_simpson,
        "num_nodes": 4,
        "dim": 1,
    },
    "sachs": {
        "dataset": "csuite_sachs",
        "generator": lambda n, rng: sachs(n, rng, sigma=1.0),
        "ate_fn": lambda n, rng: sachs_ate(
            n,
            rng,
            intervention_low=-1,
            intervention_high=1,
            sigma=1.0,
        ),
        "adj_fn": adj_sachs,
        "num_nodes": 11,
        "dim": 1,
    },
}


def parse_scenarios(value: str):
    items = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [item for item in items if item not in SCENARIOS]
    if unknown:
        raise argparse.ArgumentTypeError(
            f"Unknown scenarios: {', '.join(unknown)}. Available: {', '.join(SCENARIOS)}."
        )
    return items


def parse_n_samples(value: str):
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise argparse.ArgumentTypeError("n_samples must be a comma-separated list of integers.")
    try:
        sizes = [int(item) for item in items]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("n_samples must be a comma-separated list of integers.") from exc
    return sizes


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
    if isinstance(ate_value, dict):
        keys = list(ate_value.keys())
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(keys)
            writer.writerow([ate_value[key] for key in keys])
        return
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["n_seeds", "ate_true"])
        writer.writerow([n_seeds, ate_value])


def generate_scenario(name, cfg, n_samples_list, n_seeds, ate_mc_samples, data_root):
    dataset_dir = os.path.join(data_root, cfg["dataset"])
    os.makedirs(dataset_dir, exist_ok=True)

    all_train_data = []
    
    # Generate multiple training files with different seeds
    for n_samples in n_samples_list:
        size_dir = os.path.join(dataset_dir, f"n_{n_samples}")
        os.makedirs(size_dir, exist_ok=True)
        for seed in range(n_seeds):
            rng_train = np.random.default_rng(seed)
            train = cfg["generator"](n_samples, rng_train)
            
            train_path = os.path.join(size_dir, f"train_{seed}.csv")
            np.savetxt(train_path, train, delimiter=",")
            all_train_data.append(train)
            
            expected_cols = cfg["num_nodes"] * cfg["dim"]
            if train.shape[1] != expected_cols:
                raise ValueError(
                    f"{name}: expected {expected_cols} columns, got {train.shape[1]}"
                )
    
    # Combine all training data for computing overall variable ranges
    combined_data = np.vstack(all_train_data)

    # Write adjacency matrix (same for all seeds)
    adj = cfg["adj_fn"]()
    np.savetxt(os.path.join(dataset_dir, "adj_matrix.csv"), adj, delimiter=",", fmt="%i")

    # Write variables.json based on combined data
    write_variables(
        os.path.join(dataset_dir, "variables.json"),
        combined_data,
        cfg["num_nodes"],
        cfg["dim"],
    )

    # Compute single ATE value with large MC samples
    rng_ate = np.random.default_rng(1000)
    ate_value = cfg["ate_fn"](ate_mc_samples, rng_ate)
    write_ate(os.path.join(dataset_dir, "ate_true.csv"), n_seeds, ate_value)

    if isinstance(ate_value, dict):
        ate_summary = ", ".join(f"{key}={val:.6f}" for key, val in ate_value.items())
    else:
        ate_summary = f"{ate_value:.6f}"
    sizes_label = ", ".join(str(n) for n in n_samples_list)
    total_files = n_seeds * len(n_samples_list)
    print(
        f"[{name}] saved to {dataset_dir} "
        f"({total_files} training files; sizes={sizes_label}, ATE={ate_summary})"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Generate DAG simulation datasets for ACEE and comparison methods."
    )
    parser.add_argument(
        "--scenarios",
        type=parse_scenarios,
        default=parse_scenarios(
            "nonlin_simpson,nonlin_simpson_nonadditive,symprod_simpson,symprod_simpson_nonadditive,sachs"
        ),
        help="Comma-separated list of scenarios to generate.",
    )
    parser.add_argument(
        "--n_samples",
        type=parse_n_samples,
        default=parse_n_samples("500,1000"),
        help="Comma-separated list of training sample sizes.",
    )
    parser.add_argument("--n_seeds", type=int, default=20, help="Number of training files to generate with different seeds")
    parser.add_argument("--ate_mc_samples", type=int, default=1000000)
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

    for scenario in args.scenarios:
        cfg = SCENARIOS[scenario]
        generate_scenario(
            scenario,
            cfg,
            args.n_samples,
            args.n_seeds,
            args.ate_mc_samples,
            data_root,
        )


if __name__ == "__main__":
    main()
