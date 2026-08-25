#!/usr/bin/env python3
import argparse
import csv
import json
import os

import numpy as np


def sachs(n_samples, rng, sigma=1.0):
    x = np.zeros((n_samples, 11))
    x[:, 0] = rng.normal(0.0, sigma, size=n_samples)  # PKC
    x[:, 1] = rng.normal(0.0, sigma, size=n_samples)  # Plcg
    x[:, 2] = (
        0.5 * np.tanh(x[:, 0] ** 2)
        + 0.4 * x[:, 0]
        + 0.3 * np.sin(x[:, 0]) ** 2
        + 0.2 * x[:, 0] ** 2
        + rng.normal(0.0, sigma, size=n_samples)
    )  # PKA
    x[:, 3] = (
        0.5 * x[:, 1] ** 2
        + 0.4 * np.sin(x[:, 1]) * np.cos(x[:, 1])
        + 0.3 * np.tanh(x[:, 1] ** 2)
        + 0.2 * x[:, 1]
        + rng.normal(0.0, sigma, size=n_samples)
    )  # PIP3
    x[:, 4] = (
        0.5 * np.tanh(x[:, 0] ** 2 + x[:, 2] ** 2)
        + 0.4 * x[:, 0] * np.sin(x[:, 2])
        + 0.3 * np.cos(x[:, 0]) * np.tanh(x[:, 2])
        + 0.2 * x[:, 0] * x[:, 2]
        + rng.normal(0.0, sigma, size=n_samples)
    )  # Raf
    x[:, 5] = (
        0.5 * np.tanh(x[:, 0] ** 2 + x[:, 2] ** 2)
        + 0.4 * np.sin(x[:, 0] * x[:, 2]) * np.cos(x[:, 2])
        + 0.3 * np.tanh(x[:, 0] ** 2)
        + 0.2 * x[:, 0] * x[:, 2]
        + rng.normal(0.0, sigma, size=n_samples)
    )  # Jnk
    x[:, 6] = (
        0.5 * np.tanh(x[:, 0] ** 2)
        + 0.4 * np.sin(x[:, 2] ** 2) * np.cos(x[:, 0])
        + 0.3 * np.tanh(x[:, 0] * x[:, 2]) * np.tanh(x[:, 2])
        + 0.2 * np.sin(x[:, 0]) ** 2
        + rng.normal(0.0, sigma, size=n_samples)
    )  # P38
    x[:, 7] = (
        0.5 * np.tanh(x[:, 1] ** 2 + x[:, 3] ** 2)
        + 0.4 * np.sin(x[:, 1]) * np.tanh(x[:, 3])
        + 0.3 * np.cos(x[:, 3]) * np.tanh(x[:, 1])
        + 0.2 * x[:, 1] * x[:, 3]
        + rng.normal(0.0, sigma, size=n_samples)
    )  # PIP2
    x[:, 8] = (
        0.4 * np.tanh(x[:, 0] ** 2 + x[:, 4] ** 2)
        + 0.4 * np.tanh(x[:, 2] * x[:, 4])
        + 0.3 * np.sin(x[:, 0]) * np.tanh(x[:, 4] * x[:, 2])
        + 0.2 * x[:, 0] * x[:, 2]
        + rng.normal(0.0, sigma, size=n_samples)
    )  # Mek
    x[:, 9] = (
        0.5 * np.tanh(x[:, 8] ** 2 + x[:, 2] ** 2)
        + 0.4 * np.sin(x[:, 2]) ** 2 * np.tanh(x[:, 8])
        + 0.3 * np.tanh(x[:, 8] * x[:, 2]) * np.cos(x[:, 2])
        + 0.2 * np.tanh(x[:, 8]) * x[:, 2]
        + rng.normal(0.0, sigma, size=n_samples)
    )  # Erk
    x[:, 10] = (
        0.5 * np.tanh(x[:, 9] ** 2 + x[:, 2] ** 2)
        + 0.4 * np.cos(x[:, 2]) ** 2 * np.tanh(x[:, 9])
        + 0.3 * np.tanh(x[:, 9] * x[:, 2]) * np.sin(x[:, 2]) ** 2
        + 0.2 * np.tanh(x[:, 9]) * np.cos(x[:, 2])
        + rng.normal(0.0, sigma, size=n_samples)
    )  # Akt
    return x


def sachs_ate(n_samples, rng, intervention_low, intervention_high, sigma=1.0):
    def sample_under_do(pka_value, mek_value):
        local_rng = np.random.default_rng(rng.integers(0, 2**32 - 1))
        pkc = local_rng.normal(0.0, sigma, size=n_samples)
        plcg = local_rng.normal(0.0, sigma, size=n_samples)
        pka = np.full(n_samples, pka_value)

        raf = (
            0.5 * np.tanh(pkc ** 2 + pka ** 2)
            + 0.4 * pkc * np.sin(pka)
            + 0.3 * np.cos(pkc) * np.tanh(pka)
            + 0.2 * pkc * pka
            + local_rng.normal(0.0, sigma, size=n_samples)
        )
        jnk = (
            0.5 * np.tanh(pkc ** 2 + pka ** 2)
            + 0.4 * np.sin(pkc * pka) * np.cos(pka)
            + 0.3 * np.tanh(pkc ** 2)
            + 0.2 * pkc * pka
            + local_rng.normal(0.0, sigma, size=n_samples)
        )
        p38 = (
            0.5 * np.tanh(pkc ** 2)
            + 0.4 * np.sin(pka ** 2) * np.cos(pkc)
            + 0.3 * np.tanh(pkc * pka) * np.tanh(pka)
            + 0.2 * np.sin(pkc) ** 2
            + local_rng.normal(0.0, sigma, size=n_samples)
        )

        mek = np.full(n_samples, mek_value)
        erk = (
            0.5 * np.tanh(mek ** 2 + pka ** 2)
            + 0.4 * np.sin(pka) ** 2 * np.tanh(mek)
            + 0.3 * np.tanh(mek * pka) * np.cos(pka)
            + 0.2 * np.tanh(mek) * pka
            + local_rng.normal(0.0, sigma, size=n_samples)
        )
        akt = (
            0.5 * np.tanh(erk ** 2 + pka ** 2)
            + 0.4 * np.cos(pka) ** 2 * np.tanh(erk)
            + 0.3 * np.tanh(erk * pka) * np.sin(pka) ** 2
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


def generate_sachs_dataset(n_samples_list, n_seeds, ate_mc_samples, data_root, sigma):
    dataset_dir = os.path.join(data_root, "csuite_sachs")
    os.makedirs(dataset_dir, exist_ok=True)

    all_train_data = []
    for n_samples in n_samples_list:
        size_dir = os.path.join(dataset_dir, f"n_{n_samples}")
        os.makedirs(size_dir, exist_ok=True)
        for seed in range(n_seeds):
            rng_train = np.random.default_rng(seed)
            train = sachs(n_samples, rng_train, sigma=sigma)
            train_path = os.path.join(size_dir, f"train_{seed}.csv")
            np.savetxt(train_path, train, delimiter=",")
            all_train_data.append(train)

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
    ate_value = sachs_ate(
        ate_mc_samples,
        rng_ate,
        intervention_low=-1,
        intervention_high=1,
        sigma=sigma,
    )
    write_ate(os.path.join(dataset_dir, "ate_true.csv"), n_seeds, ate_value)

    ate_summary = ", ".join(f"{key}={val:.6f}" for key, val in ate_value.items())
    sizes_label = ", ".join(str(n) for n in n_samples_list)
    total_files = n_seeds * len(n_samples_list)
    print(
        f"[sachs] saved to {dataset_dir} "
        f"({total_files} training files; sizes={sizes_label}, ATE={ate_summary})"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Generate csuite_sachs data for computation scalability."
    )
    parser.add_argument(
        "--n_samples",
        type=parse_n_samples,
        default=parse_n_samples("1000,2000,5000,10000"),
        help="Comma-separated list of training sample sizes.",
    )
    parser.add_argument("--n_seeds", type=int, default=5)
    parser.add_argument("--ate_mc_samples", type=int, default=100000)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="Base directory to write data; defaults to computation_scalability/data.",
    )
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_root = args.data_root or os.path.join(script_dir, "data")

    generate_sachs_dataset(
        n_samples_list=args.n_samples,
        n_seeds=args.n_seeds,
        ate_mc_samples=args.ate_mc_samples,
        data_root=data_root,
        sigma=args.sigma,
    )


if __name__ == "__main__":
    main()
