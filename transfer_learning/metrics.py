#!/usr/bin/env python3
"""
Compute metrics from saved potential outcomes files.

Metrics:
1) Model residual MSE: mean((Y - mu_{D})^2)
2) kNN leave-one-out MSE on Y using X only, same D group
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


MU_FILE_RE = re.compile(r"seed_(\d+)_orig_mu_dimphi_(\d+)\.csv$")
N1_RE = re.compile(r"n1_(\d+)")


def parse_k_list(value: str):
    return [int(v) for v in value.split(",") if v.strip()]


def _infer_meta(path: Path):
    match = MU_FILE_RE.search(path.name)
    if not match:
        return None
    seed = int(match.group(1))
    dim_phi = int(match.group(2))
    if dim_phi != 20:
        return None
    n1_match = N1_RE.search(str(path.parent))
    if not n1_match:
        return None
    n1 = int(n1_match.group(1))
    return seed, dim_phi, n1


def _load_mu_file(path: Path):
    data = np.loadtxt(path, delimiter=",")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 5:
        raise ValueError(f"Unexpected column count {data.shape[1]} in {path}")
    x_dim = data.shape[1] - 4
    X = data[:, :x_dim]
    D = data[:, x_dim]
    y = data[:, x_dim + 1]
    mu1 = data[:, x_dim + 2]
    mu0 = data[:, x_dim + 3]
    return X, D, y, mu0, mu1


def _model_mse(y, D, mu0, mu1):
    mu_d = np.where(D > 0.5, mu1, mu0)
    return float(np.mean((y - mu_d) ** 2))


def _knn_mse_same_d(X, D, y, k_list):
    k_list = sorted(set(k_list))
    if not k_list:
        return {}
    max_k = max(k_list)
    sums = {k: 0.0 for k in k_list}
    counts = {k: 0 for k in k_list}

    for d_val in [0.0, 1.0]:
        idx = np.where(D == d_val)[0]
        if len(idx) <= 1:
            continue
        Xg = X[idx]
        yg = y[idx]

        diff = Xg[:, None, :] - Xg[None, :, :]
        dist = np.sum(diff ** 2, axis=2)
        np.fill_diagonal(dist, np.inf)
        order = np.argsort(dist, axis=1)

        for k in k_list:
            k_eff = min(k, len(idx) - 1)
            if k_eff <= 0:
                continue
            neigh = order[:, :k_eff]
            mu_hat = yg[neigh].mean(axis=1)
            err = (yg - mu_hat) ** 2
            sums[k] += float(err.sum())
            counts[k] += len(idx)

    results = {}
    for k in k_list:
        if counts[k] == 0:
            results[k] = (np.nan, 0)
        else:
            results[k] = (sums[k] / counts[k], counts[k])
    return results


def _aggregate_over_seeds(df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [c for c in df.columns if c.startswith("mse_") or c.startswith("knn_mse_")]
    grouped = df.groupby(["method", "n1", "dim_phi"], as_index=False)
    rows = []
    for _, group in grouped:
        row = {
            "method": group["method"].iloc[0],
            "n1": int(group["n1"].iloc[0]),
            "dim_phi": int(group["dim_phi"].iloc[0]),
            "n_seeds": int(group["seed"].nunique()),
        }
        for col in metric_cols:
            vals = pd.to_numeric(group[col], errors="coerce").dropna()
            if len(vals) == 0:
                row[f"{col}_mean"] = np.nan
                row[f"{col}_se"] = np.nan
            else:
                row[f"{col}_mean"] = float(vals.mean())
                row[f"{col}_se"] = float(vals.std(ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["method", "n1", "dim_phi"])


def main():
    parser = argparse.ArgumentParser(
        description="Compute transfer-learning metrics from mu files."
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="data",
        help="Root data directory (default: transfer_learning/data).",
    )
    parser.add_argument(
        "--methods",
        type=str,
        default="TM_positive,TM_negative",
        help="Comma-separated methods to include (TM_positive,TM_negative).",
    )
    parser.add_argument(
        "--k_list",
        type=parse_k_list,
        default=parse_k_list("5"),
        help="Comma-separated k values for kNN MSE.",
    )
    parser.add_argument(
        "--dim_phi",
        type=int,
        default=20,
        help="Filter to a single dim_phi value (default: 20).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="metrics_transfer_learning.csv",
        help="Output CSV filename.",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).parent.resolve()
    data_root = Path(args.data_root)
    if not data_root.is_absolute():
        data_root = script_dir / data_root
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = script_dir / output_path

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    rows = []

    for method in methods:
        method_dir = data_root / method
        if not method_dir.is_dir():
            print(f"Warning: missing data directory {method_dir}")
            continue
        for path in sorted(method_dir.rglob("seed_*_orig_mu_dimphi_*.csv")):
            meta = _infer_meta(path)
            if meta is None:
                continue
            seed, dim_phi, n1 = meta
            if dim_phi != args.dim_phi:
                continue
            try:
                X, D, y, mu0, mu1 = _load_mu_file(path)
            except Exception as exc:
                print(f"Warning: failed to read {path}: {exc}")
                continue

            mse_model = _model_mse(y, D, mu0, mu1)
            knn = _knn_mse_same_d(X, D, y, args.k_list)

            row = {
                "method": method,
                "seed": seed,
                "n1": n1,
                "dim_phi": dim_phi,
                "n": len(y),
                "mse_model": mse_model,
            }
            for k in args.k_list:
                mse_k, count_k = knn.get(k, (np.nan, 0))
                row[f"knn_mse_k{k}"] = mse_k
                row[f"knn_n_k{k}"] = count_k
            rows.append(row)

    if not rows:
        print("No valid mu files found.")
        return

    df = pd.DataFrame(rows)
    summary = _aggregate_over_seeds(df)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False)
    print(f"Saved metrics to: {output_path}")


if __name__ == "__main__":
    main()
