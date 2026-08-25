#!/usr/bin/env python3
"""
Combined summary for potential-outcome scenarios M1-M4.

Outputs four CSV tables in results/:
  - summary_potential_outcomes_ATE_MAE.csv
  - summary_potential_outcomes_ATE_MAE_SE.csv
  - summary_potential_outcomes_ITE_MAE.csv
  - summary_potential_outcomes_ITE_MAE_SE.csv
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import warnings

DEFAULT_ITE_MODELS = ["M1", "M2", "M3", "M4"]

METHOD_LABELS = {
    "ganite": "GANITE",
    "cfrnet": "CFRNet",
    "tarnet": "TARNet",
    "dragonnet": "DragonNet",
    "causal_forest": "CausalForest",
    "mlp": "MLP",
    "cevae": "CEVAE",
}
VERBOSE = False


def normalize_model_list(values, default):
    if values is None:
        return list(default)
    items = []
    for value in values:
        for part in value.split(","):
            part = part.strip()
            if part:
                items.append(part)
    return items


def log_warning(message: str) -> None:
    if VERBOSE:
        print(message)


def compute_mae_stats(estimates: np.ndarray, true_values: np.ndarray):
    mask = np.isfinite(estimates) & np.isfinite(true_values)
    if not np.any(mask):
        return np.nan, np.nan, 0
    errors = estimates[mask] - true_values[mask]
    abs_errors = np.abs(errors)
    mae = float(np.mean(abs_errors))
    n = int(mask.sum())
    mae_se = float(np.std(abs_errors, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return mae, mae_se, n


def summarize_mean_stats(values: np.ndarray):
    mask = np.isfinite(values)
    if not np.any(mask):
        return np.nan, np.nan, 0
    vals = values[mask]
    n = int(vals.size)
    mean = float(np.mean(vals))
    se = float(np.std(vals, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return mean, se, n


def discover_competitor_files(competitor_dir: Path, model_name: str):
    files = sorted(competitor_dir.glob(f"{model_name}_*.csv"))
    results = []
    prefix = f"{model_name}_"
    for path in files:
        stem = path.stem
        if not stem.startswith(prefix):
            continue
        suffix = stem[len(prefix):]
        if not suffix:
            continue
        label = METHOD_LABELS.get(suffix, suffix)
        results.append((label, path))
    return results


def read_ate_results(path: Path):
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        log_warning(f"Warning: failed to read {path}: {exc}")
        return None

    required = {"seed", "n_size", "ate_hat"}
    if not required.issubset(df.columns):
        log_warning(f"Warning: {path} missing columns {sorted(required)}")
        return None

    cols = ["seed", "n_size", "ate_hat"]
    if "ite_mae" in df.columns:
        cols.append("ite_mae")

    df = df[cols].copy()
    df["seed"] = df["seed"].astype(int)
    df["n_size"] = df["n_size"].astype(int)
    return df


def read_em_bc_results(path: Path, scenario: str):
    if not path.exists():
        log_warning(f"Warning: missing EM_bc results file: {path}")
        return None
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        log_warning(f"Warning: failed to read {path}: {exc}")
        return None

    required = {"scenario", "seed", "n_size", "ate_hat_bc"}
    if not required.issubset(df.columns):
        log_warning(f"Warning: {path} missing columns {sorted(required)}")
        return None

    df = df[df["scenario"].astype(str).str.upper() == scenario.upper()].copy()
    if df.empty:
        log_warning(f"Warning: no EM_bc rows for scenario {scenario} in {path}")
        return None

    df = df.rename(columns={"ate_hat_bc": "ate_hat"})[["seed", "n_size", "ate_hat"]]
    df["seed"] = df["seed"].astype(int)
    df["n_size"] = df["n_size"].astype(int)
    return df


def make_ite_true_loader(data_dir: Path):
    cache = {}

    def load_ate_true(seed_value: int, n_size_value: int):
        key = (int(seed_value), int(n_size_value))
        if key in cache:
            return cache[key]
        if not data_dir.exists():
            cache[key] = np.nan
            return np.nan
        truth_file = data_dir / f"n_{key[1]}" / f"seed_{key[0]}_train_truth.csv"
        if not truth_file.exists():
            cache[key] = np.nan
            return np.nan
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    category=UserWarning,
                    message="Input line 1 contained no data*",
                )
                row = np.loadtxt(truth_file, delimiter=",", comments="#", max_rows=1)
            if np.ndim(row) == 0:
                value = float(row)
            else:
                value = float(np.ravel(row)[0])
        except Exception:
            value = np.nan
        cache[key] = value
        return value

    return load_ate_true


def collect_methods(model_name: str, results_root: Path, competitor_root: Path):
    method_frames = []

    ddpm_path = results_root / model_name / "ate_estimates.csv"
    if ddpm_path.exists():
        df_ddpm = read_ate_results(ddpm_path)
        if df_ddpm is not None:
            method_frames.append(("DDPM", df_ddpm))
    else:
        log_warning(f"Warning: missing DDPM results for {model_name}: {ddpm_path}")

    competitor_dir = competitor_root / model_name
    if competitor_dir.exists():
        for label, path in discover_competitor_files(competitor_dir, model_name):
            df_method = read_ate_results(path)
            if df_method is None:
                continue
            method_frames.append((label, df_method))
    else:
        log_warning(f"Warning: competitor directory not found for {model_name}: {competitor_dir}")

    return method_frames


def summarize_ite_model(model_name: str, data_root: Path, results_root: Path, competitor_root: Path):
    results = []

    data_dir = data_root / model_name
    load_ate_true = make_ite_true_loader(data_dir)

    method_frames = collect_methods(model_name, results_root, competitor_root)
    em_bc_path = results_root / "EM_bc" / "ate_estimates_bc.csv"
    df_em_bc = read_em_bc_results(em_bc_path, model_name)
    if df_em_bc is not None:
        method_frames.append(("DDPM-bc", df_em_bc))
    if not method_frames:
        log_warning(f"Warning: no methods found for {model_name}")
        return results

    for label, df_method in method_frames:
        df_method = df_method.copy()
        df_method["ate_true"] = df_method.apply(
            lambda row: load_ate_true(row["seed"], row["n_size"]), axis=1
        )

        for n_size in sorted(df_method["n_size"].dropna().unique()):
            df_size = df_method[df_method["n_size"] == n_size]

            ate_mae, ate_mae_se, ate_n = compute_mae_stats(
                df_size["ate_hat"].to_numpy(dtype=float),
                df_size["ate_true"].to_numpy(dtype=float),
            )

            if "ite_mae" in df_size.columns:
                ite_mae, ite_mae_se, ite_n = summarize_mean_stats(
                    df_size["ite_mae"].to_numpy(dtype=float)
                )
            else:
                ite_mae, ite_mae_se, ite_n = np.nan, np.nan, 0

            results.append(
                {
                    "Model": model_name,
                    "Method": label,
                    "n_size": int(n_size),
                    "ATE_MAE": ate_mae,
                    "ATE_MAE_SE": ate_mae_se,
                    "ATE_n": ate_n,
                    "ITE_MAE": ite_mae,
                    "ITE_MAE_SE": ite_mae_se,
                    "ITE_n": ite_n,
                }
            )

    return results


def write_pivot(df: pd.DataFrame, value_col: str, output_path: Path, label: str = None):
    pivot = df.pivot_table(index=["Model", "Method"], columns="n_size", values=value_col)
    pivot = pivot.reindex(sorted(pivot.columns), axis=1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pivot.to_csv(output_path)
    display = value_col if label is None else label
    print(f"Saved {display} table to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Summarize M1-M4 results.")
    parser.add_argument(
        "--ite-models",
        nargs="*",
        default=None,
        help="ITE models (space or comma separated). Default: M1 M2 M3 M4",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="data",
        help="Root directory for truth files.",
    )
    parser.add_argument(
        "--results_root",
        type=str,
        default="results",
        help="Root directory for ACEE/DDPM results.",
    )
    parser.add_argument(
        "--competitor_root",
        type=str,
        default="competitors/results",
        help="Root directory for competitor results.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Directory to save summary CSV tables.",
    )
    parser.add_argument(
        "--output_prefix",
        type=str,
        default="summary_potential_outcomes",
        help="Prefix for output CSV filenames.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print warnings about missing files/results.",
    )
    args = parser.parse_args()
    global VERBOSE
    VERBOSE = args.verbose

    acee_root = Path(__file__).resolve().parents[1]
    data_root = acee_root / args.data_root
    results_root = acee_root / args.results_root
    competitor_root = acee_root / args.competitor_root
    output_dir = acee_root / args.output_dir

    ite_models = normalize_model_list(args.ite_models, DEFAULT_ITE_MODELS)

    print("=" * 80)
    print("Combined Summary for Potential-Outcome Models")
    print("=" * 80)

    ite_rows = []
    for model_name in ite_models:
        ite_rows.extend(
            summarize_ite_model(model_name, data_root, results_root, competitor_root)
        )

    if ite_rows:
        df_ite = pd.DataFrame(ite_rows)
        output_path = output_dir / f"{args.output_prefix}_ATE_MAE.csv"
        write_pivot(df_ite, "ATE_MAE", output_path, label="ATE_MAE")
        output_path = output_dir / f"{args.output_prefix}_ATE_MAE_SE.csv"
        write_pivot(df_ite, "ATE_MAE_SE", output_path, label="ATE_MAE_SE")

        output_path = output_dir / f"{args.output_prefix}_ITE_MAE.csv"
        write_pivot(df_ite, "ITE_MAE", output_path, label="ITE_MAE")
        output_path = output_dir / f"{args.output_prefix}_ITE_MAE_SE.csv"
        write_pivot(df_ite, "ITE_MAE_SE", output_path, label="ITE_MAE_SE")
    else:
        log_warning("Warning: no ITE results found to summarize.")

    print("=" * 80)
    print("Analysis Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
