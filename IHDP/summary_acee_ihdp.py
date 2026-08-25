#!/usr/bin/env python3
import argparse
import csv
import os
import warnings


def load_metrics(csv_path: str):
    if not os.path.exists(csv_path):
        warnings.warn(f"Missing results file: {csv_path}")
        return []
    rows = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        required = {"ate_abs_error", "ite_mae", "ite_mse"}
        if not required.issubset(reader.fieldnames or []):
            warnings.warn(f"Missing required columns in {csv_path}")
            return []
        for row in reader:
            rows.append(row)
    return rows


def to_float(value):
    if value is None or value == "":
        return None
    try:
        val = float(value)
    except ValueError:
        return None
    if val != val:
        return None
    return val


def summarize(rows):
    summary = {}
    items = list(rows)
    if not items:
        return summary

    key = "all"
    for key, items in {"all": items}.items():
        ate_mae_vals = []
        ite_mae_vals = []
        ite_mse_vals = []
        for row in items:
            ate_val = to_float(row.get("ate_abs_error"))
            ite_mae_val = to_float(row.get("ite_mae"))
            ite_mse_val = to_float(row.get("ite_mse"))
            if ate_val is not None:
                ate_mae_vals.append(ate_val)
            if ite_mae_val is not None:
                ite_mae_vals.append(ite_mae_val)
            if ite_mse_val is not None:
                ite_mse_vals.append(ite_mse_val)

        def mean_and_se(vals):
            if not vals:
                return float("nan"), float("nan")
            mean = sum(vals) / len(vals)
            if len(vals) < 2:
                return mean, float("nan")
            var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
            se = (var ** 0.5) / (len(vals) ** 0.5)
            return mean, se

        ate_mean, ate_se = mean_and_se(ate_mae_vals)
        ite_mae_mean, ite_mae_se = mean_and_se(ite_mae_vals)
        ite_mse_mean, ite_mse_se = mean_and_se(ite_mse_vals)

        summary[key] = {
            "mean_ate_mae": ate_mean,
            "se_ate_mae": ate_se,
            "mean_ite_mae": ite_mae_mean,
            "se_ite_mae": ite_mae_se,
            "mean_ite_mse": ite_mse_mean,
            "se_ite_mse": ite_mse_se,
            "n": len(items),
        }
    return summary


def print_summary(label: str, summary: dict):
    print(f"\n{label}")
    print("-" * len(label))
    keys = sorted(summary.keys(), key=lambda k: (k != "all", str(k)))
    for key in keys:
        stats = summary[key]
        print(
            f"n={stats['n']} | "
            f"ATE_MAE={stats['mean_ate_mae']:.6f} (SE={stats['se_ate_mae']:.6f}) | "
            f"ITE_MAE={stats['mean_ite_mae']:.6f} (SE={stats['se_ite_mae']:.6f}) | "
            f"ITE_MSE={stats['mean_ite_mse']:.6f} (SE={stats['se_ite_mse']:.6f})"
        )


def write_summary_csv(path: str, rows: list):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = [
        "method",
        "n",
        "ate_mae",
        "ate_mae_se",
        "ite_mae",
        "ite_mae_se",
        "ite_mse",
        "ite_mse_se",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    acee_root = os.path.dirname(script_dir)
    default_results_dir = os.path.join(script_dir, "results")

    parser = argparse.ArgumentParser(
        description="Summarize IHDP ACEE (adaptive) and competitor results."
    )
    parser.add_argument(
        "--acee_csv",
        type=str,
        default=os.path.join(default_results_dir, "ihdp_ate_estimates_adaptive.csv"),
        help="Path to ACEE adaptive results CSV.",
    )
    parser.add_argument(
        "--competitors_dir",
        type=str,
        default=os.path.join(acee_root, "competitors", "results", "IHDP"),
        help="Directory containing competitor CSVs.",
    )
    parser.add_argument(
        "--summary_csv",
        type=str,
        default=os.path.join(default_results_dir, "ihdp_summary.csv"),
        help="Path to write summary CSV (mean and SE).",
    )
    args = parser.parse_args()

    summary_rows = []

    acee_rows = load_metrics(args.acee_csv)
    if acee_rows:
        acee_summary = summarize(acee_rows)
        print_summary("ACEE", acee_summary)
        for key, stats in acee_summary.items():
            summary_rows.append(
                {
                    "method": "ACEE",
                    "n": stats["n"],
                    "ate_mae": stats["mean_ate_mae"],
                    "ate_mae_se": stats["se_ate_mae"],
                    "ite_mae": stats["mean_ite_mae"],
                    "ite_mae_se": stats["se_ite_mae"],
                    "ite_mse": stats["mean_ite_mse"],
                    "ite_mse_se": stats["se_ite_mse"],
                }
            )
    else:
        print("\nACEE\n----\nNo data.")

    competitor_files = [
        ("CFRNet", "IHDP_cfrnet.csv"),
        ("TARNet", "IHDP_tarnet.csv"),
        ("DragonNet", "IHDP_dragonnet.csv"),
        ("GANITE", "IHDP_ganite.csv"),
        ("CausalForest", "IHDP_causal_forest.csv"),
        ("CEVAE", "IHDP_cevae.csv"),
    ]
    for label, fname in competitor_files:
        path = os.path.join(args.competitors_dir, fname)
        rows = load_metrics(path)
        if rows:
            summary = summarize(rows)
            print_summary(label, summary)
            for key, stats in summary.items():
                summary_rows.append(
                    {
                        "method": label,
                        "n": stats["n"],
                        "ate_mae": stats["mean_ate_mae"],
                        "ate_mae_se": stats["se_ate_mae"],
                        "ite_mae": stats["mean_ite_mae"],
                        "ite_mae_se": stats["se_ite_mae"],
                        "ite_mse": stats["mean_ite_mse"],
                        "ite_mse_se": stats["se_ite_mse"],
                    }
                )
        else:
            print(f"\n{label}\n" + "-" * len(label) + "\nNo data.")

    if summary_rows:
        write_summary_csv(args.summary_csv, summary_rows)
        print(f"\nSaved summary CSV to {args.summary_csv}")


if __name__ == "__main__":
    main()
