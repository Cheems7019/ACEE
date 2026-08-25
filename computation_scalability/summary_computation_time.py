#!/usr/bin/env python3
"""
Summarize computation time results for ACEE and competitors.

Outputs (in ACEE/computation_scalability):
  - total_time_M4_n.csv
  - total_time_M4_xdim.csv
  - total_time_M4_mc_total.csv
  - total_time_sachs.csv
  - ACEE_time_M4_n.csv
  - ACEE_time_M4_xdim.csv
  - ACEE_time_M4_mc_total.csv
  - ACEE_time_sachs.csv
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


M4_METHOD_MAP = {
    "cfrnet": "CFRNet",
    "ganite": "GANITE",
}

MC_TOTAL_BASELINE_FROM_N_SIZE = 100000
MC_TOTAL_BASELINE_N_SIZE = 1000

SACHS_METHOD_MAP = {
    "gaussian": "DECI Gaussian",
    "spline": "DECI Spline",
}


def trimmed_mean(values) -> float:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan")
    vals = np.sort(vals)
    if vals.size > 2:
        vals = vals[1:-1]
    return float(np.mean(vals))


def sort_with_method_order(df: pd.DataFrame, method_order, config_cols):
    df = df.copy()
    df["method"] = pd.Categorical(df["method"], categories=method_order, ordered=True)
    return df.sort_values(config_cols + ["method"]).reset_index(drop=True)


def summarize_acee_m4(results_dir: Path):
    total_rows_n = []
    total_rows_xdim = []
    total_rows_mc_total = []
    acee_rows_n = []
    acee_rows_xdim = []
    acee_rows_mc_total = []
    baseline_mc_candidates = []

    for path in sorted(results_dir.glob("M4_time_n*.csv")):
        df = pd.read_csv(path)
        n_size = int(df["n_size"].iloc[0])
        x_dim = int(df["x_dim"].iloc[0])

        total_rows_n.append(
            {
                "method": "ACEE",
                "n_size": n_size,
                "x_dim": x_dim,
                "total_time_s": trimmed_mean(df["total_time_s"]),
            }
        )
        acee_rows_n.append(
            {
                "method": "ACEE",
                "n_size": n_size,
                "x_dim": x_dim,
                "train_time_s": trimmed_mean(df["train_time_s"]),
                "sample_time_s": trimmed_mean(df["sample_time_s"]),
            }
        )
        if n_size == MC_TOTAL_BASELINE_N_SIZE:
            baseline_mc_candidates.append(
                {
                    "method": "ACEE",
                    "n_size": n_size,
                    "x_dim": x_dim,
                    "mc_total": MC_TOTAL_BASELINE_FROM_N_SIZE,
                    "total_time_s": trimmed_mean(df["total_time_s"]),
                    "train_time_s": trimmed_mean(df["train_time_s"]),
                    "sample_time_s": trimmed_mean(df["sample_time_s"]),
                }
            )

    for path in sorted(results_dir.glob("M4_time_xdim_*.csv")):
        df = pd.read_csv(path)
        n_size = int(df["n_size"].iloc[0])
        x_dim = int(df["x_dim"].iloc[0])

        total_rows_xdim.append(
            {
                "method": "ACEE",
                "n_size": n_size,
                "x_dim": x_dim,
                "total_time_s": trimmed_mean(df["total_time_s"]),
            }
        )
        acee_rows_xdim.append(
            {
                "method": "ACEE",
                "n_size": n_size,
                "x_dim": x_dim,
                "train_time_s": trimmed_mean(df["train_time_s"]),
                "sample_time_s": trimmed_mean(df["sample_time_s"]),
            }
        )

    mc_total_keys = set()
    for path in sorted(results_dir.glob("M4_time_mc_total_*.csv")):
        df = pd.read_csv(path)
        n_size = int(df["n_size"].iloc[0])
        x_dim = int(df["x_dim"].iloc[0])
        mc_total = int(df["mc_total"].iloc[0])
        mc_total_keys.add((n_size, x_dim, mc_total))

        total_rows_mc_total.append(
            {
                "method": "ACEE",
                "n_size": n_size,
                "x_dim": x_dim,
                "mc_total": mc_total,
                "total_time_s": trimmed_mean(df["total_time_s"]),
            }
        )
        acee_rows_mc_total.append(
            {
                "method": "ACEE",
                "n_size": n_size,
                "x_dim": x_dim,
                "mc_total": mc_total,
                "train_time_s": trimmed_mean(df["train_time_s"]),
                "sample_time_s": trimmed_mean(df["sample_time_s"]),
            }
        )

    for row in baseline_mc_candidates:
        key = (row["n_size"], row["x_dim"], row["mc_total"])
        if key in mc_total_keys:
            continue
        total_rows_mc_total.append(
            {
                "method": row["method"],
                "n_size": row["n_size"],
                "x_dim": row["x_dim"],
                "mc_total": row["mc_total"],
                "total_time_s": row["total_time_s"],
            }
        )
        acee_rows_mc_total.append(
            {
                "method": row["method"],
                "n_size": row["n_size"],
                "x_dim": row["x_dim"],
                "mc_total": row["mc_total"],
                "train_time_s": row["train_time_s"],
                "sample_time_s": row["sample_time_s"],
            }
        )

    return (
        total_rows_n,
        total_rows_xdim,
        total_rows_mc_total,
        acee_rows_n,
        acee_rows_xdim,
        acee_rows_mc_total,
    )


def summarize_competitor_m4(competitor_dir: Path):
    rows_n = []
    rows_xdim = []

    pattern = re.compile(r"^M4_time_(n\d+|xdim_\d+)_([a-z0-9_]+)\.csv$")
    for path in sorted(competitor_dir.glob("M4_time_*.csv")):
        match = pattern.match(path.name)
        if not match:
            continue
        config_tag, method_slug = match.groups()
        method = M4_METHOD_MAP.get(method_slug, method_slug)

        df = pd.read_csv(path)
        n_size = int(df["n_size"].iloc[0])
        x_dim = int(df["x_dim"].iloc[0])
        row = {
            "method": method,
            "n_size": n_size,
            "x_dim": x_dim,
            "total_time_s": trimmed_mean(df["total_time_s"]),
        }

        if config_tag.startswith("n"):
            rows_n.append(row)
        else:
            rows_xdim.append(row)

    return rows_n, rows_xdim


def summarize_acee_sachs(results_dir: Path):
    total_rows = []
    acee_rows = []

    for path in sorted(results_dir.glob("sachs_time_n*.csv")):
        df = pd.read_csv(path)
        n_samples = int(df["n_samples"].iloc[0])

        total_rows.append(
            {
                "method": "ACEE",
                "n_samples": n_samples,
                "total_time_s": trimmed_mean(df["total_time_s"]),
            }
        )
        acee_rows.append(
            {
                "method": "ACEE",
                "n_samples": n_samples,
                "train_time_s": trimmed_mean(df["train_time_s"]),
                "sample_time_s": trimmed_mean(df["sample_time_s"]),
            }
        )

    return total_rows, acee_rows


def summarize_competitor_sachs(competitor_dir: Path):
    rows = []
    pattern = re.compile(r"^sachs_time_n(\d+)_deci_([a-z0-9_]+)\.csv$")

    for path in sorted(competitor_dir.glob("sachs_time_n*_deci_*.csv")):
        match = pattern.match(path.name)
        if not match:
            continue
        n_samples = int(match.group(1))
        method_slug = match.group(2)
        method = SACHS_METHOD_MAP.get(method_slug, method_slug)

        df = pd.read_csv(path)
        rows.append(
            {
                "method": method,
                "n_samples": n_samples,
                "total_time_s": trimmed_mean(df["total_time_s"]),
            }
        )

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize computation time CSVs.")
    parser.add_argument(
        "--acee-results",
        type=Path,
        default=None,
        help="Path to ACEE/computation_scalability/results",
    )
    parser.add_argument(
        "--competitor-m4",
        type=Path,
        default=None,
        help="Path to ACEE/competitors/results/M4_time",
    )
    parser.add_argument(
        "--competitor-sachs",
        type=Path,
        default=None,
        help="Path to ACEE/competitors/results/sachs_time",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write summary CSVs",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    acee_root = script_dir.parent

    acee_results = (
        args.acee_results
        if args.acee_results is not None
        else script_dir / "results"
    )
    competitor_m4 = (
        args.competitor_m4
        if args.competitor_m4 is not None
        else acee_root / "competitors" / "results" / "M4_time"
    )
    competitor_sachs = (
        args.competitor_sachs
        if args.competitor_sachs is not None
        else acee_root / "competitors" / "results" / "sachs_time"
    )
    output_dir = args.output_dir if args.output_dir is not None else script_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    (
        total_rows_n,
        total_rows_xdim,
        total_rows_mc_total,
        acee_rows_n,
        acee_rows_xdim,
        acee_rows_mc_total,
    ) = summarize_acee_m4(acee_results)
    comp_rows_n, comp_rows_xdim = summarize_competitor_m4(competitor_m4)

    total_m4_n = pd.DataFrame(total_rows_n + comp_rows_n)
    total_m4_xdim = pd.DataFrame(total_rows_xdim + comp_rows_xdim)
    total_m4_mc_total = pd.DataFrame(total_rows_mc_total)

    total_m4_n = sort_with_method_order(
        total_m4_n, ["ACEE", "CFRNet", "GANITE"], ["n_size", "x_dim"]
    )
    total_m4_xdim = sort_with_method_order(
        total_m4_xdim, ["ACEE", "CFRNet", "GANITE"], ["x_dim", "n_size"]
    )
    if not total_m4_mc_total.empty:
        total_m4_mc_total = sort_with_method_order(
            total_m4_mc_total, ["ACEE"], ["mc_total", "n_size", "x_dim"]
        )
    else:
        total_m4_mc_total = pd.DataFrame(
            columns=["method", "n_size", "x_dim", "mc_total", "total_time_s"]
        )

    acee_m4_n = pd.DataFrame(acee_rows_n).sort_values(
        ["n_size", "x_dim"]
    ).reset_index(drop=True)
    acee_m4_xdim = pd.DataFrame(acee_rows_xdim).sort_values(
        ["x_dim", "n_size"]
    ).reset_index(drop=True)
    acee_m4_mc_total = pd.DataFrame(acee_rows_mc_total)
    if not acee_m4_mc_total.empty:
        acee_m4_mc_total = acee_m4_mc_total.sort_values(
            ["mc_total", "n_size", "x_dim"]
        ).reset_index(drop=True)
    else:
        acee_m4_mc_total = pd.DataFrame(
            columns=[
                "method",
                "n_size",
                "x_dim",
                "mc_total",
                "train_time_s",
                "sample_time_s",
            ]
        )

    total_sachs_rows, acee_sachs_rows = summarize_acee_sachs(acee_results)
    comp_sachs_rows = summarize_competitor_sachs(competitor_sachs)
    total_sachs = pd.DataFrame(total_sachs_rows + comp_sachs_rows)
    total_sachs = sort_with_method_order(
        total_sachs, ["ACEE", "DECI Gaussian", "DECI Spline"], ["n_samples"]
    )
    acee_sachs = pd.DataFrame(acee_sachs_rows).sort_values(
        ["n_samples"]
    ).reset_index(drop=True)

    total_m4_n.to_csv(
        output_dir / "total_time_M4_n.csv", index=False, float_format="%.3f"
    )
    total_m4_xdim.to_csv(
        output_dir / "total_time_M4_xdim.csv", index=False, float_format="%.3f"
    )
    total_m4_mc_total.to_csv(
        output_dir / "total_time_M4_mc_total.csv", index=False, float_format="%.3f"
    )
    total_sachs.to_csv(
        output_dir / "total_time_sachs.csv", index=False, float_format="%.3f"
    )
    acee_m4_n.to_csv(
        output_dir / "ACEE_time_M4_n.csv", index=False, float_format="%.3f"
    )
    acee_m4_xdim.to_csv(
        output_dir / "ACEE_time_M4_xdim.csv", index=False, float_format="%.3f"
    )
    acee_m4_mc_total.to_csv(
        output_dir / "ACEE_time_M4_mc_total.csv", index=False, float_format="%.3f"
    )
    acee_sachs.to_csv(
        output_dir / "ACEE_time_sachs.csv", index=False, float_format="%.3f"
    )

    print(f"Saved summaries to {output_dir}")


if __name__ == "__main__":
    main()
