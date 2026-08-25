#!/usr/bin/env python3
import argparse
import csv
import os
import random
import sys

import numpy as np
import torch
from sklearn.preprocessing import QuantileTransformer, StandardScaler

# Add parent directory to path to import utils
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.ddpm import (  # noqa: E402
    generate_conditional_samples,
    prepare_conditional_data,
    train_conditional_ddpm,
)


_TRAIN_FILE = "ihdp_npci_1-100.train.npz"
_TEST_FILE = "ihdp_npci_1-100.test.npz"


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


def parse_hidden_dims(value: str):
    return [int(v) for v in value.split(",") if v.strip()]


def ensure_file(data_dir: str, filename: str) -> str:
    file_path = os.path.join(data_dir, filename)
    if os.path.exists(file_path):
        return file_path
    raise FileNotFoundError(f"Missing IHDP file: {file_path}")


def load_npz(data_dir: str):
    train_path = ensure_file(data_dir, _TRAIN_FILE)
    test_path = ensure_file(data_dir, _TEST_FILE)
    train = np.load(train_path)
    test = np.load(test_path)
    return train, test


def extract_rep(split, rep_idx: int):
    x = split["x"][:, :, rep_idx].astype("float32")
    t = split["t"][:, rep_idx].astype("float32")
    y = split["yf"][:, rep_idx].astype("float32")
    ycf = split["ycf"][:, rep_idx].astype("float32")
    mu0 = split["mu0"][:, rep_idx].astype("float32")
    mu1 = split["mu1"][:, rep_idx].astype("float32")
    return x, t, y, ycf, mu0, mu1


def combine_splits(train, test, rep_idx: int):
    x_tr, t_tr, y_tr, ycf_tr, mu0_tr, mu1_tr = extract_rep(train, rep_idx)
    x_te, t_te, y_te, ycf_te, mu0_te, mu1_te = extract_rep(test, rep_idx)
    x = np.concatenate((x_tr, x_te), axis=0)
    t = np.concatenate((t_tr, t_te), axis=0)
    y = np.concatenate((y_tr, y_te), axis=0)
    ycf = np.concatenate((ycf_tr, ycf_te), axis=0)
    mu0 = np.concatenate((mu0_tr, mu0_te), axis=0)
    mu1 = np.concatenate((mu1_tr, mu1_te), axis=0)
    return x, t, y, ycf, mu0, mu1


def compute_factors(x: np.ndarray, n_factors: int):
    if n_factors <= 0:
        return None
    n, p = x.shape
    k = min(n_factors, n, p)
    if k <= 0:
        return None
    u, _, _ = np.linalg.svd(x, full_matrices=False)
    f = u[:, :k]
    w = f.T @ x / np.sqrt(n)
    f_scores = x @ w.T / p
    return f_scores.astype("float32")


def compute_adaptive_k(x: np.ndarray) -> int:
    if x.ndim != 2 or x.shape[1] < 6:
        raise ValueError("Expected x with at least 6 covariates.")
    x6 = x[:, :6]
    _, s, _ = np.linalg.svd(x6, full_matrices=False)
    if s.size < 6:
        s = np.pad(s, (0, 6 - s.size), mode="constant", constant_values=0.0)
    ratios = []
    for k in range(5):
        denom = s[k + 1]
        ratios.append(float("inf") if denom == 0 else s[k] / denom)
    return int(np.argmax(ratios)) + 1  # 1..5


def estimate_ite_ddpm(
    ddpm,
    X: np.ndarray,
    X_orig: np.ndarray,
    scaler_y,
    scaler_xd,
    monte_carlo_size: int,
    mc_batch_size: int,
):
    n = X.shape[0]

    xd0_orig = np.column_stack([X_orig, np.zeros((n, 1))])
    xd1_orig = np.column_stack([X_orig, np.ones((n, 1))])

    cond0 = scaler_xd.transform(xd0_orig).astype(np.float32)
    cond1 = scaler_xd.transform(xd1_orig).astype(np.float32)

    if mc_batch_size is None or mc_batch_size <= 0:
        mc_batch_size = monte_carlo_size

    batch_sizes = [mc_batch_size] * (monte_carlo_size // mc_batch_size)
    if monte_carlo_size % mc_batch_size != 0:
        batch_sizes.append(monte_carlo_size % mc_batch_size)

    sum0 = np.zeros(n, dtype=np.float64)
    sum1 = np.zeros(n, dtype=np.float64)
    total_samples = 0

    ddpm.eval()

    with torch.no_grad():
        for batch_idx, batch_size in enumerate(batch_sizes):
            if batch_idx % 10 == 0 and len(batch_sizes) > 1:
                print(f"    MC batch {batch_idx+1}/{len(batch_sizes)}")

            samples0 = generate_conditional_samples(ddpm, cond0, n_samples=batch_size)
            samples1 = generate_conditional_samples(ddpm, cond1, n_samples=batch_size)

            samples0_np = samples0.detach().cpu().numpy().reshape(batch_size * n, 1)
            samples1_np = samples1.detach().cpu().numpy().reshape(batch_size * n, 1)

            y0 = scaler_y.inverse_transform(samples0_np).reshape(batch_size, n)
            y1 = scaler_y.inverse_transform(samples1_np).reshape(batch_size, n)

            sum0 += y0.sum(axis=0)
            sum1 += y1.sum(axis=0)
            total_samples += batch_size

    mean0 = sum0 / total_samples
    mean1 = sum1 / total_samples
    tau_hat = mean1 - mean0
    ate_hat = float(np.mean(tau_hat))
    return tau_hat, ate_hat


def main() -> None:
    parser = argparse.ArgumentParser(
        description="IHDP ACEE on combined data with adaptive n_factors."
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--rep_start", type=int, default=0)
    parser.add_argument("--rep_end", type=int, default=None)
    parser.add_argument(
        "--hidden_dims",
        type=parse_hidden_dims,
        default=parse_hidden_dims("512,256,256,256,128"),
        help="Hidden dims for diffusion model",
    )
    parser.add_argument("--dim_t", type=int, default=128)
    parser.add_argument("--n_steps", type=int, default=1000)
    parser.add_argument("--n_epochs", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument(
        "--batch_size",
        type=int,
        default=0,
        help="Training batch size (0 for full-batch)",
    )
    parser.add_argument(
        "--monte_carlo_size",
        type=int,
        default=400,
        help="Monte Carlo samples for both ATE and ITE.",
    )
    parser.add_argument("--mc_batch_size", type=int, default=200)
    parser.add_argument("--use_standard_scaler", action="store_true")
    parser.add_argument("--seed_base", type=int, default=0)
    parser.add_argument(
        "--results_dir",
        type=str,
        default=None,
        help="Directory to save CSV summaries (default: ACEE/IHDP/results).",
    )
    parser.add_argument("--save_tau_dir", type=str, default=None)

    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "data")
    results_dir = args.results_dir or os.path.join(script_dir, "results")

    train_npz, test_npz = load_npz(data_dir)
    n_rep = train_npz["t"].shape[1]

    rep_start = args.rep_start
    rep_end = args.rep_end if args.rep_end is not None else n_rep - 1
    if rep_start < 0 or rep_end < rep_start or rep_end >= n_rep:
        raise ValueError(
            f"rep range must be within [0, {n_rep - 1}] (got {rep_start}..{rep_end})"
        )

    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, "ihdp_ate_estimates_adaptive.csv")
    if os.path.exists(results_file):
        os.remove(results_file)

    with open(results_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "rep_idx",
                "n_factors",
                "n_total",
                "ate_hat",
                "ate_true",
                "ate_abs_error",
                "ite_mse",
                "ite_mae",
            ]
        )

    for rep_idx in range(rep_start, rep_end + 1):
        print("=" * 70)
        print(f"REP {rep_idx}/{rep_end}")
        print("=" * 70)

        seed_everything(args.seed_base + rep_idx)

        X_all, D_all, y_all, _, mu0_all, mu1_all = combine_splits(
            train_npz, test_npz, rep_idx
        )

        X_base = X_all[:, :6].astype("float32")
        n_factors = compute_adaptive_k(X_base)
        factors = compute_factors(X_base, n_factors)
        X_use = X_base if factors is None else np.concatenate([X_base, factors], axis=1)

        X_with_d = np.column_stack([X_use, D_all])
        if args.use_standard_scaler:
            scaler_xd = StandardScaler()
            scaler_y = StandardScaler()
        else:
            scaler_xd = QuantileTransformer(output_distribution="normal", random_state=rep_idx)
            scaler_y = QuantileTransformer(output_distribution="normal", random_state=rep_idx)

        scaler_xd.fit(X_with_d)
        scaler_y.fit(y_all.reshape(-1, 1))

        X_norm = scaler_xd.transform(X_with_d)
        y_norm = scaler_y.transform(y_all.reshape(-1, 1)).reshape(-1)
        X_only = X_norm[:, :-1]
        D_norm = X_norm[:, -1]

        y_target, cond = prepare_conditional_data(X_only, D_norm, y_norm)

        print("Training conditional DDPM on combined data...")
        ddpm = train_conditional_ddpm(
            train_target=torch.tensor(y_target, dtype=torch.float32),
            train_cond=torch.tensor(cond, dtype=torch.float32),
            n_epochs=args.n_epochs,
            lr=args.lr,
            hidden_dims=args.hidden_dims,
            dim_t=args.dim_t,
            n_steps=args.n_steps,
            device=args.device,
            batch_size=args.batch_size,
            verbose=True,
        )

        mc_size = max(1, int(args.monte_carlo_size))
        print(f"Estimating ATE/ITE with {mc_size} MC samples...")
        tau_hat, ate_hat = estimate_ite_ddpm(
            ddpm=ddpm,
            X=X_only.astype(np.float32),
            X_orig=X_use.astype(np.float32),
            scaler_y=scaler_y,
            scaler_xd=scaler_xd,
            monte_carlo_size=mc_size,
            mc_batch_size=args.mc_batch_size,
        )

        tau_true = mu1_all - mu0_all
        ate_true = float(np.mean(tau_true))
        ate_abs_error = float(np.abs(ate_hat - ate_true))
        ite_mse = float(np.mean((tau_hat - tau_true) ** 2))
        ite_mae = float(np.mean(np.abs(tau_hat - tau_true)))

        if args.save_tau_dir:
            os.makedirs(args.save_tau_dir, exist_ok=True)
            np.savetxt(
                os.path.join(args.save_tau_dir, f"rep_{rep_idx}_tau_hat.csv"),
                tau_hat,
                delimiter=",",
            )

        with open(results_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    rep_idx,
                    n_factors,
                    X_use.shape[0],
                    ate_hat,
                    ate_true,
                    ate_abs_error,
                    ite_mse,
                    ite_mae,
                ]
            )

        print(f"ATE_hat={ate_hat:.6f} | ATE_true={ate_true:.6f} | ITE_MSE={ite_mse:.6f} | ITE_MAE={ite_mae:.6f}")

    print(f"Saved results to {results_file}")


if __name__ == "__main__":
    main()
