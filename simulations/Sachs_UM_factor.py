#!/usr/bin/env python3
import argparse
import os
import random
import sys
import warnings

# Add parent directory to path to import utils
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

import torch

from sklearn.decomposition import PCA

from utils.vae_factor import ConfounderExtractorX, FitConfig


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


def parse_n_sizes(value: str):
    return [int(v) for v in value.split(",") if v.strip()]


def parse_components(value: str):
    return [int(v) for v in value.split(",") if v.strip()]


def discover_seeds(data_dir: str, n_sizes):
    seeds = set()
    for n_size in n_sizes:
        subdir = os.path.join(data_dir, f"n_{n_size}")
        if not os.path.isdir(subdir):
            continue
        for fname in os.listdir(subdir):
            if not (fname.startswith("train_") and fname.endswith(".csv")):
                continue
            seed_str = fname[len("train_") : -len(".csv")]
            if seed_str.isdigit():
                seeds.add(int(seed_str))
    return sorted(seeds)


def load_data(data_file: str, x_dim_expected: int | None = None):
    data = np.loadtxt(data_file, delimiter=",")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 1:
        raise ValueError(f"Expected at least 1 column, got {data.shape[1]}")

    x_dim = data.shape[1]
    if x_dim_expected is not None and x_dim_expected != x_dim:
        warnings.warn(
            f"Expected x_dim={x_dim_expected}, but file has x_dim={x_dim}. Using file value."
        )
    return data


def compute_adaptive_k(X: np.ndarray) -> int:
    if X.ndim != 2:
        raise ValueError("Expected X to be 2D.")
    if X.shape[1] < 2:
        return 1

    _, s, _ = np.linalg.svd(X, full_matrices=False)
    if s.size < 2:
        return 1

    max_k = min(X.shape[1] - 1, s.size - 1)
    if max_k <= 0:
        return 1

    ratios = []
    for k in range(max_k):
        denom = s[k + 1]
        ratios.append(float("inf") if denom == 0 else s[k] / denom)
    return int(np.argmax(ratios)) + 1


def compute_pca(X: np.ndarray, n_components: int):
    pca = PCA(n_components=n_components)
    return pca.fit_transform(X)


def compute_vae(
    X: np.ndarray,
    latent_dim: int,
    hidden_dim: int,
    num_layers: int,
    device: str | None,
    cfg: FitConfig,
):
    extractor = ConfounderExtractorX(
        x_dim=X.shape[1],
        latent_dim=latent_dim,
        x_binary_idx=[],
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        device=device,
    )
    extractor.fit(X, cfg=cfg)
    return extractor.transform(X)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute latent factors for Sachs_UM using PCA and VAE (X-only, n_500)."
    )
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--n_seeds", type=int, default=None)
    parser.add_argument(
        "--n_sizes",
        type=parse_n_sizes,
        default=parse_n_sizes("500"),
        help="Training sizes to process (only 500 is supported).",
    )
    parser.add_argument("--x_dim", type=int, default=None)

    parser.add_argument(
        "--pca_components",
        type=parse_components,
        default=parse_components("1,2,3"),
        help="Comma-separated PCA components to compute on X.",
    )
    parser.add_argument(
        "--pca_adaptive",
        action="store_true",
        default=True,
        help="Compute adaptive PCA components based on singular value ratios.",
    )
    parser.add_argument(
        "--no_pca_adaptive",
        action="store_false",
        dest="pca_adaptive",
        help="Disable adaptive PCA.",
    )

    parser.add_argument(
        "--vae_components",
        type=parse_components,
        default=parse_components("1,2,3"),
        help="Comma-separated VAE latent dimensions.",
    )
    parser.add_argument("--vae_epochs", type=int, default=500)
    parser.add_argument("--vae_batch_size", type=int, default=128)
    parser.add_argument("--vae_lr", type=float, default=5e-4)
    parser.add_argument("--vae_weight_decay", type=float, default=1e-4)
    parser.add_argument("--vae_beta", type=float, default=0.5)
    parser.add_argument("--vae_mc_samples", type=int, default=5)
    parser.add_argument("--vae_log_every", type=int, default=50)
    parser.add_argument("--vae_hidden_dim", type=int, default=200)
    parser.add_argument("--vae_num_layers", type=int, default=3)
    parser.add_argument("--vae_device", type=str, default=None)

    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    acee_root = os.path.dirname(script_dir)
    data_dir = args.data_dir or os.path.join(acee_root, "data", "sachs_um")

    if not os.path.isdir(data_dir):
        raise SystemExit(f"Data dir not found: {data_dir}")

    n_sizes = sorted(set(args.n_sizes))
    if n_sizes != [500]:
        warnings.warn(
            f"Sachs_UM factor extraction supports only n_500. Using [500] instead of {n_sizes}."
        )
        n_sizes = [500]

    if args.n_seeds is None:
        seeds = discover_seeds(data_dir, n_sizes)
    else:
        seeds = list(range(args.n_seeds))
    if not seeds:
        raise SystemExit("No seeds found to process.")

    if args.vae_mc_samples < 1:
        raise SystemExit("vae_mc_samples must be >= 1")

    print("\n" + "=" * 70)
    print("SACHS_UM FACTOR EXTRACTION")
    print("=" * 70)
    print(f"Data dir: {data_dir}")
    print(f"Seeds: {seeds if args.n_seeds is None else f'0..{args.n_seeds-1}'}")
    print(f"Training sizes: {n_sizes}")
    print(f"PCA components: {args.pca_components}")
    print(f"PCA adaptive: {args.pca_adaptive}")
    print(f"VAE components: {args.vae_components}")
    print(f"VAE epochs: {args.vae_epochs}")
    print(f"VAE batch size: {args.vae_batch_size}")
    print(f"VAE lr: {args.vae_lr}")
    print(f"VAE weight_decay: {args.vae_weight_decay}")
    print(f"VAE beta: {args.vae_beta}")
    print(f"VAE mc_samples: {args.vae_mc_samples}")
    print(f"VAE hidden dim: {args.vae_hidden_dim}")
    print(f"VAE num layers: {args.vae_num_layers}")
    print(f"VAE device: {args.vae_device or 'auto'}")
    print("=" * 70 + "\n")

    for seed in seeds:
        print(f"\n{'='*70}")
        print(f"SEED {seed}")
        print(f"{'='*70}")

        seed_everything(seed)

        for n_size in n_sizes:
            print(f"\n{'='*50}")
            print(f"SEED {seed} | Training Size: {n_size}")
            print(f"{'='*50}")

            data_subdir = os.path.join(data_dir, f"n_{n_size}")
            data_file = os.path.join(data_subdir, f"train_{seed}.csv")
            if not os.path.exists(data_file):
                warnings.warn(f"Missing data file: {data_file}")
                continue

            try:
                X = load_data(data_file, x_dim_expected=args.x_dim)
            except Exception as exc:
                warnings.warn(f"Failed to read {data_file}: {exc}")
                continue

            print(f"Loaded data: X shape {X.shape}")

            for k in args.pca_components:
                if k <= 0:
                    warnings.warn(f"Skipping PCA with non-positive n_components={k}")
                    continue
                try:
                    Z_pca = compute_pca(X=X, n_components=k)
                    pca_file = os.path.join(
                        data_subdir, f"seed_{seed}_pca_k{k}_z.csv"
                    )
                    np.savetxt(pca_file, Z_pca, delimiter=",")
                    print(f"Saved PCA factors (k={k}) to {pca_file}")
                except Exception as exc:
                    warnings.warn(f"PCA failed for {data_file} (k={k}): {exc}")

            if args.pca_adaptive:
                try:
                    k_adapt = compute_adaptive_k(X)
                    Z_pca_adapt = compute_pca(X=X, n_components=k_adapt)
                    pca_file = os.path.join(
                        data_subdir, f"seed_{seed}_pca_adaptive_k{k_adapt}_z.csv"
                    )
                    np.savetxt(pca_file, Z_pca_adapt, delimiter=",")
                    print(
                        f"Saved adaptive PCA factors (k={k_adapt}) to {pca_file}"
                    )
                except Exception as exc:
                    warnings.warn(f"Adaptive PCA failed for {data_file}: {exc}")

            for k in args.vae_components:
                if k <= 0:
                    warnings.warn(f"Skipping VAE with non-positive latent_dim={k}")
                    continue
                try:
                    cfg = FitConfig(
                        epochs=args.vae_epochs,
                        batch_size=args.vae_batch_size,
                        lr=args.vae_lr,
                        weight_decay=args.vae_weight_decay,
                        beta=args.vae_beta,
                        mc_samples=args.vae_mc_samples,
                        log_every=args.vae_log_every,
                    )
                    Z_vae = compute_vae(
                        X=X,
                        latent_dim=k,
                        hidden_dim=args.vae_hidden_dim,
                        num_layers=args.vae_num_layers,
                        device=args.vae_device,
                        cfg=cfg,
                    )
                    vae_file = os.path.join(data_subdir, f"seed_{seed}_vae_k{k}_z.csv")
                    np.savetxt(vae_file, Z_vae, delimiter=",")
                    print(f"Saved VAE factors (k={k}) to {vae_file}")
                except Exception as exc:
                    warnings.warn(f"VAE failed for {data_file} (k={k}): {exc}")


if __name__ == "__main__":
    main()
