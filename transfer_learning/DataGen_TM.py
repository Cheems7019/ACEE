#!/usr/bin/env python3
import argparse
import csv
import os
import random
import numpy as np


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)


def parse_hidden_dims(value: str):
    return [int(v) for v in value.split(",") if v.strip()]


def parse_n1_list(value: str):
    return [int(v) for v in value.split(",") if v.strip()]


class Sampler_TM_positive:
    """
    Positive transfer sampler: M1-style DGP with shared distribution across domains.

    - Correlated covariates X
    - Treatment D from propensity score based on X
    - Outcome network m([X, D]) via tanh MLP
    - Original and auxiliary data share the same network and noise scale
    """

    def __init__(
        self,
        sigma_orig=1.0,
        sigma_aux=1.0,
        x_dim=50,
        seed=None,
        hidden_dims=(5, 20),
        weight_scale=0.5,
        correlation=0.5,
    ):
        self.sigma = sigma_orig
        self.sigma_orig = sigma_orig
        self.sigma_aux = sigma_orig
        self.x_dim = x_dim
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.hidden_dims = tuple(hidden_dims)
        self.weight_scale = weight_scale
        self.correlation = correlation
        self._init_outcome_network()

    def _init_outcome_network(self):
        if len(self.hidden_dims) != 2:
            raise ValueError("hidden_dims must contain exactly two layer sizes.")
        input_dim = self.x_dim + 1
        h1, h2 = self.hidden_dims
        self.W1 = self.rng.normal(0, self.weight_scale, size=(input_dim, h1))
        self.b1 = self.rng.normal(0, self.weight_scale, size=(h1,))
        self.W2 = self.rng.normal(0, self.weight_scale, size=(h1, h2))
        self.b2 = self.rng.normal(0, self.weight_scale, size=(h2,))
        self.W3 = self.rng.normal(0, self.weight_scale, size=(h2, 1))
        self.b3 = self.rng.normal(0, self.weight_scale, size=(1,))

    def _build_correlation_matrix(self):
        correlation_matrix = np.eye(self.x_dim)
        for i in range(self.x_dim):
            for j in range(self.x_dim):
                if i != j:
                    distance = abs(i - j)
                    correlation_matrix[i, j] = self.correlation ** distance
        return correlation_matrix

    def _propensity_score(self, X):
        def get_col(idx):
            if idx < X.shape[1]:
                return X[:, idx]
            return np.zeros(X.shape[0])

        indicator_part = (
            1.2 * (get_col(0) > 0).astype(float)
            - 1.2 * (get_col(1) < 0).astype(float)
            + 1.0 * (get_col(2) > 0.5).astype(float)
            - 1.0 * (get_col(3) < -0.5).astype(float)
            + 0.8 * (get_col(4) > 0.5).astype(float)
            - 0.8 * (get_col(5) < -0.5).astype(float)
        )
        return 0.1 + 0.8 / (1 + np.exp(-indicator_part))

    def _outcome_mean(self, X, D):
        D = D.reshape(-1, 1)
        inputs = np.concatenate([X, D], axis=1)
        h1 = np.tanh(inputs @ self.W1 + self.b1)
        h2 = np.tanh(h1 @ self.W2 + self.b2)
        y = h2 @ self.W3 + self.b3
        return y.reshape(-1)

    def _sample_domain(self, n, domain: str):
        correlation_matrix = self._build_correlation_matrix()
        mean = np.zeros(self.x_dim)
        X = self.rng.multivariate_normal(mean, correlation_matrix, size=n)
        prop = self._propensity_score(X)
        D = self.rng.binomial(n=1, p=prop, size=n).reshape(-1, 1)
        mean_vector = self._outcome_mean(X, D)
        y = mean_vector + self.sigma * self.rng.standard_normal(n)
        X_with_d = np.concatenate([X, D], axis=1)
        return X_with_d, y

    def sample(self, n_0=1000, n_1=1000):
        X_with_d_orig, y_orig = self._sample_domain(n_0, "orig")
        X_with_d_aux, y_aux = self._sample_domain(n_1, "aux")
        return X_with_d_orig, y_orig, X_with_d_aux, y_aux

    def estimate_ate(self, n_mc=100000):
        rng = np.random.default_rng(None if self.seed is None else self.seed + 1000003)
        correlation_matrix = self._build_correlation_matrix()
        mean = np.zeros(self.x_dim)
        X = rng.multivariate_normal(mean, correlation_matrix, size=n_mc)
        y0 = self._outcome_mean(X, np.zeros(n_mc))
        y1 = self._outcome_mean(X, np.ones(n_mc))
        return float(np.mean(y1 - y0))

    def estimate_sate(self, X_orig: np.ndarray) -> float:
        n = X_orig.shape[0]
        y0 = self._outcome_mean(X_orig, np.zeros(n))
        y1 = self._outcome_mean(X_orig, np.ones(n))
        return float(np.mean(y1 - y0))


class Sampler_TM_negative:
    """
    Negative transfer sampler: separate MLPs for original and auxiliary domains
    using disjoint covariate subsets.

    - Correlated covariates X
    - Original outcome MLP uses first half of covariates
    - Auxiliary outcome MLP uses last half of covariates
    - Propensity score for aux also uses last half
    """

    def __init__(
        self,
        sigma_orig=1.0,
        sigma_aux=1.0,
        x_dim=50,
        seed=None,
        hidden_dims=(20, 20),
        weight_scale=0.5,
        correlation=0.5,
    ):
        self.sigma_orig = sigma_orig
        self.sigma_aux = sigma_aux
        self.x_dim = x_dim
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.hidden_dims = tuple(hidden_dims)
        self.weight_scale = weight_scale
        self.correlation = correlation
        self._validate_dimensions()
        self._init_domain_networks()

    def _validate_dimensions(self):
        if self.x_dim < 2:
            raise ValueError("x_dim must be at least 2 to split covariates.")
        if len(self.hidden_dims) < 1:
            raise ValueError("hidden_dims must contain at least one layer size.")

    def _init_domain_networks(self):
        half = self.x_dim // 2
        if half == 0 or self.x_dim - half == 0:
            raise ValueError("x_dim split must leave at least one covariate per half.")

        orig_input_dim = half + 1
        aux_input_dim = (self.x_dim - half) + 1
        orig_layer_dims = (orig_input_dim,) + self.hidden_dims + (1,)
        aux_layer_dims = (aux_input_dim,) + self.hidden_dims + (1,)

        self.W_orig = []
        self.b_orig = []
        for in_dim, out_dim in zip(orig_layer_dims[:-1], orig_layer_dims[1:]):
            self.W_orig.append(self.rng.normal(0, self.weight_scale, size=(in_dim, out_dim)))
            self.b_orig.append(self.rng.normal(0, self.weight_scale, size=(out_dim,)))

        self.W_aux = []
        self.b_aux = []
        for in_dim, out_dim in zip(aux_layer_dims[:-1], aux_layer_dims[1:]):
            self.W_aux.append(self.rng.normal(0, self.weight_scale, size=(in_dim, out_dim)))
            self.b_aux.append(self.rng.normal(0, self.weight_scale, size=(out_dim,)))

    def _build_correlation_matrix(self):
        correlation_matrix = np.eye(self.x_dim)
        for i in range(self.x_dim):
            for j in range(self.x_dim):
                if i != j:
                    distance = abs(i - j)
                    correlation_matrix[i, j] = self.correlation ** distance
        return correlation_matrix

    def _propensity_score(self, X):
        def get_col(idx):
            if idx < X.shape[1]:
                return X[:, idx]
            return np.zeros(X.shape[0])

        indicator_part = (
            1.2 * (get_col(0) > 0).astype(float)
            - 1.2 * (get_col(1) < 0).astype(float)
            + 1.0 * (get_col(2) > 0.5).astype(float)
            - 1.0 * (get_col(3) < -0.5).astype(float)
            + 0.8 * (get_col(4) > 0.5).astype(float)
            - 0.8 * (get_col(5) < -0.5).astype(float)
        )
        return 0.1 + 0.8 / (1 + np.exp(-indicator_part))

    def _mlp_forward(self, X_sub, D, W_layers, b_layers):
        D = D.reshape(-1, 1)
        h = np.concatenate([X_sub, D], axis=1)
        for i, (W, b) in enumerate(zip(W_layers, b_layers)):
            h = h @ W + b
            if i < len(W_layers) - 1:
                h = np.tanh(h)
        return h.reshape(-1)

    def _outcome_mean(self, X, D, domain: str):
        if domain == "orig":
            return self._mlp_forward(X, D, self.W_orig, self.b_orig)
        if domain == "aux":
            return self._mlp_forward(X, D, self.W_aux, self.b_aux)
        raise ValueError(f"Unknown domain: {domain}")

    def _sample_domain(self, n, domain: str):
        correlation_matrix = self._build_correlation_matrix()
        mean = np.zeros(self.x_dim)
        X = self.rng.multivariate_normal(mean, correlation_matrix, size=n)
        half = self.x_dim // 2
        if domain == "orig":
            X_sub = X[:, :half]
            prop = self._propensity_score(X_sub)
        elif domain == "aux":
            X_sub = X[:, half:]
            prop = self._propensity_score(X_sub)
        else:
            raise ValueError(f"Unknown domain: {domain}")
        D = self.rng.binomial(n=1, p=prop, size=n).reshape(-1, 1)
        if domain == "orig":
            mean_vector = self._outcome_mean(X_sub, D, "orig")
            y = mean_vector + self.sigma_orig * self.rng.standard_normal(n)
        elif domain == "aux":
            mean_vector = self._outcome_mean(X_sub, D, "aux")
            y = mean_vector + self.sigma_aux * self.rng.standard_normal(n)
        X_with_d = np.concatenate([X, D], axis=1)
        return X_with_d, y

    def sample(self, n_0=1000, n_1=1000):
        X_with_d_orig, y_orig = self._sample_domain(n_0, "orig")
        X_with_d_aux, y_aux = self._sample_domain(n_1, "aux")
        return X_with_d_orig, y_orig, X_with_d_aux, y_aux

    def estimate_ate(self, n_mc=100000):
        rng = np.random.default_rng(None if self.seed is None else self.seed + 1000003)
        correlation_matrix = self._build_correlation_matrix()
        mean = np.zeros(self.x_dim)
        X = rng.multivariate_normal(mean, correlation_matrix, size=n_mc)
        half = self.x_dim // 2
        X_sub = X[:, :half]
        y0 = self._outcome_mean(X_sub, np.zeros(n_mc), "orig")
        y1 = self._outcome_mean(X_sub, np.ones(n_mc), "orig")
        return float(np.mean(y1 - y0))

    def estimate_sate(self, X_orig: np.ndarray) -> float:
        n = X_orig.shape[0]
        half = self.x_dim // 2
        X_sub = X_orig[:, :half]
        y0 = self._outcome_mean(X_sub, np.zeros(n), "orig")
        y1 = self._outcome_mean(X_sub, np.ones(n), "orig")
        return float(np.mean(y1 - y0))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate transfer-learning data for TM positive/negative samplers."
    )
    parser.add_argument("--n_seeds", type=int, default=50)
    parser.add_argument("--n0", type=int, default=200, help="Original dataset size.")
    parser.add_argument(
        "--n1",
        type=parse_n1_list,
        default=None,
        help="Auxiliary dataset sizes (comma-separated).",
    )
    parser.add_argument("--x_dim", type=int, default=15)
    parser.add_argument("--sigma_orig", type=float, default=1.0)
    parser.add_argument("--sigma_aux", type=float, default=1.0)
    parser.add_argument(
        "--sampler",
        type=str,
        default="positive",
        choices=["positive", "negative"],
        help="Sampler type: positive (correlated heads), negative (separate MLPs).",
    )
    parser.add_argument(
        "--weight_scale",
        type=float,
        default=0.75,
        help="Weight scale for representation network.",
    )
    parser.add_argument(
        "--correlation",
        type=float,
        default=0.5,
        help="Correlation parameter for Sampler_TM_positive covariates.",
    )
    parser.add_argument(
        "--head_weight_scale",
        type=float,
        default=0.5,
        help="Weight scale for domain-specific heads (final linear layer).",
    )
    parser.add_argument(
        "--head_corr_rho",
        type=float,
        default=0.9,
        help="Correlation between orig/aux head parameters for Sampler_TM_positive.",
    )
    parser.add_argument("--hidden_dims", type=parse_hidden_dims, default=None,
                        help="Hidden dimensions for shared representation network (comma-separated).")
    parser.add_argument("--domain_head_dims", type=parse_hidden_dims, default=None,
                        help="Hidden dimensions for domain-specific MLPs (comma-separated).")
    parser.add_argument("--ate_mc_samples", type=int, default=100000)
    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="Base directory to write data; defaults to ACEE/transfer_learning/data",
    )

    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    acee_root = os.path.dirname(script_dir)
    data_root = args.data_root or os.path.join(acee_root, "transfer_learning", "data")

    n0 = args.n0
    n1_list = args.n1 or [0, 200, 500, 1000]
    if args.hidden_dims is not None:
        hidden_dims = args.hidden_dims
    elif args.sampler == "positive":
        hidden_dims = [5, 20]
    else:
        hidden_dims = [20, 20]
    domain_head_dims = args.domain_head_dims or [20, 20]
    max_n1 = max(n1_list) if n1_list else 0

    if args.sampler == "positive":
        data_dir = os.path.join(data_root, "TM_positive")
        sampler_label = "TM_positive"
    else:
        data_dir = os.path.join(data_root, "TM_negative")
        sampler_label = "TM_negative"
    for n1 in n1_list:
        data_subdir = os.path.join(data_dir, f"n0_{n0}_n1_{n1}")
        os.makedirs(data_subdir, exist_ok=True)

    print("\n" + "=" * 70)
    print(f"DATA GENERATION: {sampler_label}")
    print("=" * 70)
    print(f"Seeds: {args.n_seeds}")
    print(f"Original size: {n0}")
    print(f"Auxiliary sizes: {n1_list}")
    print(f"X dimension: {args.x_dim}")
    print(f"Sigma (orig): {args.sigma_orig}")
    print(f"Sigma (aux): {args.sigma_aux}")
    print(f"Sampler: {args.sampler}")
    print(f"Weight scale (representation): {args.weight_scale}")
    if args.sampler == "positive":
        print(f"Correlation: {args.correlation}")
        print(f"Weight scale (outcome network): {args.weight_scale}")
        print(f"Outcome hidden dims: {hidden_dims}")
        print("Outcome network: tanh MLP on [X, D]")
    elif args.sampler == "negative":
        print(f"Correlation: {args.correlation}")
        print(f"Weight scale (domain MLPs): {args.weight_scale}")
        print(f"Domain MLP hidden dims (orig/aux): {hidden_dims}")
        print("Outcome covariates: orig uses first half, aux uses last half")
    print(f"ATE MC samples: {args.ate_mc_samples}")
    print(f"Data dir: {data_dir}")
    print("=" * 70 + "\n")

    ate_true_file = os.path.join(data_dir, "ate_true.csv")
    if os.path.exists(ate_true_file):
        os.remove(ate_true_file)

    with open(ate_true_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["seed", "ate_true_sate"])

    for seed in range(args.n_seeds):
        print(f"\n{'='*70}")
        print(f"SEED {seed}/{args.n_seeds - 1}")
        print(f"{'='*70}")

        seed_everything(seed)

        if args.sampler == "positive":
            sampler = Sampler_TM_positive(
                sigma_orig=args.sigma_orig,
                sigma_aux=args.sigma_aux,
                x_dim=args.x_dim,
                seed=seed,
                hidden_dims=hidden_dims,
                weight_scale=args.weight_scale,
                correlation=args.correlation,
            )
        else:
            sampler = Sampler_TM_negative(
                sigma_orig=args.sigma_orig,
                sigma_aux=args.sigma_aux,
                x_dim=args.x_dim,
                seed=seed,
                hidden_dims=hidden_dims,
                weight_scale=args.weight_scale,
                correlation=args.correlation,
            )

        print("\nSampling original data once per seed...")
        X_with_d_orig, y_orig = sampler._sample_domain(n0, "orig")
        if X_with_d_orig.shape != (n0, args.x_dim + 1):
            raise ValueError(
                f"Expected orig X_with_d shape ({n0}, {args.x_dim + 1}), got {X_with_d_orig.shape}"
            )
        if y_orig.shape != (n0,):
            raise ValueError(f"Expected orig y shape ({n0},), got {y_orig.shape}")

        X_with_d_aux_all = None
        y_aux_all = None
        if max_n1 > 0:
            print(f"Sampling auxiliary data once per seed (n1={max_n1})...")
            X_with_d_aux_all, y_aux_all = sampler._sample_domain(max_n1, "aux")
            if X_with_d_aux_all.shape != (max_n1, args.x_dim + 1):
                raise ValueError(
                    f"Expected aux X_with_d shape ({max_n1}, {args.x_dim + 1}), got {X_with_d_aux_all.shape}"
                )
            if y_aux_all.shape != (max_n1,):
                raise ValueError(f"Expected aux y shape ({max_n1},), got {y_aux_all.shape}")

        print(
            f"\nComputing true ATE for seed {seed}'s DGP with {args.ate_mc_samples} Monte Carlo samples..."
        )
        ate_true_pate = sampler.estimate_ate(n_mc=args.ate_mc_samples)
        X_orig_only = X_with_d_orig[:, :-1]
        ate_true_sate = sampler.estimate_sate(X_orig_only)
        print(f"True ATE (PATE) for this DGP: {ate_true_pate:.6f}")
        print(f"True ATE (SATE) for this seed's sample: {ate_true_sate:.6f}")

        with open(ate_true_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([seed, ate_true_sate])

        for n1 in n1_list:
            try:
                print(f"\nGenerating training data (n1={n1})...")
                X_with_d_aux = None
                y_aux = None
                if n1 > 0:
                    if X_with_d_aux_all is None or y_aux_all is None:
                        raise ValueError("Auxiliary data is not available but n1 > 0.")
                    if n1 > max_n1:
                        raise ValueError(f"Requested n1={n1} exceeds max_n1={max_n1}.")
                    X_with_d_aux = X_with_d_aux_all[:n1]
                    y_aux = y_aux_all[:n1]
                if n1 > 0:
                    if X_with_d_aux.shape != (n1, args.x_dim + 1):
                        raise ValueError(
                            f"Expected aux X_with_d shape ({n1}, {args.x_dim + 1}), got {X_with_d_aux.shape}"
                        )
                    if y_aux.shape != (n1,):
                        raise ValueError(f"Expected aux y shape ({n1},), got {y_aux.shape}")

                D_orig = X_with_d_orig[:, -1]

                print(f"Original treatment balance: {D_orig.mean():.3f}")
                print(f"Original y range: [{y_orig.min():.3f}, {y_orig.max():.3f}]")
                if n1 > 0:
                    D_aux = X_with_d_aux[:, -1]
                    print(f"Auxiliary treatment balance: {D_aux.mean():.3f}")
                    print(f"Auxiliary y range: [{y_aux.min():.3f}, {y_aux.max():.3f}]")

                data_subdir = os.path.join(data_dir, f"n0_{n0}_n1_{n1}")
                orig_file = os.path.join(data_subdir, f"seed_{seed}_orig.csv")
                orig_out = np.concatenate([X_with_d_orig, y_orig.reshape(-1, 1)], axis=1)
                np.savetxt(orig_file, orig_out, delimiter=",")
                print(f"Saved original data to {orig_file}")
                if n1 > 0:
                    aux_file = os.path.join(data_subdir, f"seed_{seed}_aux.csv")
                    aux_out = np.concatenate([X_with_d_aux, y_aux.reshape(-1, 1)], axis=1)
                    np.savetxt(aux_file, aux_out, delimiter=",")
                    print(f"Saved auxiliary data to {aux_file}")

            except Exception as exc:
                print(f"\nERROR processing seed={seed}, n1={n1}: {exc}")
                continue


if __name__ == "__main__":
    main()
