import numpy as np
from sklearn.neighbors import NearestNeighbors


def mbc(X, Y, Tr, M, Model1=None, Model0=None):
    """
    Bias-corrected nearest-neighbor matching estimator (Abadie-Imbens style).

    Args:
        X: Covariates, shape [N, d]
        Y: Outcomes, shape [N]
        Tr: Treatment indicator (0/1), shape [N]
        M: Number of nearest neighbors
        Model1: E[Y | X, D=1] for each unit, shape [N] (optional)
        Model0: E[Y | X, D=0] for each unit, shape [N] (optional)

    Returns:
        float: Bias-corrected ATE estimate
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float).reshape(-1)
    Tr = np.asarray(Tr).reshape(-1)

    if X.ndim != 2:
        raise ValueError("X must be 2D.")
    if len(Y) != len(Tr) or len(Y) != X.shape[0]:
        raise ValueError("X, Y, and Tr must have compatible lengths.")
    if M <= 0:
        raise ValueError("M must be positive.")

    if Model1 is None:
        Model1 = np.zeros_like(Y)
    if Model0 is None:
        Model0 = np.zeros_like(Y)
    Model1 = np.asarray(Model1, dtype=float).reshape(-1)
    Model0 = np.asarray(Model0, dtype=float).reshape(-1)
    if len(Model1) != len(Y) or len(Model0) != len(Y):
        raise ValueError("Model1/Model0 must match Y length.")

    # Scale covariates (column-wise) to match R's scale() behavior.
    mean = X.mean(axis=0)
    std = X.std(axis=0, ddof=1)
    std[std == 0] = 1.0
    Xs = (X - mean) / std

    mask1 = Tr == 1
    mask0 = Tr == 0
    if not np.any(mask1) or not np.any(mask0):
        raise ValueError("Both treatment groups must be non-empty.")

    X1 = Xs[mask1]
    X0 = Xs[mask0]
    Y1 = Y[mask1]
    Y0 = Y[mask0]
    M1 = Model1[mask1]
    M0 = Model0[mask0]

    N1 = X1.shape[0]
    N0 = X0.shape[0]
    if M > min(N0, N1):
        raise ValueError("M is larger than the smallest treatment group.")

    # Match controls (X0) to treated (X1)
    nn1 = NearestNeighbors(n_neighbors=M, algorithm="auto")
    nn1.fit(X1)
    index1 = nn1.kneighbors(X0, return_distance=False)
    k1m = np.bincount(index1.ravel(), minlength=N1) / M

    # Match treated (X1) to controls (X0)
    nn0 = NearestNeighbors(n_neighbors=M, algorithm="auto")
    nn0.fit(X0)
    index0 = nn0.kneighbors(X1, return_distance=False)
    k0m = np.bincount(index0.ravel(), minlength=N0) / M

    res1 = (1.0 + k1m) * (Y1 - M1)
    res0 = (1.0 + k0m) * (Y0 - M0)

    est = np.mean(Model1 - Model0) + np.mean(np.concatenate([res1, -res0]))
    return float(est)
