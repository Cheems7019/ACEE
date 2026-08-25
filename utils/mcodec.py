import numpy as np
from sklearn.metrics import pairwise_distances

def _nearest_neighbor_indices_tiebreak(X, rng=None):
    """
    For each row of X, return the index of a nearest neighbor (excluding itself),
    breaking distance ties uniformly at random.
    """
    X = np.asarray(X)
    n = X.shape[0]
    dmat = pairwise_distances(X)
    np.fill_diagonal(dmat, np.inf)

    if rng is None:
        rng = np.random.default_rng()

    mins = dmat.min(axis=1)
    is_min = dmat == mins[:, None]

    nn_idx = np.empty(n, dtype=int)
    for i in range(n):
        candidates = np.flatnonzero(is_min[i])
        nn_idx[i] = rng.choice(candidates)

    return nn_idx

def codec_unconditional(y, Z):
    """
    Unconditional Azadkia–Chatterjee coefficient T_n(Y, Z).
    
    Parameters
    ----------
    y : (n,) or (n,1) array-like
        Response Y (real-valued).
    Z : (n, q) array-like
        Covariates Z.

    Returns
    -------
    float in [0, 1]
    """
    y = np.asarray(y).reshape(-1)
    Z = np.asarray(Z)
    n = y.shape[0]
    if Z.shape[0] != n:
        raise ValueError("y and Z must have the same number of rows")

    # Sort by y ascending
    sort_idx = np.argsort(y)
    Z_sorted = Z[sort_idx, :]

    # NN indices in Z-space
    nn_Z = _nearest_neighbor_indices_tiebreak(Z_sorted)  # shape (n,)

    num = 0.0
    den = 0.0
    for r in range(n):
        r_M = nn_Z[r]
        # L = n - r, but we never need L explicitly
        num += n * (min(r, r_M) + 1) - (n - r) ** 2
        den += (n - r) * r

    if den == 0.0:
        # Degenerate case: Y constant
        return np.nan

    return num / den

def codec_conditional(y, Z, X):
    """
    Conditional Azadkia–Chatterjee coefficient T_n(Y, Z | X).
    
    Parameters
    ----------
    y : (n,) or (n,1) array-like
        Response Y.
    Z : (n, q_z) array-like
        Covariates Z.
    X : (n, p) array-like
        Conditioning variables X.

    Returns
    -------
    float in [0, 1]
    """
    y = np.asarray(y).reshape(-1)
    Z = np.asarray(Z)
    X = np.asarray(X)

    n = y.shape[0]
    if Z.shape[0] != n or X.shape[0] != n:
        raise ValueError("y, Z, X must have the same number of rows")

    # Sort by y
    sort_idx = np.argsort(y)
    Z_sorted = Z[sort_idx, :]
    X_sorted = X[sort_idx, :]

    # Distances in X-space
    nn_X = _nearest_neighbor_indices_tiebreak(X_sorted)  # N(r)

    # Distances in (Z, X)-space: concat Z and X as columns
    ZX_sorted = np.concatenate([Z_sorted, X_sorted], axis=1)
    nn_ZX = _nearest_neighbor_indices_tiebreak(ZX_sorted)  # M(r)

    num = 0.0
    den = 0.0
    for r in range(n):
        r_N = nn_X[r]
        r_M = nn_ZX[r]
        num += min(r, r_M) - min(r, r_N)
        den += r - min(r, r_N)

    if den == 0.0:
        # Degenerate: Y almost-deterministic in X
        return np.nan

    return num / den

def codec(y, Z, X=None):
    """
    Wrapper matching the FOCI::codec behavior:
    - codec(y, Z)        -> T_n(Y, Z)
    - codec(y, Z, X=X)   -> T_n(Y, Z | X)
    """
    if X is None:
        return codec_unconditional(y, Z)
    else:
        return codec_conditional(y, Z, X)

def didec_no_perm(X, Y, codec=codec):
    """
    Python analogue of Codec.Tq(X, Y) (no permutations).

    X: (n, p)
    Y: (n, q)
    codec: function(codec(y, Z, X=None)) implementing A–C CODEC.
    """
    X = np.asarray(X)
    Y = np.asarray(Y)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)

    n, q = Y.shape
    if X.shape[0] != n:
        raise ValueError("X and Y must have the same number of rows")

    ZW = np.zeros(q)
    weight = np.zeros(q)

    # First coordinate: Y_1
    ZW[0] = codec(Y[:, 0], Z=X, X=None)   # T_n(Y1, X)
    weight[0] = 0.0

    # Subsequent coordinates
    for i in range(1, q):
        prevY = Y[:, :i]
        # weight[i] = codec(Y_{i+1}, previous Y's)
        weight[i] = codec(Y[:, i], Z=prevY, X=None)
        # ZW[i] = codec(Y_{i+1}, (X, previous Y's))
        ZW[i] = codec(Y[:, i], Z=np.column_stack([X, prevY]), X=None)

    num = q - ZW.sum()
    den = q - weight.sum()
    if den <= 0:
        return np.nan

    return 1.0 - num / den

def mcodec(Y, X, Z, eps=1e-10):
    """
    Compute T_{vec,n}(Y, X | Z)
      = (T_n(Y, [Z, X]) - T_n(Y, Z)) / (1 - T_n(Y, Z)),

    where T_n(Y, W) is the multivariate directed dependence T^q(Y | W)
    implemented by didec_no_perm(W, Y).
    """
    Y = np.asarray(Y)
    X = np.asarray(X)
    Z = np.asarray(Z)

    # predictors = [Z, X]
    XZ = np.concatenate([Z, X], axis=1)

    # T_n(Y, [Z, X]) = T^q(Y | [Z, X])
    T_full = didec_no_perm(XZ, Y)

    # T_n(Y, Z) = T^q(Y | Z)
    T_Z = didec_no_perm(Z, Y)

    if 1 - T_Z < eps:   # degenerate: Y almost a function of Z
        # you can choose to return np.nan, 1.0, or raise
        return np.nan

    T_vec_hat = (T_full - T_Z) / (1.0 - T_Z)
    return T_vec_hat
