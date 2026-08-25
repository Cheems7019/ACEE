import numpy as np

class Sampler_M1:
    """
    Sampler with correlated covariates X, treatment D from a propensity score,
    and nonlinear outcome Y with homoscedastic (constant variance) noise.
    
    Outcome model: Y = m(X,D) + sigma * epsilon, where epsilon ~ N(0,1)
    and m(X,D) is a random neural network (not learned).
    """
    def __init__(
        self,
        sigma=1.0,
        correlation=0.5,
        x_dim=20,
        seed=None,
        hidden_dims=(50, 50),
        weight_scale=0.5,
    ):
        self.sigma = sigma
        self.correlation = correlation
        self.x_dim = x_dim
        self.rng = np.random.default_rng(seed)
        self.hidden_dims = hidden_dims
        self.weight_scale = weight_scale
        self._init_outcome_network()

    def _init_outcome_network(self):
        input_dim = self.x_dim + 1  # X plus treatment
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

    def _binarize_covariates(self, X):
        """
        Binarize the last 1/3 of covariates.
        
        Args:
            X: [n, x_dim] array of continuous covariates
        
        Returns:
            X_modified: [n, x_dim] array with last 1/3 binarized
        """
        X_modified = X.copy()
        n_binary = self.x_dim // 3  # Last 1/3
        binary_start_idx = self.x_dim - n_binary
        
        # Binarize: 1 if > 0, else 0
        X_modified[:, binary_start_idx:] = (X[:, binary_start_idx:] > 0).astype(float)
        
        return X_modified

    def _propensity_score(self, X):
        """
        Compute propensity score with discontinuities (DISCRETE version, indicator only).
        
        Includes:
        - Indicator functions (jumps/discontinuities) on continuous normal covariates
        - Bounded to [0.1, 0.9] for identifiability
        
        Note: X is kept as correlated normal (not binarized).
              Only indicator jumps, no binary or continuous components.
        """
        def get_col(idx):
            if idx < X.shape[1]:
                return X[:, idx]
            return np.zeros(X.shape[0])
        
        # Discontinuous components (indicators) - BALANCED AROUND 0
        indicator_part = (
            1.2 * (get_col(0) > 0).astype(float)       # Jump at X[0]=0: +1.2 if X[0]>0
            - 1.2 * (get_col(1) < 0).astype(float)     # Jump at X[1]=0: -1.2 if X[1]<0
            + 1.0 * (get_col(2) > 0.5).astype(float)   # Jump at X[2]=0.5: +1.0 if X[2]>0.5
            - 1.0 * (get_col(3) < -0.5).astype(float)  # Jump at X[3]=-0.5: -1.0 if X[3]<-0.5
            + 0.8 * (get_col(4) > 0.5).astype(float)   # Jump at X[4]=0.5: +0.8 if X[4]>0.5
            - 0.8 * (get_col(5) < -0.5).astype(float)  # Jump at X[5]=-0.5: -0.8 if X[5]<-0.5
        )
        
        # Total score (only indicator part)
        score = indicator_part
        
        # Bound to [0.1, 0.9]
        return 0.1 + 0.8 / (1 + np.exp(-score))

    def _outcome_mean(self, X, D):
        """Compute mean function m(X,D) via random neural network"""
        D = D.reshape(-1, 1)
        inputs = np.concatenate([X, D], axis=1)
        h1 = np.tanh(inputs @ self.W1 + self.b1)
        h2 = np.tanh(h1 @ self.W2 + self.b2)
        y = h2 @ self.W3 + self.b3
        return y.reshape(-1)

    def sample(self, n=1000, return_potential_outcomes=False, x_scale=1.0):
        """
        Sample n observations from the data generating process.
        
        Outcome model: Y = m(X,D) + sigma * epsilon, where epsilon ~ N(0,1)
        
        Args:
            n: Number of samples
            return_potential_outcomes: If True, also return mu0 and mu1 (conditional means)
        
        Returns:
            X_with_d: [n, x_dim+1] array with features and treatment
            y: [n] array of observed outcomes
            mu0, mu1: [n] arrays of conditional means m(X,0) and m(X,1)
                     (only if return_potential_outcomes=True)
        
        Note:
            - Returned potential outcomes are conditional means (noise-free):
              mu0 = m(X,0), mu1 = m(X,1)
            - This is appropriate for ATE since E[Y(d)] = E[m(X,d)] as E[epsilon]=0
            - To compute the true population ATE, use estimate_ate() with a large n_mc.
        """
        correlation_matrix = self._build_correlation_matrix()
        mean = np.zeros(self.x_dim)
        X = self.rng.multivariate_normal(mean, correlation_matrix, size=n)
        if x_scale is not None and x_scale != 1.0:
            X = X * float(x_scale)
        
        # Keep X as correlated normal (no binarization)

        prop = self._propensity_score(X)
        D = self.rng.binomial(n=1, p=prop, size=n)
        D = D.reshape(-1, 1)

        mean_vector = self._outcome_mean(X, D)
        y = mean_vector + self.sigma * self.rng.standard_normal(n)

        X_with_d = np.concatenate([X, D], axis=1)

        if return_potential_outcomes:
            # Return conditional means (noise-free potential outcomes)
            mu0 = self._outcome_mean(X, np.zeros(n))
            mu1 = self._outcome_mean(X, np.ones(n))
            return X_with_d, y, mu0, mu1

        return X_with_d, y

    def estimate_ate(self, n_mc=100000):
        """
        Estimate the true population Average Treatment Effect (ATE).
        
        Args:
            n_mc: Number of Monte Carlo samples (use large value for accurate estimate)
        
        Returns:
            float: Estimated population ATE = E[Y(1) - Y(0)]
        
        Note:
            This uses Monte Carlo integration over the population distribution of X.
            Use a large n_mc (e.g., 100000) for an accurate estimate of the true ATE.
        """
        correlation_matrix = self._build_correlation_matrix()
        mean = np.zeros(self.x_dim)
        X = self.rng.multivariate_normal(mean, correlation_matrix, size=n_mc)
        y0 = self._outcome_mean(X, np.zeros(n_mc))
        y1 = self._outcome_mean(X, np.ones(n_mc))
        return float(np.mean(y1 - y0))


class Sampler_M2:
    """
    Sampler with correlated covariates X, treatment D from a propensity score,
    and nonlinear outcome Y with heteroscedastic noise: Y = m(X,D) + sigma * g(X,D) * epsilon
    where epsilon ~ N(0,1) and g(X,D) is a random neural variance function.
    
    Both m(X,D) and g(X,D) are randomly initialized neural networks (not learned).
    """
    def __init__(
        self,
        sigma=1.0,
        correlation=0.5,
        x_dim=20,
        seed=None,
        hidden_dims=(50, 50),
        weight_scale=0.5,
    ):
        self.sigma = sigma
        self.correlation = correlation
        self.x_dim = x_dim
        self.rng = np.random.default_rng(seed)
        self.hidden_dims = hidden_dims
        self.weight_scale = weight_scale
        self._init_outcome_networks()

    def _init_outcome_networks(self):
        """Initialize two neural networks: one for mean m(X,D) and one for variance g(X,D)"""
        input_dim = self.x_dim + 1  # X plus treatment
        h1, h2 = self.hidden_dims
        
        # Network for mean function m(X,D)
        self.W1_m = self.rng.normal(0, self.weight_scale, size=(input_dim, h1))
        self.b1_m = self.rng.normal(0, self.weight_scale, size=(h1,))
        self.W2_m = self.rng.normal(0, self.weight_scale, size=(h1, h2))
        self.b2_m = self.rng.normal(0, self.weight_scale, size=(h2,))
        self.W3_m = self.rng.normal(0, self.weight_scale, size=(h2, 1))
        self.b3_m = self.rng.normal(0, self.weight_scale, size=(1,))
        
        # Network for variance function g(X,D)
        self.W1_g = self.rng.normal(0, self.weight_scale, size=(input_dim, h1))
        self.b1_g = self.rng.normal(0, self.weight_scale, size=(h1,))
        self.W2_g = self.rng.normal(0, self.weight_scale, size=(h1, h2))
        self.b2_g = self.rng.normal(0, self.weight_scale, size=(h2,))
        self.W3_g = self.rng.normal(0, self.weight_scale, size=(h2, 1))
        self.b3_g = self.rng.normal(0, self.weight_scale, size=(1,))

    def _build_correlation_matrix(self):
        correlation_matrix = np.eye(self.x_dim)
        for i in range(self.x_dim):
            for j in range(self.x_dim):
                if i != j:
                    distance = abs(i - j)
                    correlation_matrix[i, j] = self.correlation ** distance
        return correlation_matrix

    def _binarize_covariates(self, X):
        """
        Binarize the last 1/3 of covariates.
        
        Args:
            X: [n, x_dim] array of continuous covariates
        
        Returns:
            X_modified: [n, x_dim] array with last 1/3 binarized
        """
        X_modified = X.copy()
        n_binary = self.x_dim // 3  # Last 1/3
        binary_start_idx = self.x_dim - n_binary
        
        # Binarize: 1 if > 0, else 0
        X_modified[:, binary_start_idx:] = (X[:, binary_start_idx:] > 0).astype(float)
        
        return X_modified

    def _propensity_score(self, X):
        """
        Compute propensity score with discontinuities (DISCRETE version, indicator only).
        
        Includes:
        - Indicator functions (jumps/discontinuities) on continuous normal covariates
        - Bounded to [0.1, 0.9] for identifiability
        
        Note: X is kept as correlated normal (not binarized).
              Only indicator jumps, no binary or continuous components.
        """
        def get_col(idx):
            if idx < X.shape[1]:
                return X[:, idx]
            return np.zeros(X.shape[0])
        
        # Discontinuous components (indicators) - BALANCED AROUND 0
        indicator_part = (
            1.2 * (get_col(0) > 0).astype(float)       # Jump at X[0]=0: +1.2 if X[0]>0
            - 1.2 * (get_col(1) < 0).astype(float)     # Jump at X[1]=0: -1.2 if X[1]<0
            + 1.0 * (get_col(2) > 0.5).astype(float)   # Jump at X[2]=0.5: +1.0 if X[2]>0.5
            - 1.0 * (get_col(3) < -0.5).astype(float)  # Jump at X[3]=-0.5: -1.0 if X[3]<-0.5
            + 0.8 * (get_col(4) > 0.5).astype(float)   # Jump at X[4]=0.5: +0.8 if X[4]>0.5
            - 0.8 * (get_col(5) < -0.5).astype(float)  # Jump at X[5]=-0.5: -0.8 if X[5]<-0.5
        )
        
        # Total score (only indicator part)
        score = indicator_part
        
        # Bound to [0.1, 0.9]
        return 0.1 + 0.8 / (1 + np.exp(-score))

    def _outcome_mean(self, X, D):
        """Compute mean function m(X,D) via random neural network"""
        D = D.reshape(-1, 1)
        inputs = np.concatenate([X, D], axis=1)
        h1 = np.tanh(inputs @ self.W1_m + self.b1_m)
        h2 = np.tanh(h1 @ self.W2_m + self.b2_m)
        y = h2 @ self.W3_m + self.b3_m
        return y.reshape(-1)

    def _outcome_variance_scale(self, X, D):
        """
        Compute variance scaling function g(X,D) with positivity constraint, floor, and ceiling.
        
        This is a random neural network (fixed, not learned) that modulates noise variance.
        Uses softplus activation with floor and clipping: g(X,D) in [0.1, 5.0]
        This prevents extreme variance values from dominating the simulation.
        """
        D = D.reshape(-1, 1)
        inputs = np.concatenate([X, D], axis=1)
        h1 = np.tanh(inputs @ self.W1_g + self.b1_g)
        h2 = np.tanh(h1 @ self.W2_g + self.b2_g)
        g_raw = (h2 @ self.W3_g + self.b3_g).reshape(-1)
        g = 0.1 + np.log1p(np.exp(g_raw))   # softplus + floor
        g = np.clip(g, 0.1, 5.0)             # clip to [0.1, 5.0]
        return g

    def sample(self, n=1000, return_potential_outcomes=False, x_scale=1.0):
        """
        Sample n observations from the data generating process.
        
        Outcome model: Y = m(X,D) + sigma * g(X,D) * epsilon
        where epsilon ~ N(0,1), m is the mean function, and g is the variance scaling function.
        
        Args:
            n: Number of samples
            return_potential_outcomes: If True, also return mu0 and mu1 (conditional means)
        
        Returns:
            X_with_d: [n, x_dim+1] array with features and treatment
            y: [n] array of observed outcomes
            mu0, mu1: [n] arrays of conditional means m(X,0) and m(X,1)
                     (only if return_potential_outcomes=True)
        
        Note:
            - Returned potential outcomes are conditional means (noise-free): 
              mu0 = m(X,0), mu1 = m(X,1)
            - This is appropriate for ATE since E[Y(d)] = E[m(X,d)] as E[epsilon]=0
            - To compute the true population ATE, use estimate_ate() with a large n_mc.
        """
        correlation_matrix = self._build_correlation_matrix()
        mean = np.zeros(self.x_dim)
        X = self.rng.multivariate_normal(mean, correlation_matrix, size=n)
        if x_scale is not None and x_scale != 1.0:
            X = X * float(x_scale)
        
        # Keep X as correlated normal (no binarization)

        prop = self._propensity_score(X)
        D = self.rng.binomial(n=1, p=prop, size=n)
        D = D.reshape(-1, 1)

        # Heteroscedastic noise: Y = m(X,D) + sigma * g(X,D) * epsilon
        mean_vector = self._outcome_mean(X, D)
        variance_scale = self._outcome_variance_scale(X, D)
        epsilon = self.rng.standard_normal(n)
        y = mean_vector + self.sigma * variance_scale * epsilon

        X_with_d = np.concatenate([X, D], axis=1)

        if return_potential_outcomes:
            # Return conditional means (noise-free potential outcomes)
            mu0 = self._outcome_mean(X, np.zeros(n))
            mu1 = self._outcome_mean(X, np.ones(n))
            return X_with_d, y, mu0, mu1

        return X_with_d, y

    def estimate_ate(self, n_mc=100000):
        """
        Estimate the true population Average Treatment Effect (ATE).
        
        Args:
            n_mc: Number of Monte Carlo samples (use large value for accurate estimate)
        
        Returns:
            float: Estimated population ATE = E[Y(1) - Y(0)]
        
        Note:
            This uses Monte Carlo integration over the population distribution of X.
            Use a large n_mc (e.g., 100000) for an accurate estimate of the true ATE.
            The ATE is based on the mean function m(X,D), not affected by the noise structure.
        """
        correlation_matrix = self._build_correlation_matrix()
        mean = np.zeros(self.x_dim)
        X = self.rng.multivariate_normal(mean, correlation_matrix, size=n_mc)
        y0 = self._outcome_mean(X, np.zeros(n_mc))
        y1 = self._outcome_mean(X, np.ones(n_mc))
        return float(np.mean(y1 - y0))


class Sampler_M3:
    """
    Sampler with correlated covariates X, treatment D from a propensity score,
    and nonlinear outcome Y with multiplicative log-normal noise (mean-one).
    
    Outcome model: Y = m(X,D) × exp(ε - σ²/2), where ε ~ N(0, σ²)
    
    The mean-one property ensures E[Y(d)|X] = m(X,d), so the ATE is independent of σ:
    ATE = E[m(X,1) - m(X,0)]
    
    m(X,D) is a random neural network (not learned), can be negative.
    """
    def __init__(
        self,
        sigma=1.0,
        correlation=0.5,
        x_dim=20,
        seed=None,
        hidden_dims=(50, 50),
        weight_scale=0.5,
    ):
        self.sigma = sigma
        self.correlation = correlation
        self.x_dim = x_dim
        self.rng = np.random.default_rng(seed)
        self.hidden_dims = hidden_dims
        self.weight_scale = weight_scale
        self._init_outcome_network()

    def _init_outcome_network(self):
        """Initialize neural network for mean function m(X,D)"""
        input_dim = self.x_dim + 1  # X plus treatment
        h1, h2 = self.hidden_dims
        
        # Network for mean function m(X,D)
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

    def _binarize_covariates(self, X):
        """
        Binarize the last 1/3 of covariates.
        
        Args:
            X: [n, x_dim] array of continuous covariates
        
        Returns:
            X_modified: [n, x_dim] array with last 1/3 binarized
        """
        X_modified = X.copy()
        n_binary = self.x_dim // 3  # Last 1/3
        binary_start_idx = self.x_dim - n_binary
        
        # Binarize: 1 if > 0, else 0
        X_modified[:, binary_start_idx:] = (X[:, binary_start_idx:] > 0).astype(float)
        
        return X_modified

    def _propensity_score(self, X):
        """
        Compute propensity score with discontinuities (DISCRETE version, indicator only).
        
        Includes:
        - Indicator functions (jumps/discontinuities) on continuous normal covariates
        - Bounded to [0.1, 0.9] for identifiability
        
        Note: X is kept as correlated normal (not binarized).
              Only indicator jumps, no binary or continuous components.
        """
        def get_col(idx):
            if idx < X.shape[1]:
                return X[:, idx]
            return np.zeros(X.shape[0])
        
        # Discontinuous components (indicators) - BALANCED AROUND 0
        indicator_part = (
            1.2 * (get_col(0) > 0).astype(float)       # Jump at X[0]=0: +1.2 if X[0]>0
            - 1.2 * (get_col(1) < 0).astype(float)     # Jump at X[1]=0: -1.2 if X[1]<0
            + 1.0 * (get_col(2) > 0.5).astype(float)   # Jump at X[2]=0.5: +1.0 if X[2]>0.5
            - 1.0 * (get_col(3) < -0.5).astype(float)  # Jump at X[3]=-0.5: -1.0 if X[3]<-0.5
            + 0.8 * (get_col(4) > 0.5).astype(float)   # Jump at X[4]=0.5: +0.8 if X[4]>0.5
            - 0.8 * (get_col(5) < -0.5).astype(float)  # Jump at X[5]=-0.5: -0.8 if X[5]<-0.5
        )
        
        # Total score (only indicator part)
        score = indicator_part
        
        # Bound to [0.1, 0.9]
        return 0.1 + 0.8 / (1 + np.exp(-score))

    def _outcome_mean(self, X, D):
        """Compute mean function m(X,D) via random neural network"""
        D = D.reshape(-1, 1)
        inputs = np.concatenate([X, D], axis=1)
        h1 = np.tanh(inputs @ self.W1 + self.b1)
        h2 = np.tanh(h1 @ self.W2 + self.b2)
        y = h2 @ self.W3 + self.b3
        return y.reshape(-1)

    def sample(self, n=1000, return_potential_outcomes=False, x_scale=1.0):
        """
        Sample n observations from the data generating process.
        
        Outcome model: Y = m(X,D) × exp(ε - σ²/2)
        where ε ~ N(0, σ²), giving E[Y(d)|X] = m(X,d) (mean-one multiplicative noise).
        
        Args:
            n: Number of samples
            return_potential_outcomes: If True, also return mu0 and mu1 (conditional means)
        
        Returns:
            X_with_d: [n, x_dim+1] array with features and treatment
            y: [n] array of observed outcomes
            mu0, mu1: [n] arrays of conditional means m(X,0) and m(X,1)
                     (only if return_potential_outcomes=True)
        
        Note:
            - Returned potential outcomes are conditional means (noise-free):
              mu0 = m(X,0), mu1 = m(X,1)
            - This is appropriate for ATE since E[Y(d)] = E[m(X,d)] with mean-one noise
            - To compute the true population ATE, use estimate_ate() with a large n_mc.
        """
        correlation_matrix = self._build_correlation_matrix()
        mean = np.zeros(self.x_dim)
        X = self.rng.multivariate_normal(mean, correlation_matrix, size=n)
        if x_scale is not None and x_scale != 1.0:
            X = X * float(x_scale)
        
        # Keep X as correlated normal (no binarization)

        prop = self._propensity_score(X)
        D = self.rng.binomial(n=1, p=prop, size=n)
        D = D.reshape(-1, 1)

        # Multiplicative log-normal noise with mean-one property
        mean_vector = self._outcome_mean(X, D)
        epsilon = self.rng.normal(0, self.sigma, size=n)
        y = mean_vector * np.exp(epsilon - self.sigma**2 / 2)

        X_with_d = np.concatenate([X, D], axis=1)

        if return_potential_outcomes:
            # Return conditional means (noise-free potential outcomes)
            mu0 = self._outcome_mean(X, np.zeros(n))
            mu1 = self._outcome_mean(X, np.ones(n))
            return X_with_d, y, mu0, mu1

        return X_with_d, y

    def estimate_ate(self, n_mc=100000):
        """
        Estimate the true population Average Treatment Effect (ATE).
        
        Args:
            n_mc: Number of Monte Carlo samples (use large value for accurate estimate)
        
        Returns:
            float: Estimated population ATE = E[Y(1) - Y(0)]
        
        Note:
            This uses Monte Carlo integration over the population distribution of X.
            Use a large n_mc (e.g., 100000) for an accurate estimate of the true ATE.
            With mean-one multiplicative noise, ATE = E[m(X,1) - m(X,0)] is independent of σ.
        """
        correlation_matrix = self._build_correlation_matrix()
        mean = np.zeros(self.x_dim)
        X = self.rng.multivariate_normal(mean, correlation_matrix, size=n_mc)
        y0 = self._outcome_mean(X, np.zeros(n_mc))
        y1 = self._outcome_mean(X, np.ones(n_mc))
        return float(np.mean(y1 - y0))


class Sampler_M4:
    """
    Sampler with correlated covariates X, treatment D from a propensity score,
    and nonlinear outcome where noise enters as an input to the function.
    
    Outcome model: Y = m(X, D, ε), where ε ~ N(0, σ²)
    
    Unlike M1-M3 where noise is additive or multiplicative, here ε is an input
    to the neural network, allowing for arbitrary interaction with (X, D).
    
    m(X, D, ε) is a random neural network (not learned).
    """
    def __init__(
        self,
        sigma=1.0,
        correlation=0.5,
        x_dim=20,
        seed=None,
        hidden_dims=(50, 50),
        weight_scale=0.5,
        n_mc_conditional=10000,
    ):
        self.sigma = sigma
        self.correlation = correlation
        self.x_dim = x_dim
        self.rng = np.random.default_rng(seed)
        self.hidden_dims = hidden_dims
        self.weight_scale = weight_scale
        self.n_mc_conditional = n_mc_conditional  # MC samples for computing conditional means
        self._init_outcome_network()

    def _init_outcome_network(self):
        """Initialize neural network for outcome function m(X, D, ε)"""
        input_dim = self.x_dim + 1 + 1  # X plus treatment plus epsilon
        h1, h2 = self.hidden_dims
        
        # Network for outcome function m(X, D, ε)
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

    def _binarize_covariates(self, X):
        """
        Binarize the last 1/3 of covariates.
        
        Args:
            X: [n, x_dim] array of continuous covariates
        
        Returns:
            X_modified: [n, x_dim] array with last 1/3 binarized
        """
        X_modified = X.copy()
        n_binary = self.x_dim // 3  # Last 1/3
        binary_start_idx = self.x_dim - n_binary
        
        # Binarize: 1 if > 0, else 0
        X_modified[:, binary_start_idx:] = (X[:, binary_start_idx:] > 0).astype(float)
        
        return X_modified

    def _propensity_score(self, X):
        """
        Compute propensity score with discontinuities (DISCRETE version, indicator only).
        
        Includes:
        - Indicator functions (jumps/discontinuities) on continuous normal covariates
        - Bounded to [0.1, 0.9] for identifiability
        
        Note: X is kept as correlated normal (not binarized).
              Only indicator jumps, no binary or continuous components.
        """
        def get_col(idx):
            if idx < X.shape[1]:
                return X[:, idx]
            return np.zeros(X.shape[0])
        
        # Discontinuous components (indicators) - BALANCED AROUND 0
        indicator_part = (
            1.2 * (get_col(0) > 0).astype(float)       # Jump at X[0]=0: +1.2 if X[0]>0
            - 1.2 * (get_col(1) < 0).astype(float)     # Jump at X[1]=0: -1.2 if X[1]<0
            + 1.0 * (get_col(2) > 0.5).astype(float)   # Jump at X[2]=0.5: +1.0 if X[2]>0.5
            - 1.0 * (get_col(3) < -0.5).astype(float)  # Jump at X[3]=-0.5: -1.0 if X[3]<-0.5
            + 0.8 * (get_col(4) > 0.5).astype(float)   # Jump at X[4]=0.5: +0.8 if X[4]>0.5
            - 0.8 * (get_col(5) < -0.5).astype(float)  # Jump at X[5]=-0.5: -0.8 if X[5]<-0.5
        )
        
        # Total score (only indicator part)
        score = indicator_part
        
        # Bound to [0.1, 0.9]
        return 0.1 + 0.8 / (1 + np.exp(-score))

    def _outcome_function(self, X, D, epsilon):
        """
        Compute outcome function m(X, D, ε) via random neural network.
        
        Args:
            X: [n, x_dim] array of features
            D: [n] array of treatment
            epsilon: [n] array of noise values
        
        Returns:
            [n] array of outcomes
        """
        D = D.reshape(-1, 1)
        epsilon = epsilon.reshape(-1, 1)
        inputs = np.concatenate([X, D, epsilon], axis=1)
        h1 = np.tanh(inputs @ self.W1 + self.b1)
        h2 = np.tanh(h1 @ self.W2 + self.b2)
        y = h2 @ self.W3 + self.b3
        return y.reshape(-1)

    def _compute_conditional_mean(self, X, D, epsilon_samples=None, n_mc=None, mc_chunk_size=None):
        """
        Compute conditional mean E_ε[m(X, D, ε)] via Monte Carlo (vectorized).
        
        Args:
            X: [n, x_dim] array of features
            D: [n] array of treatment
            epsilon_samples: [n_mc, n] array of pre-drawn epsilon values (for common random numbers)
                           If provided, function is fully deterministic (no RNG used)
                           If None, draws fresh epsilons internally using self.rng
            n_mc: Number of Monte Carlo samples for ε (default: self.n_mc_conditional)
                 Ignored if epsilon_samples is provided
            mc_chunk_size: Process MC samples in chunks to reduce memory (default: None = all at once)
                          Useful for very large n × n_mc (e.g., n=10000, n_mc=500)
        
        Returns:
            [n] array of conditional means
        
        Note:
            Memory usage: O(n × n_mc × x_dim) for tiled arrays.
            For larger scales, use mc_chunk_size to process in batches.
        """
        n = X.shape[0]
        D = np.asarray(D).reshape(-1)
        # Use provided epsilon samples (deterministic) or generate new ones (stochastic)
        if epsilon_samples is not None:
            # Deterministic path: use provided epsilons, no RNG access
            n_mc_actual = epsilon_samples.shape[0]
        else:
            # Stochastic path: generate fresh epsilons using RNG
            if n_mc is None:
                n_mc_actual = self.n_mc_conditional
            else:
                n_mc_actual = n_mc
            epsilon_samples = self.rng.normal(0, self.sigma, size=(n_mc_actual, n))
        
        # Chunked processing for memory efficiency (if requested)
        if mc_chunk_size is not None and mc_chunk_size < n_mc_actual:
            mu = np.zeros(n)
            n_chunks = (n_mc_actual + mc_chunk_size - 1) // mc_chunk_size
            
            for chunk_idx in range(n_chunks):
                start_mc = chunk_idx * mc_chunk_size
                end_mc = min(start_mc + mc_chunk_size, n_mc_actual)
                eps_chunk = epsilon_samples[start_mc:end_mc]
                
                # Process this chunk
                n_mc_chunk = eps_chunk.shape[0]
                X_tiled = np.tile(X, (n_mc_chunk, 1))
                D_tiled = np.tile(D, n_mc_chunk)
                epsilon_flat = eps_chunk.flatten()
                
                y_all = self._outcome_function(X_tiled, D_tiled, epsilon_flat)
                y_matrix = y_all.reshape(n_mc_chunk, n)
                
                # Accumulate weighted sum (for averaging later)
                mu += y_matrix.sum(axis=0)
            
            mu /= n_mc_actual
            return mu
        
        # Standard vectorized computation (all MC samples at once):
        # Tile X and D to [n_mc * n, ...] shape
        # Process all MC samples in one forward pass
        # 
        # Layout choice: MC-first, unit varying faster
        # X_tiled: [X[0], X[1], ..., X[n-1], X[0], X[1], ..., X[n-1], ...]
        #          [------- mc=0 -------][------- mc=1 -------]...
        # D_tiled: [D[0], D[1], ..., D[n-1], D[0], D[1], ..., D[n-1], ...]
        # ε_flat:  [ε[0,0], ε[0,1], ..., ε[0,n-1], ε[1,0], ε[1,1], ..., ε[1,n-1], ...]
        X_tiled = np.tile(X, (n_mc_actual, 1))   # [n_mc*n, x_dim]
        D_tiled = np.tile(D, n_mc_actual)         # [n_mc*n]
        epsilon_flat = epsilon_samples.flatten()  # [n_mc*n] - row-major flatten
        
        # Single forward pass for all MC samples
        # y_all[i] = m(X[i % n], D[i % n], ε[i // n, i % n])
        y_all = self._outcome_function(X_tiled, D_tiled, epsilon_flat)  # [n_mc*n]
        
        # Reshape to [n_mc, n] to get MC samples × units
        # Row i contains: [y(X[0],...,ε[i,0]), y(X[1],...,ε[i,1]), ..., y(X[n-1],...,ε[i,n-1])]
        y_matrix = y_all.reshape(n_mc_actual, n)  # [n_mc, n]
        
        # Average over MC samples (axis=0) to get conditional mean for each unit
        mu = y_matrix.mean(axis=0)  # [n]
        
        return mu

    def sample(self, n=1000, return_potential_outcomes=False, x_scale=1.0):
        """
        Sample n observations from the data generating process.
        
        Outcome model: Y = m(X, D, ε), where ε ~ N(0, σ²) is an input to the function.
        
        Args:
            n: Number of samples
            return_potential_outcomes: If True, also return mu0 and mu1 (conditional means)
        
        Returns:
            X_with_d: [n, x_dim+1] array with features and treatment
            y: [n] array of observed outcomes
            mu0, mu1: [n] arrays of conditional means E_ε[m(X,0,ε)] and E_ε[m(X,1,ε)]
                     (only if return_potential_outcomes=True)
        
        Note:
            - Conditional means are computed via Monte Carlo over ε:
              mu0 = E_ε[m(X,0,ε)], mu1 = E_ε[m(X,1,ε)]
            - Uses common random numbers (same ε draws) for mu0 and mu1 to:
              * Match the SCM coupling where ε is a unit-level property
              * Reduce Monte Carlo noise for individual-level effect estimation
              * Enable meaningful unit-level comparisons of Y(1) vs Y(0)
            - Number of MC samples controlled by n_mc_conditional parameter
            - ATE = E_X[mu1 - mu0] by linearity of expectation
            - To compute the true population ATE, use estimate_ate() with a large n_mc.
        """
        correlation_matrix = self._build_correlation_matrix()
        mean = np.zeros(self.x_dim)
        X = self.rng.multivariate_normal(mean, correlation_matrix, size=n)
        if x_scale is not None and x_scale != 1.0:
            X = X * float(x_scale)
        
        # Keep X as correlated normal (no binarization)

        prop = self._propensity_score(X)
        D = self.rng.binomial(n=1, p=prop, size=n)
        D = D.reshape(-1, 1).flatten()

        # Generate noise for observed outcomes
        epsilon = self.rng.normal(0, self.sigma, size=n)
        y = self._outcome_function(X, D, epsilon)

        X_with_d = np.concatenate([X, D.reshape(-1, 1)], axis=1)

        if return_potential_outcomes:
            # Compute conditional means via Monte Carlo over ε
            # Use common random numbers (same epsilon draws) for mu0 and mu1
            # to match SCM coupling and reduce MC noise for individual-level effects
            epsilon_samples = self.rng.normal(0, self.sigma, size=(self.n_mc_conditional, n))
            mu0 = self._compute_conditional_mean(X, np.zeros(n), epsilon_samples=epsilon_samples)
            mu1 = self._compute_conditional_mean(X, np.ones(n), epsilon_samples=epsilon_samples)
            return X_with_d, y, mu0, mu1

        return X_with_d, y

    def estimate_ate(self, n_mc=100000):
        """
        Estimate the true population Average Treatment Effect (ATE).
        
        Args:
            n_mc: Number of Monte Carlo samples (use large value for accurate estimate)
        
        Returns:
            float: Estimated population ATE = E[Y(1) - Y(0)]
        
        Note:
            This uses Monte Carlo integration over the joint distribution of (X, ε).
            Since X and ε are independent, we sample them together:
            ATE = E_X,ε[m(X,1,ε) - m(X,0,ε)]
            
            Uses the same ε for Y(0) and Y(1) to match SCM coupling and reduce variance.
            Use a large n_mc (e.g., 100000) for an accurate estimate of the true ATE.
        """
        correlation_matrix = self._build_correlation_matrix()
        mean = np.zeros(self.x_dim)
        X = self.rng.multivariate_normal(mean, correlation_matrix, size=n_mc)
        epsilon = self.rng.normal(0, self.sigma, size=n_mc)
        
        # Use same epsilon for both potential outcomes (CRN for variance reduction)
        y0 = self._outcome_function(X, np.zeros(n_mc), epsilon)
        y1 = self._outcome_function(X, np.ones(n_mc), epsilon)
        
        return float(np.mean(y1 - y0))


class Sampler_UM1:
    """
    Sampler with unmeasured confounders H that affect X, D, and Y.

    H ~ N(0, I_q)
    X = H @ W_xh + x_noise * epsilon_x
    P(D=1|X,H) = 1 / ((1 + exp(f(X))) * (1 + exp(g(H)))), clipped to [0.1, 0.9]
    Y = m(X, D) + g_y(H) + sigma * epsilon_y

    m(X, D) is a random neural network (not learned).
    """
    def __init__(
        self,
        sigma=1.0,
        x_dim=30,
        q=2,
        delta=0.1,
        x_noise=0.5,
        seed=None,
        hidden_dims=(50, 50),
        weight_scale=0.5,
    ):
        self.sigma = sigma
        self.x_dim = x_dim
        self.q = q
        self.delta = delta
        self.x_noise = x_noise
        self.rng = np.random.default_rng(seed)
        self.hidden_dims = hidden_dims
        self.weight_scale = weight_scale

        self._init_outcome_network()
        self._init_confounding_structure()

    def _init_outcome_network(self):
        """Initialize neural network for mean function m(X,D)."""
        input_dim = self.x_dim + 1  # X plus treatment
        h1, h2 = self.hidden_dims
        self.W1 = self.rng.normal(0, self.weight_scale, size=(input_dim, h1))
        self.b1 = self.rng.normal(0, self.weight_scale, size=(h1,))
        self.W2 = self.rng.normal(0, self.weight_scale, size=(h1, h2))
        self.b2 = self.rng.normal(0, self.weight_scale, size=(h2,))
        self.W3 = self.rng.normal(0, self.weight_scale, size=(h2, 1))
        self.b3 = self.rng.normal(0, self.weight_scale, size=(1,))

    def _init_confounding_structure(self):
        """Initialize linear coefficients linking H to X, D, and Y."""
        self.W_xh = self.rng.uniform(1 - self.delta, 1 + self.delta, size=(self.q, self.x_dim))
        self.w_dx = self.rng.normal(0, self.weight_scale, size=(self.x_dim,))
        self.w_dh = self.rng.normal(0, self.weight_scale, size=(self.q,))
        self.w_yh = self.rng.normal(0, self.weight_scale, size=(self.q,))

    def _outcome_mean(self, X, D):
        """Compute mean function m(X,D) via random neural network."""
        D = D.reshape(-1, 1)
        inputs = np.concatenate([X, D], axis=1)
        h1 = np.tanh(inputs @ self.W1 + self.b1)
        h2 = np.tanh(h1 @ self.W2 + self.b2)
        y = h2 @ self.W3 + self.b3
        return y.reshape(-1)

    def _propensity_score(self, X, H):
        """Compute propensity scores using linear functions of X and H."""
        f_x = X @ self.w_dx
        g_h = H @ self.w_dh
        prop = 1.0 / ((1.0 + np.exp(f_x)) * (1.0 + np.exp(g_h)))
        return np.clip(prop, 0.1, 0.9)

    def _sample_h(self, n):
        return self.rng.standard_normal((n, self.q))

    def _sample_x(self, H):
        noise = self.rng.standard_normal((H.shape[0], self.x_dim))
        return H @ self.W_xh + self.x_noise * noise

    def sample(self, n=1000, return_potential_outcomes=False):
        """
        Sample n observations from the data generating process.

        Returns:
            H: [n, q] array of unmeasured confounders
            X_with_d: [n, x_dim+1] array with features and treatment
            y: [n] array of observed outcomes
            mu0, mu1: [n] arrays of conditional means (if return_potential_outcomes=True)
        """
        H = self._sample_h(n)
        X = self._sample_x(H)
        prop = self._propensity_score(X, H)
        D = self.rng.binomial(n=1, p=prop, size=n)

        mean_vector = self._outcome_mean(X, D)
        y = mean_vector + (H @ self.w_yh) + self.sigma * self.rng.standard_normal(n)

        X_with_d = np.concatenate([X, D.reshape(-1, 1)], axis=1)

        if return_potential_outcomes:
            h_term = H @ self.w_yh
            mu0 = self._outcome_mean(X, np.zeros(n)) + h_term
            mu1 = self._outcome_mean(X, np.ones(n)) + h_term
            return H, X_with_d, y, mu0, mu1

        return H, X_with_d, y

    def estimate_ate(self, n_mc=100000):
        """
        Estimate the true population Average Treatment Effect (ATE).

        Note:
            g_y(H) cancels in Y(1) - Y(0), so ATE depends only on m(X, D).
        """
        H = self._sample_h(n_mc)
        X = self._sample_x(H)
        y0 = self._outcome_mean(X, np.zeros(n_mc))
        y1 = self._outcome_mean(X, np.ones(n_mc))
        return float(np.mean(y1 - y0))



