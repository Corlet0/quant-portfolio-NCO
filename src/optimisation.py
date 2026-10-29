#####################################################################################################################################
###########################################               INNER OPTIMISATION               ##########################################
#####################################################################################################################################

def solve_inner_min_variance(sub_cov: np.ndarray) -> np.ndarray:
    """
    Solves the long-only Minimum Variance portfolio on an isolated covariance sub-block.

    Parameters:
        sub_cov (np.ndarray): Local covariance block (|k| x |k|) for a given cluster.

    Returns:
        np.ndarray: Optimal intra-cluster weight vector summing to 1.0.
    """
    n = sub_cov.shape[0]

    # Trivial base case: a single-asset cluster receives 100% weight
    if n == 1:
        return np.array([1.0])

    w = cp.Variable(n)

    # cp.psd_wrap prevents DCP errors caused by floating-point precision artifacts
    risk = cp.quad_form(w, cp.psd_wrap(sub_cov))

    # Budget and long-only non-negativity constraints
    constraints = [
        cp.sum(w) == 1.0,
        w >= 0.0
    ]

    problem = cp.Problem(cp.Minimize(risk), constraints)

    try:
        problem.solve(solver=cp.CLARABEL)

        if w.value is None:
            raise ValueError("Solver failed to converge.")

        # Eliminate tiny floating-point negatives and re-normalize
        weights = np.maximum(0.0, w.value)
        return weights / np.sum(weights)

    except Exception:
        # Fallback: Inverse-variance allocation to ensure uninterrupted execution
        inv_diag = 1.0 / np.clip(np.diag(sub_cov), 1e-8)
        return inv_diag / np.sum(inv_diag)

#-------------------------------------------------------------------------------------------##-------------------------------------------------------------------------------------------#

def solve_inner_equal_risk_contribution(sub_cov: np.ndarray) -> np.ndarray:
    """
    Solves the long-only equal risk contribution portfolio on an isolated covariance sub-block.

    Parameters:
        sub_cov (np.ndarray): Local covariance block (|k| x |k|) for a given cluster.

    Returns:
        np.ndarray: Optimal intra-cluster weight vector summing to 1.0.
    """
    n = sub_cov.shape[0]

    # Trivial base case: a single-asset cluster receives 100% weight
    if n == 1:
        return np.array([1.0])

    # 1. Extract the diagonal variances and compute standard deviations (volatilities)
    vols = np.sqrt(np.clip(np.diag(sub_cov), 1e-8))

    # 2. Compute inverse-volatility weights
    inv_vols = 1.0 / vols

    # 3. Normalize weights to sum to 1.0
    return inv_vols / np.sum(inv_vols)

#-------------------------------------------------------------------------------------------##-------------------------------------------------------------------------------------------#

def inner_cluster_weights(cov_clean: np.ndarray, cluster_ids: np.ndarray, method: str):
    """
    Builds the (N x K) block-diagonal intra-cluster weight matrix V.

    Parameters:
        cov_clean (np.ndarray): Full N x N cleaned/re-toned covariance matrix.
        cluster_ids (np.ndarray): 1D array of length N containing 0-indexed cluster IDs.
        method (str): How the optimisation is computed. 'MV' for Minimum Variance; 'ERC' for Equal Risk Contribution

    Returns:
        np.ndarray: Block-diagonal weight matrix V of shape (N, K).
    """
    n_assets = cov_clean.shape[0]
    unique_clusters = np.unique(cluster_ids)
    n_clusters = len(unique_clusters)

    # Initialize the global block-diagonal matrix V
    V = np.zeros((n_assets, n_clusters))

    for k in range(n_clusters):
        # 1. Use the cluster IDs as a logical mask to find the asset indices
        asset_indices = np.where(cluster_ids == k)[0]

        # 2. Slice the local sub-covariance block dynamically
        sub_cov = cov_clean[np.ix_(asset_indices, asset_indices)]

        # 3. Run inner optimization
        if method == 'MV':
          w_cluster = solve_inner_min_variance(sub_cov)
        elif method == 'ERC':
            w_cluster = solve_inner_equal_risk_contribution(sub_cov)
        else:
            raise ValueError(f"Unknown optimization method: {method}. Use 'MV' or 'ERC'.")
        
        # 4. Map the local weights back to their global positions in V
        V[asset_indices, k] = w_cluster

    return V

#####################################################################################################################################
###########################################               OUTER OPTIMISATION               ##########################################
#####################################################################################################################################

def compute_shrunk_expected_returns(
    returns: pd.DataFrame,
    lookback_days: int = 126,
    daily_risk_free_rate: float = 0.0
) -> pd.Series:
    """
    Computes expected asset returns adjusted via a Sortino ratio momentum filter
    and shrunk toward the cross-sectional grand mean using James-Stein estimation.

    Parameters:
        returns (pd.DataFrame): Historical simple returns (Dates x Assets).
        lookback_days (int): Historical window size in trading days (default is 126 ~ 6 months).
        daily_risk_free_rate (float): Daily risk-free benchmark rate (default is 0.0).

    Returns:
        pd.Series: Expected return estimates per asset shrunk by James-Stein estimation.
    """
    returns_window = returns.tail(lookback_days)
    n_observations, n_assets = returns_window.shape

    # 1. Momentum & downside risk (Sortino ratio)
    mean_daily_returns = returns_window.mean(axis=0).values
    negative_returns = np.minimum(0.0, returns_window.values - daily_risk_free_rate)
    downside_variance = np.sum(negative_returns**2, axis=0) / (n_observations - 1)
    downside_deviation = np.sqrt(np.clip(downside_variance, 1e-8, None))

    sortino_ratio = (mean_daily_returns - daily_risk_free_rate) / downside_deviation
    raw_signals = mean_daily_returns * np.maximum(0.1, sortino_ratio)

    # 2. James-Stein Shrinkage toward the market grand mean
    grand_mean = np.mean(raw_signals)
    mean_sampling_variance = np.mean(np.var(returns_window.values, axis=0, ddof=1)) / n_observations
    signal_dispersion = np.sum((raw_signals - grand_mean)**2)

    if signal_dispersion > 0 and n_assets > 2:
        shrinkage_intensity = ((n_assets - 2) * mean_sampling_variance) / signal_dispersion
        shrinkage_intensity = np.clip(shrinkage_intensity, 0.0, 1.0)
    else:
        shrinkage_intensity = 1.0

    shrunk_expected_returns = (
        (1.0 - shrinkage_intensity) * raw_signals + shrinkage_intensity * grand_mean
    )

    return pd.Series(
        shrunk_expected_returns,
        index=returns.columns,
        name="shrunk_expected_returns"
    )

#-------------------------------------------------------------------------------------------##-------------------------------------------------------------------------------------------#

def solve_outer_cluster_weights(
    cov_retoned: np.ndarray,
    inner_weight_matrix: np.ndarray,
    asset_expected_returns: np.ndarray,
    risk_aversion: float = 0.5,
    solver=cp.CLARABEL
) -> tuple[np.ndarray, np.ndarray]:
    """
    Solves the inter-cluster optimization (outer loop) in the reduced K-dimensional
    cluster space and projects allocations back to synthesize final asset weights.

    Parameters:
        cov_retoned (np.ndarray): Cleaned global covariance matrix (N x N)[cite: 3].
        inner_weight_matrix (np.ndarray): Block-diagonal intra-cluster weight matrix V (N x K).
        asset_expected_returns (np.ndarray): Expected return vector across all N individual assets (N,).
        risk_aversion (float): Trade-off parameter balancing risk minimization against return maximization.
        solver: CVXPY-compatible conic optimization solver.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - final_asset_weights: Synthesized asset allocations across all N individual securities (N,).
            - outer_cluster_weights: Optimal capital allocation across the K synthetic clusters (K,).
    """
    n_clusters = inner_weight_matrix.shape[1]

    # 1. Project global covariance and expected returns into K-dimensional cluster space
    cluster_covariance = inner_weight_matrix.T @ cov_retoned @ inner_weight_matrix
    cluster_expected_returns = inner_weight_matrix.T @ asset_expected_returns

    # 2. Formulate convex quadratic program for inter-cluster allocation
    outer_weights = cp.Variable(n_clusters)
    cluster_portfolio_return = cluster_expected_returns @ outer_weights
    cluster_portfolio_risk = cp.quad_form(outer_weights, cp.psd_wrap(cluster_covariance))

    objective = cp.Minimize(0.5 * cluster_portfolio_risk - risk_aversion * cluster_portfolio_return)
    constraints = [
        cp.sum(outer_weights) == 1.0,
        outer_weights >= 0.0
    ]

    optimization_problem = cp.Problem(objective, constraints)

    try:
        optimization_problem.solve(solver=solver)
        if outer_weights.value is None:
            raise ValueError("Solver failed to converge.")

        cleaned_outer_weights = np.maximum(0.0, outer_weights.value)
        optimal_cluster_weights = cleaned_outer_weights / np.sum(cleaned_outer_weights)

    except Exception as error:
        print(f"Warning: Outer optimization solver failure ({error}). Defaulting to equal cluster weights.")
        optimal_cluster_weights = np.ones(n_clusters) / n_clusters

    # 3. Synthesize final asset weights: w_final = V @ w_outer[cite: 7, 8]
    final_asset_weights = inner_weight_matrix @ optimal_cluster_weights

    return final_asset_weights, optimal_cluster_weights

#-------------------------------------------------------------------------------------------##-------------------------------------------------------------------------------------------#

def run_nco_outer_pipeline(
    returns: pd.DataFrame,
    cov_retoned: np.ndarray,
    inner_weight_matrix: np.ndarray,
    lookback_days: int = 126,
    daily_risk_free_rate: float = 0.0,
    risk_aversion: float = 0.5
) -> pd.Series:
    """
    Coordinates expected return estimation, outer-loop cluster allocation,
    and asset-level synthesis using simple returns directly.

    Parameters:
        returns (pd.DataFrame): Daily historical return series (Dates x Assets).
        cov_retoned (np.ndarray): Cleaned, re-scaled global covariance matrix (N x N)[cite: 3].
        inner_weight_matrix (np.ndarray): Block-diagonal matrix V containing intra-cluster weights (N x K)[cite: 7, 8].
        lookback_days (int): Estimation window for expected returns.
        daily_risk_free_rate (float): Daily benchmark rate.
        risk_aversion (float): Mean-variance risk aversion multiplier.

    Returns:
       tuple[pd.Series, np.ndarray]:
        - optimal_portfolio_weights: Final portfolio weights indexed by asset tickers summing to 1.0.
        - outer_cluster_weights: Optimal capital allocation across the K synthetic clusters (K,).
    """
    expected_returns_series = compute_shrunk_expected_returns(
        returns=returns,
        lookback_days=lookback_days,
        daily_risk_free_rate=daily_risk_free_rate
    )

    final_weights, optimal_cluster_weights = solve_outer_cluster_weights(
        cov_retoned=cov_retoned,
        inner_weight_matrix=inner_weight_matrix,
        asset_expected_returns=expected_returns_series.values,
        risk_aversion=risk_aversion
    )

    return pd.DataFrame(final_weights, index=returns.columns), optimal_cluster_weights
