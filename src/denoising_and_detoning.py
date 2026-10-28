import numpy as np
import pandas as pd

def denoise_matrix(raw_returns: pd.DataFrame) -> np.ndarray:
    """
    Filters an empirical correlation matrix using Random Matrix Theory (RMT)
    and the Marchenko-Pastur clipping method.

    Parameters:
        raw_returns (pd.DataFrame): T observations x N assets DataFrame of unstandardized asset returns.

    Returns:
        np.ndarray: The filtered, noise-reduced correlation matrix.
    """
    # Get number of observations (T) and number of assets (N)
    T, N = raw_returns.shape

    # Compute the empirical correlation matrix internally (inherently standardized)
    empirical_correlation = raw_returns.corr().values

    # Calculate the quality factor Q (ratio of observations to assets)
    Q = T / N

    # Step 1: Perform spectral decomposition (eigenvalues and eigenvectors)
    eigenvalues, eigenvectors = np.linalg.eigh(empirical_correlation)

    # Sort eigenvalues and eigenvectors in descending order
    idx = eigenvalues.argsort()[::-1]
    sorted_eigenvalues = eigenvalues[idx]
    sorted_eigenvectors = eigenvectors[:, idx]

    # Step 2: Apply the Marchenko-Pastur theoretical upper bound
    marchenko_pastur_upper_bound = 1.0 * (1.0 + np.sqrt(1.0 / Q))**2

    # Step 3: Identify noise (eigenvalues below or equal to the theoretical threshold)
    noise_idx = sorted_eigenvalues <= marchenko_pastur_upper_bound

    # Step 4: Calculate the average of the noise eigenvalues (clipping level)
    if noise_idx.any():
        average_noise_level = sorted_eigenvalues[noise_idx].mean()
    else:
        average_noise_level = 0.0

    # Step 5: Create the cleaned eigenvalue array by replacing noise with the average
    cleaned_eigenvalues = sorted_eigenvalues.copy()
    cleaned_eigenvalues[noise_idx] = average_noise_level

    # Step 6: Reconstruct the cleaned correlation matrix
    reconstructed_correlation = sorted_eigenvectors @ np.diag(cleaned_eigenvalues) @ sorted_eigenvectors.T

    # Set diagonal elements to exactly 1.0 using proper mathematical rescaling
    diag_factor = 1.0 / np.sqrt(np.diag(reconstructed_correlation))
    denoised_corr = np.outer(diag_factor, diag_factor) * reconstructed_correlation
    np.fill_diagonal(denoised_corr, 1.0)
    
    return denoised_corr

#-------------------------------------------------------------------------------------------#

def detone_matrix(corr_matrix: np.ndarray, num_market_factors: int = 1) -> np.ndarray:
    """
    Removal of the primary market eigenvector to isolate idiosyncratic correlations.
    
    Parameters:
        corr_matrix (np.ndarray): A denoised N x N correlation matrix.
        num_market_factors (int): Number of dominant market factors to remove. Default is 1.
    
    Returns:
        np.ndarray: The detoned correlation matrix.
    """
    # 1. Finding matrix eigenvalues, eigenvectors (decomposition)
    eigenvalues, eigenvectors = np.linalg.eigh(corr_matrix)

    # Sort eigenvalues and eigenvectors in descending order
    idx = eigenvalues.argsort()[::-1]
    sorted_eigenvalues = eigenvalues[idx]
    sorted_eigenvectors = eigenvectors[:, idx]

    # 2. Subtracting the market risk from the original matrix (parameterized)
    market_risk = np.zeros_like(corr_matrix)
    for i in range(num_market_factors):
        market_risk += sorted_eigenvalues[i] * np.outer(sorted_eigenvectors[:, i], sorted_eigenvectors[:, i])

    detoned_matrix = corr_matrix - market_risk

    # 3. Rescaling and normalizing using the diagonal elements
    d = 1.0 / np.sqrt(np.diag(detoned_matrix))
    correlation_detoned = np.outer(d, d) * detoned_matrix
    np.fill_diagonal(correlation_detoned, 1.0)

    return correlation_detoned
      
#-------------------------------------------------------------------------------------------#

def denoise_and_detone(
    raw_returns: pd.DataFrame, 
    detone: bool = True, 
    num_market_factors: int = 1
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Coordinates the full cleaning pipeline by passing raw returns to the denoising module, 
    isolating idiosyncratic relations (detoning), and reconstructing the filtered covariance matrix.

    Parameters:
        raw_returns (pd.DataFrame): T x N DataFrame of unstandardized asset returns.
        detone (bool): Whether to apply market detoning. Default is True.
        num_market_factors (int): Number of leading eigenvectors to eliminate if detoning. Default is 1

    Returns:
        A tuple containing:
            - pd.DataFrame: The cleaned, noise-reduced correlation matrix (N x N).
            - pd.DataFrame: The re-toned, noise-reduced covariance matrix (N x N).
    """
    # 1. Denoise matrix 
    corr_denoised = denoise_matrix(raw_returns)
    
    # 2. Detone denoised matrix
    if detone:
        corr_detoned = detone_matrix(corr_denoised, num_market_factors)
    else:
        corr_detoned = corr_denoised

    # 3. Extract empirical volatilities for the re-toning phase
    empirical_volatilities = np.sqrt(np.diag(raw_returns.cov().values))
    volatility_diagonal_matrix = np.diag(empirical_volatilities)

    # 4. Scale the correlation matrix back to covariance: D * corr_detoned * D
    cov_retoned = volatility_diagonal_matrix @ corr_detoned @ volatility_diagonal_matrix

    # 5. Convert back to Pandas DataFrames with original asset labels
    corr_df = pd.DataFrame(corr_detoned, index=raw_returns.columns, columns=raw_returns.columns)
    cov_df = pd.DataFrame(cov_retoned, index=raw_returns.columns, columns=raw_returns.columns)

    return corr_df, cov_df
