def denoise_matrix(standarized_returns: pd.DataFrame) -> pd.DataFrame: 
    """
    Filters an empirical covariance matrix using Random Matrix Theory (RMT)
    and the Marchenko-Pastur clipping method.

    Parameters:
        standarized_returns: pd.DataFrame (T days x N assets) of standardized asset returns.
    
    Returns:
        pd.DataFrame: The filtered, noise-reduced correlation matrix.
    """
    # Get number of observations (T) and number of assets (N)
    T, N = standarized_returns.shape

    # Calculate the quality factor Q (ratio of observations to assets)
    Q = T / N

    # Step 1: Compute the empirical correlation matrix
    empirical_correlation = standarized_returns.corr()

    # Step 2: Perform spectral decomposition (eigenvalues and eigenvectors)
    # np.linalg.eigh is optimized for symmetric matrices like correlation matrices
    eigenvalues, eigenvectors = np.linalg.eigh(empirical_correlation.values)

    # Sort eigenvalues and eigenvectors in descending order
    idx = eigenvalues.argsort()[::-1]
    sorted_eigenvalues = eigenvalues[idx]
    sorted_eigenvectors = eigenvectors[:, idx]

    # Step 3: Apply the Marchenko-Pastur theoretical upper bound
    # Since we are working with correlation, the theoretical variance is 1
    marchenko_pastur_upper_bound = 1 * (1 + np.sqrt(1 / Q))**2

    # Step 4: Identify noise (eigenvalues below or equal to the theoretical threshold)
    noise_idx = sorted_eigenvalues <= marchenko_pastur_upper_bound

    # Step 5: Calculate the average of the noise eigenvalues (clipping level)
    if noise_idx.any():
        average_noise_level = sorted_eigenvalues[noise_idx].mean()
    else:
        average_noise_level = 0.0

    # Step 6: Create the cleaned eigenvalue array by replacing noise with the average
    cleaned_eigenvalues = sorted_eigenvalues.copy()
    cleaned_eigenvalues[noise_idx] = average_noise_level

    # Step 7: Reconstruct the cleaned correlation matrix
    reconstructed_correlation = sorted_eigenvectors @ np.diag(cleaned_eigenvalues) @ sorted_eigenvectors.T

    # Set diagonal elements to exactly 1.0 (an asset's correlation with itself is always 1)
    np.fill_diagonal(reconstructed_correlation, 1.0)

    # Convert back to a Pandas DataFrame with original asset labels
    cleaned_correlation_df = pd.DataFrame(
        reconstructed_correlation,
        index=empirical_correlation.index,
        columns=empirical_correlation.columns
    )

    return cleaned_correlation_df

#-------------------------------------------------------------------------------------------#

def detone_matrix(corr_matrix: pd.DataFrame) -> pd.DataFrame:
  """
  Removal of the primary market eigenvector to isolate idiosyncratic correlations.
  
  Parameters:
      corr_matrix: pd.DataFrame a NxN correlation matrix.
  
  Returns:
      pd.DataFrame: The detoned correlation matrix.
  """
  # 1. Finding matrix eigenvalues, eigenvectors (decomposition)
  eigenvalues, eigenvectors = np.linalg.eigh(corr_matrix.values)

  # Sort eigenvalues and eigenvectors in descending order
  idx = eigenvalues.argsort()[::-1]
  sorted_eigenvalues = eigenvalues[idx]
  sorted_eigenvectors = eigenvectors[:, idx]

  # 2. Identifying biggest Eigenvalue and its Eigenvector
  lambda_1 = sorted_eigenvalues[0]
  v_1 = sorted_eigenvectors[:, 0]

  # 3. Calculating market systematic risk (tone matrix) using the outer product
  market_risk = lambda_1 * np.outer(v_1, v_1)

  # 4. Subtracting the market risk from the original matrix
  detoned_matrix = corr_matrix.values - market_risk

  # 5. Rescaling and normalizing using the diagonal elements
  d = np.sqrt(np.diag(detoned_matrix))
  correlation_detoned = detoned_matrix / np.outer(d, d)
  np.fill_diagonal(correlation_detoned, 1.0)

  return pd.DataFrame(
      correlation_detoned,
      index=corr_matrix.index,
      columns=corr_matrix.columns
  )

#-------------------------------------------------------------------------------------------#

def reconstruct_and_retone(standarized_returns: pd.DataFrame, empirical_volatilities: pd.Series) -> pd.DataFrame:
  """
Coordinates the full cleaning pipeline by denoising the empirical correlation matrix, 
isolating idiosyncratic relations (detoning), and reconstructing the filtered covariance matrix.

Parameters:
    standarized_returns (pd.DataFrame): T x N DataFrame of standardized asset returns.
    empirical_volatilities (pd.Series): Series containing the historical standard deviation of each asset.

Returns:
    tuple: A tuple containing:
        - pd.DataFrame: The detoned, noise-reduced correlation matrix (N x N).
        - pd.DataFrame: The re-toned, noise-reduced covariance matrix (N x N).
"""
  # denoise matrix
  corr_denoised = denoise_matrix(standarized_returns)
  
  #detone denoised matrix
  corr_detoned = detone_matrix(corr_denoised)

  # Convert volatilities series into a diagonal matrix
  volatility_diagonal_matrix = np.diag(empirical_volatilities.values)

  # Scale the correlation matrix back to covariance: D * corr_denoised * D
  cov_denoised = volatility_diagonal_matrix @ corr_denoised.values @ volatility_diagonal_matrix

  return corr_detoned, cov_denoised

