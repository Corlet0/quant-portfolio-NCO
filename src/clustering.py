"""
src/clustering.py
Hierarchical clustering engine:
- Correlation to Euclidean distance transformation
- Ward's minimum-variance linkage construction
- Optimal K extraction via Silhouette Coefficient maximization
- Dendrogram visualization
"""

def correlation_to_distance(corr_matrix):
    """
    Transforms a correlation matrix into a proper Euclidean distance metric bounded in [0, 1].
    d_ij = sqrt(0.5 * (1 - rho_ij))
    """
    dist = np.sqrt(np.clip(0.5 * (1.0 - corr_matrix), 0.0, 1.0))
    np.fill_diagonal(dist, 0.0)
    
    return dist

#------------------------------------------------------------------------------------------------#

def build_linkage_matrix(corr_matrix, method="ward"):
    """
    Calculates distance matrix and computes hierarchical agglomerative linkage.
    """
    dist_matrix = correlation_to_distance(corr_matrix)
    condensed_dist = squareform(dist_matrix, checks=False)
    linkage_tree = linkage(condensed_dist, method=method)
    
    return dist_matrix, linkage_tree

#------------------------------------------------------------------------------------------------#

def find_optimal_clusters(corr_matrix, min_clusters=2, max_clusters=None, method="ward"):
    """
    Evaluates candidate partitions to find the optimal cluster count K
    that maximizes the global mean Silhouette score.
    """
    n_assets = corr_matrix.shape[0]

    # Upper bound default: evaluate up to half the universe size
    if max_clusters is None:
        max_clusters = max(min_clusters + 1, n_assets // 2)
    else:
        max_clusters = min(max_clusters, n_assets - 1)

    dist_matrix, linkage_tree = build_linkage_matrix(corr_matrix, method=method)

    best_score = -1.0
    optimal_num_clusters = min_clusters
    asset_cluster_ids = None

    # Unsupervised search across the candidate cluster space
    for k in range(min_clusters, max_clusters + 1):
        # 'maxclust' cuts the dendrogram into exactly k flat clusters
        labels = fcluster(linkage_tree, t=k, criterion="maxclust") - 1

        # 'metric="precomputed"' evaluates cohesion and separation directly from dist_matrix
        score = silhouette_score(dist_matrix, labels, metric="precomputed")

        if score > best_score:
            best_score = score
            optimal_num_clusters = k
            asset_cluster_ids = labels

    return asset_cluster_ids, optimal_num_clusters, linkage_tree

#------------------------------------------------------------------------------------------------#

def get_cluster_sub_covariances(denoised_cov: np.ndarray, asset_to_cluster: dict) -> dict:
    """
    Extracts the localized sub-covariance matrices for all clusters.
    
    Parameters:
        denoised_cov (np.ndarray or pd.DataFrame): Global denoised covariance matrix.
        asset_to_cluster (dict): Mapping of {asset_name: cluster_id}.
        
    Returns:
        dict: A dictionary mapping cluster_id -> pd.DataFrame of the sliced covariance matrix.
    """
    assets_list = list(asset_to_cluster.keys())
    
    # Convert denoised_cov to a DataFrame if it's currently a numpy array
    if isinstance(denoised_cov, np.ndarray):
        denoised_cov_df = pd.DataFrame(denoised_cov, index=assets_list, columns=assets_list)
    else:
        denoised_cov_df = denoised_cov

    cluster_sub_covariances = {}
    unique_clusters = np.unique(list(asset_to_cluster.values()))

    for cluster_id in unique_clusters:
        # Identify all assets assigned to the current cluster
        cluster_assets = [asset for asset, cid in asset_to_cluster.items() if cid == cluster_id]
        
        # Slice and store the sub-covariance matrix
        cluster_sub_covariances[cluster_id] = denoised_cov_df.loc[cluster_assets, cluster_assets]

    return cluster_sub_covariances

#------------------------------------------------------------------------------------------------#

def get_hierarchical_sub_covariances(corr_matrix: pd.DataFrame, denoised_cov: np.ndarray, min_clusters=2, max_clusters=None, method="ward") -> tuple:
    """
    High-level coordination function that finds the optimal clusters, maps the assets,
    and slices the denoised covariance matrix into localized sub-covariance matrices.

    Parameters:
        corr_matrix (pd.DataFrame): The N x N correlation matrix.
        denoised_cov (np.ndarray or pd.DataFrame): Global denoised covariance matrix.
        min_clusters (int): Minimum number of clusters to evaluate.
        max_clusters (int): Maximum number of clusters to evaluate.
        method (str): Linkage method.

    Returns:
        tuple: (asset_to_cluster_dict, cluster_sub_covariances_dict)
    """
    # 1. Extract the optimal clusters
    optimal_clusters_id, _, _ = find_optimal_clusters(corr_matrix.values, min_clusters, max_clusters, method)
    
    # 2. Build the mapping dictionary internally
    mapping_series = pd.Series(optimal_clusters_id, index=corr_matrix.columns)
    asset_to_cluster_dict = mapping_series.to_dict()
    
    # 3. Slice and generate the sub-covariances
    sub_covariances = get_cluster_sub_covariances(denoised_cov, asset_to_cluster_dict)
    
    return asset_to_cluster_dict, sub_covariances
#------------------------------------------------------------------------------------------------#
                                            #VISUALS#
#------------------------------------------------------------------------------------------------#

def plot_cluster_dendrogram(linkage_tree, labels, optimal_k=None, figsize=(10, 5)):
    """
    Renders the hierarchical dendrogram tree with an optional cut-threshold line.
    """
    fig, ax = plt.subplots(figsize=figsize)

    color_thresh = 0.0
    if optimal_k is not None and optimal_k > 1:
        # Distance at which the tree splits into optimal_k clusters
        color_thresh = linkage_tree[-(optimal_k - 1), 2]

    dendrogram(
        linkage_tree,
        labels=labels,
        ax=ax,
        color_threshold=color_thresh,
        leaf_rotation=90,
        leaf_font_size=10
    )
    title_suffix = f" (Cut at K={optimal_k})" if optimal_k else ""
    ax.set_title(f"Hierarchical Clustering Dendrogram - Ward Linkage{title_suffix}")
    ax.set_ylabel("Distance")
    plt.tight_layout()
    return fig
