"""
src/clustering.py
Hierarchical clustering engine:
- Correlation to Euclidean distance transformation
- Ward's minimum-variance linkage construction
- Optimal K extraction via Silhouette Coefficient maximization
- Dendrogram visualization
"""

from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.metrics import silhouette_score
import pandas as pd
import numpy as np
import matplolib.py as plt

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
