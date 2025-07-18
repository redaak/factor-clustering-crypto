
# ml_pipeline.py
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN
import numpy as np

def normalize_data(df, columns_to_normalize):
    """
    Normalizes specified columns of a DataFrame using StandardScaler.

    Args:
        df (pd.DataFrame): The input DataFrame.
        columns_to_normalize (list): A list of column names to be normalized.

    Returns:
        pd.DataFrame: A new DataFrame with normalized columns.
        sklearn.preprocessing.StandardScaler: The fitted scaler object.
    """
    df_normalized = df.copy()
    scaler = StandardScaler()
    df_normalized[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])
    return df_normalized, scaler

def apply_pca(df, n_components):
    """
    Applies Principal Component Analysis (PCA) for dimensionality reduction.

    Args:
        df (pd.DataFrame): The input DataFrame (already normalized).
        n_components (int): The number of principal components to keep (2 or 3).

    Returns:
        pd.DataFrame: DataFrame with 'DR_Component_1', 'DR_Component_2' (and 'DR_Component_3') columns.
        sklearn.decomposition.PCA: The fitted PCA model.
    """
    pca = PCA(n_components=n_components)
    components = pca.fit_transform(df)
    col_names = [f'DR_Component_{i+1}' for i in range(n_components)]
    df_dr = pd.DataFrame(data=components, columns=col_names, index=df.index)
    return df_dr, pca

def apply_tsne(df, n_components):
    """
    Applies t-Distributed Stochastic Neighbor Embedding (t-SNE) for dimensionality reduction.

    Args:
        df (pd.DataFrame): The input DataFrame (already normalized).
        n_components (int): The number of dimensions to embed into (2 or 3).

    Returns:
        pd.DataFrame: DataFrame with 'DR_Component_1', 'DR_Component_2' (and 'DR_Component_3') columns.
        sklearn.manifold.TSNE: The fitted t-SNE model.
    """
    tsne = TSNE(n_components=n_components, random_state=42, perplexity=min(5, len(df)-1), n_iter=1000)
    # Perplexity should be less than the number of samples.
    # Adjust perplexity if dataset is very small.
    if len(df) <= 1:
        return pd.DataFrame(), None # Not enough samples for t-SNE

    components = tsne.fit_transform(df)
    col_names = [f'DR_Component_{i+1}' for i in range(n_components)]
    df_dr = pd.DataFrame(data=components, columns=col_names, index=df.index)
    return df_dr, tsne

def perform_kmeans_clustering(df, n_clusters):
    """
    Performs K-Means clustering.

    Args:
        df (pd.DataFrame): The input DataFrame (after dimensionality reduction).
        n_clusters (int): The number of clusters to form.

    Returns:
        pd.Series: A Series of cluster labels.
        sklearn.cluster.KMeans: The fitted K-Means model.
    """
    if len(df) < n_clusters:
        print(f"Warning: Number of samples ({len(df)}) is less than n_clusters ({n_clusters}). K-Means might not perform as expected.")
        # Fallback: assign all to one cluster or handle as error
        return pd.Series([0] * len(df), index=df.index, name='Cluster'), None

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10) # n_init for robustness
    clusters = kmeans.fit_predict(df)
    return pd.Series(clusters, index=df.index, name='Cluster'), kmeans

def perform_dbscan_clustering(df, eps, min_samples):
    """
    Performs DBSCAN clustering.

    Args:
        df (pd.DataFrame): The input DataFrame (after dimensionality reduction).
        eps (float): The maximum distance between two samples for one to be considered as in the neighborhood of the other.
        min_samples (int): The number of samples (or total weight) in a neighborhood for a point to be considered as a core point.

    Returns:
        pd.Series: A Series of cluster labels (-1 indicates noise).
        sklearn.cluster.DBSCAN: The fitted DBSCAN model.
    """
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = dbscan.fit_predict(df)
    return pd.Series(clusters, index=df.index, name='Cluster'), dbscan

def run_ml_pipeline(df_factors, dr_method, n_components, clustering_method, n_clusters=None, eps=None, min_samples=None):
    """
    Orchestrates the entire ML pipeline: normalization, dimensionality reduction, and clustering.

    Args:
        df_factors (pd.DataFrame): DataFrame with 'Asset' and factor columns.
        dr_method (str): "PCA" or "t-SNE".
        n_components (int): Number of components for DR (2 or 3).
        clustering_method (str): "K-Means" or "DBSCAN".
        n_clusters (int, optional): Number of clusters for K-Means.
        eps (float, optional): Epsilon for DBSCAN.
        min_samples (int, optional): Min samples for DBSCAN.

    Returns:
        pd.DataFrame: Original df_factors with added 'DR_Component_X' and 'Cluster' columns.
                      Returns None if any step fails.
    """
    if df_factors is None or df_factors.empty:
        print("Input DataFrame for ML pipeline is empty or None.")
        return None

    # Ensure 'Asset' is set as index for ML operations, then reset later
    df_ml = df_factors.set_index('Asset').copy()
    factor_cols = [col for col in df_ml.columns if col not in ['Date']] # Assuming factors are the only non-asset/date columns

    if not factor_cols:
        print("No factor columns found for ML pipeline.")
        return None

    # 1. Normalization
    try:
        df_normalized, _ = normalize_data(df_ml, factor_cols)
    except Exception as e:
        print(f"Error during normalization: {e}")
        return None

    # 2. Dimensionality Reduction
    df_dr = pd.DataFrame()
    if dr_method == "PCA":
        try:
            df_dr, _ = apply_pca(df_normalized, n_components)
        except Exception as e:
            print(f"Error during PCA: {e}")
            return None
    elif dr_method == "t-SNE":
        try:
            df_dr, _ = apply_tsne(df_normalized, n_components)
        except Exception as e:
            print(f"Error during t-SNE: {e}")
            return None
    else:
        print(f"Unknown dimensionality reduction method: {dr_method}")
        return None

    if df_dr.empty:
        print("Dimensionality reduction resulted in an empty DataFrame.")
        return None

    # 3. Clustering
    clusters = pd.Series()
    if clustering_method == "K-Means":
        if n_clusters is None or n_clusters < 2:
            print("K-Means requires n_clusters >= 2.")
            return None
        try:
            clusters, _ = perform_kmeans_clustering(df_dr, n_clusters)
        except Exception as e:
            print(f"Error during K-Means clustering: {e}")
            return None
    elif clustering_method == "DBSCAN":
        if eps is None or min_samples is None:
            print("DBSCAN requires eps and min_samples.")
            return None
        try:
            clusters, _ = perform_dbscan_clustering(df_dr, eps, min_samples)
        except Exception as e:
            print(f"Error during DBSCAN clustering: {e}")
            return None
    else:
        print(f"Unknown clustering method: {clustering_method}")
        return None

    if clusters.empty:
        print("Clustering resulted in empty clusters.")
        return None

    # Combine results
    df_clustered = df_factors.copy()
    df_clustered = df_clustered.set_index('Asset') # Align index for merging
    df_clustered = df_clustered.merge(df_dr, left_index=True, right_index=True, how='left')
    df_clustered['Cluster'] = clusters
    df_clustered = df_clustered.reset_index() # Bring 'Asset' back as a column

    return df_clustered
