# crypto_analyzer.py
import streamlit as st
import pandas as pd
from data_fetcher import fetch_crypto_data
from factor_calculator import calculate_all_factors
from ml_pipeline import run_ml_pipeline

class CryptoAnalyzer:
    """
    A class to encapsulate the entire cryptocurrency analysis pipeline,
    including data fetching, factor calculation, and machine learning clustering.
    """
    def __init__(self, coin_ids, start_date, end_date,
                 volatility_window, momentum_window, volume_window,
                 dr_method, n_components, clustering_method, n_clusters=None, eps=None, min_samples=None):
        """
        Initializes the CryptoAnalyzer with analysis parameters.

        Args:
            coin_ids (list): List of CoinGecko coin IDs.
            start_date (datetime.date): Start date for data.
            end_date (datetime.date): End date for data.
            volatility_window (int): Window for volatility calculation.
            momentum_window (int): Window for momentum calculation.
            volume_window (int): Window for volume score calculation.
            dr_method (str): Dimensionality reduction method ("PCA" or "t-SNE").
            n_components (int): Number of components for DR.
            clustering_method (str): Clustering method ("K-Means" or "DBSCAN").
            n_clusters (int, optional): Number of clusters for K-Means.
            eps (float, optional): Epsilon for DBSCAN.
            min_samples (int, optional): Min samples for DBSCAN.
        """
        self.coin_ids = coin_ids
        self.start_date = start_date
        self.end_date = end_date
        self.volatility_window = volatility_window
        self.momentum_window = momentum_window
        self.volume_window = volume_window
        self.dr_method = dr_method
        self.n_components = n_components
        self.clustering_method = clustering_method
        self.n_clusters = n_clusters
        self.eps = eps
        self.min_samples = min_samples

        self.df_data = None        # Raw fetched data
        self.df_factors = None     # Data with calculated factors
        self.df_clustered = None   # Data after ML pipeline (DR and clustering)

    @st.cache_data(ttl=3600) # Cache this method's results
    def _fetch_data_cached(self):
        """Caches the data fetching process."""
        return fetch_crypto_data(self.coin_ids, self.start_date, self.end_date)

    @st.cache_data(ttl=3600) # Cache this method's results
    def _calculate_factors_cached(self, df):
        """Caches the factor calculation process."""
        return calculate_all_factors(df, self.volatility_window, self.momentum_window, self.volume_window)

    @st.cache_data(ttl=3600) # Cache this method's results
    def _run_ml_pipeline_cached(self, df_factors_copy):
        """Caches the ML pipeline execution."""
        return run_ml_pipeline(
            df_factors_copy,
            self.dr_method,
            self.n_components,
            self.clustering_method,
            self.n_clusters,
            self.eps,
            self.min_samples
        )

    def run_analysis(self):
        """
        Executes the full analysis pipeline: fetch data, calculate factors, run ML pipeline.

        Returns:
            bool: True if analysis was successful, False otherwise.
        """
        st.info(f"Fetching data for {len(self.coin_ids)} coins from {self.start_date} to {self.end_date}...")
        self.df_data = self._fetch_data_cached()

        if self.df_data is None or self.df_data.empty:
            st.error("No data fetched. Please check coin selections or date range. CoinGecko API might have rate limits or specific coin data might not be available for the selected period.")
            return False

        st.info("Calculating Volatility, Momentum, and Volume Score...")
        self.df_factors = self._calculate_factors_cached(self.df_data)

        if self.df_factors is None or self.df_factors.empty:
            st.error("Factor calculation failed. This might happen if there's not enough historical data for the selected windows.")
            return False

        st.info(f"Applying {self.dr_method} and {self.clustering_method} clustering...")
        # Pass a copy to _run_ml_pipeline_cached to ensure caching works correctly
        # as st.cache_data requires immutable inputs for cache hit.
        self.df_clustered = self._run_ml_pipeline_cached(self.df_factors.copy())

        if self.df_clustered is None or self.df_clustered.empty:
            st.error("ML pipeline failed. Check parameters or data quality.")
            return False

        return True

