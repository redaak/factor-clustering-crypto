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

    @staticmethod
    @st.cache_data(ttl=3600)
    def _fetch_data_cached(coin_ids, start_date, end_date):
        """Caches the data fetching process."""
        return fetch_crypto_data(coin_ids, start_date, end_date)

    @staticmethod
    @st.cache_data(ttl=3600)
    def _calculate_factors_cached(df, volatility_window, momentum_window, volume_window):
        """Caches the factor calculation process."""
        return calculate_all_factors(df, volatility_window, momentum_window, volume_window)

    @staticmethod
    @st.cache_data(ttl=3600)
    def _run_ml_pipeline_cached(df_factors_copy, dr_method, n_components, clustering_method, n_clusters, eps, min_samples):
        """Caches the ML pipeline execution."""
        return run_ml_pipeline(
            df_factors_copy,
            dr_method,
            n_components,
            clustering_method,
            n_clusters,
            eps,
            min_samples
        )

    def run_analysis(self):
        """
        Executes the full analysis pipeline: fetch data, calculate factors, run ML pipeline.

        Returns:
            bool: True if analysis was successful, False otherwise.
        """
        st.info(f"Fetching data for {len(self.coin_ids)} coins from {self.start_date} to {self.end_date}...")
        self.df_data = self._fetch_data_cached(self.coin_ids, self.start_date, self.end_date)

        if self.df_data is None or self.df_data.empty:
            st.error("No data fetched. Please check coin selections or date range. CoinGecko API might have rate limits or specific coin data might not be available for the selected period.")
            return False

        st.info("Calculating Volatility, Momentum, and Volume Score...")
        self.df_factors = self._calculate_factors_cached(self.df_data, self.volatility_window, self.momentum_window, self.volume_window)

        if self.df_factors is None or self.df_factors.empty:
            st.error("Factor calculation failed. This might happen if there's not enough historical data for the selected windows.")
            return False

        st.info(f"Applying {self.dr_method} and {self.clustering_method} clustering...")
        self.df_clustered = self._run_ml_pipeline_cached(
            self.df_factors.copy(),
            self.dr_method,
            self.n_components,
            self.clustering_method,
            self.n_clusters,
            self.eps,
            self.min_samples
        )

        if self.df_clustered is None or self.df_clustered.empty:
            st.error("ML pipeline failed. Check parameters or data quality.")
            return False

        return True