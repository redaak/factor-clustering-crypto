# app.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, timedelta

# Import custom modules and the new CryptoAnalyzer class
from crypto_analyzer import CryptoAnalyzer
from visualization import plot_clusters_2d, plot_clusters_3d, plot_factor_contributions
from investment_metrics import (
    calculate_investment_metrics,
    plot_correlation_matrix,
    plot_price_trends,
    plot_cumulative_returns,
    plot_drawdown_analysis
)

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="Crypto Factor Screener & Cluster Explorer",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Constants ---
DEFAULT_COINS = [
    "BTC-USD", "ETH-USD", "XRP-USD", "SOL-USD", "ADA-USD", "DOGE-USD", "DOT-USD", "LTC-USD", "LINK-USD", "AVAX-USD",
    "BNB-USD", "MATIC-USD", "UNI-USD", "ATOM-USD", "ALGO-USD", "NEAR-USD", "FTM-USD", "VET-USD", "HBAR-USD", "ONE-USD",
    "SAND-USD", "MANA-USD", "AXS-USD", "THETA-USD", "GRT-USD", "ENJ-USD", "CHZ-USD", "HOT-USD", "BAT-USD", "ZIL-USD"
]
FACTOR_COLUMNS = ['Volatility', 'Momentum', 'Volume_Score'] # Define factor columns for easier access

# --- Session State Initialization ---
# Initialize CryptoAnalyzer instance in session state if it doesn't exist
if 'crypto_analyzer' not in st.session_state:
    st.session_state.crypto_analyzer = None
if 'df_data' not in st.session_state:
    st.session_state.df_data = None
if 'df_factors' not in st.session_state:
    st.session_state.df_factors = None
if 'df_clustered' not in st.session_state:
    st.session_state.df_clustered = None
if 'selected_cluster_id' not in st.session_state:
    st.session_state.selected_cluster_id = None
if 'portfolio_assets' not in st.session_state:
    st.session_state.portfolio_assets = []

# --- Helper Functions ---
def calculate_portfolio_risk_summary(df_clustered, selected_assets):
    """Calculates a basic risk summary for the selected portfolio assets."""
    if not selected_assets or df_clustered is None:
        return "No assets selected for portfolio."

    # Filter for selected assets
    portfolio_df = df_clustered[df_clustered['Asset'].isin(selected_assets)]

    if portfolio_df.empty:
        return "Selected assets not found in the analyzed data."

    # Calculate average factors for the portfolio
    avg_factors = portfolio_df[FACTOR_COLUMNS].mean()

    # Simple risk assessment based on factors (example logic)
    # Higher volatility, lower momentum, higher volume score (indicating more trading activity/liquidity)
    # might imply different risk profiles. This is a very simplified example.
    risk_score = (avg_factors['Volatility'] * 0.4) + \
                 ((1 - avg_factors['Momentum']) * 0.3) + \
                 (avg_factors['Volume_Score'] * 0.3) # Assuming higher volume score means higher liquidity/risk

    summary = f"""
    **Portfolio Summary:**
    - **Selected Assets:** {', '.join(selected_assets)}
    - **Average Volatility:** {avg_factors['Volatility']:.4f}
    - **Average Momentum:** {avg_factors['Momentum']:.4f}
    - **Average Volume Score:** {avg_factors['Volume_Score']:.4f}
    - **Estimated Risk Score (0-1):** {risk_score:.4f} (Higher score indicates potentially higher risk/liquidity)

    *Note: This is a simplified risk assessment based on normalized factors.*
    """
    return summary

# --- UI Layout ---
st.title("💰 Crypto Factor Screener & Cluster Explorer")

st.markdown("""
Welcome to the **Crypto Factor Screener & Cluster Explorer**!

This application helps you understand and group digital assets based on their fundamental financial characteristics and behavioral patterns using machine learning.

### What does it do?

1.  **Factor Screening**: We first analyze each cryptocurrency to calculate key financial "factors" that describe its behavior. Think of these as its unique financial DNA.
""")

st.markdown("    * **Volatility ($\sigma$)**: How much an asset's price fluctuates. A higher value means more unpredictable price swings.")
st.latex(r"\sigma = \sqrt{\frac{1}{N-1} \sum_{i=1}^{N} (R_i - \bar{R})^2}")
st.markdown("    Where $R_i$ is the daily return, $\\bar{R}$ is the average return, and $N$ is the number of days in the window.")

st.markdown("    * **Momentum (M)**: The tendency of an asset's price to continue in its current direction. Positive momentum means it's been going up, negative means down.")
st.latex(r"M = \frac{P_{\text{current}} - P_{\text{past}}}{P_{\text{past}}}")
st.markdown("    Where $P_{\text{current}}$ is the current price and $P_{\text{past}}$ is the price from a defined period ago.")

st.markdown("    * **Volume Score (V)**: A measure of trading activity, normalized to allow fair comparison across assets of different sizes.")
st.latex(r"V = \log(1 + \text{Average Daily Volume})")
st.markdown("    Using $\log(1+x)$ helps to scale down very large volume numbers and makes them comparable.")

st.markdown("""
2.  **Cluster Exploration**: Once we have these factors, we use advanced machine learning (like PCA/t-SNE for visualization and K-Means/DBSCAN for grouping) to find natural "clusters" of cryptocurrencies that behave similarly. This helps you:
    * **Identify similar assets**: Discover which cryptos move together or share risk profiles.
    * **Diversify your portfolio**: Pick assets from different clusters to spread risk.
    * **Spot emerging trends**: See if new groups are forming or existing ones are changing.

Dive in and start exploring the hidden patterns in the crypto market!
            click run analysis on the sidebar to begin.
""")

# --- Sidebar Configuration ---
st.sidebar.markdown("[Created by Reda Akdim](https://www.linkedin.com/in/reda-akdim/)")
st.sidebar.markdown("---")

# 1. Coin Selection
st.sidebar.subheader("Select Cryptocurrencies")
selected_coins = st.sidebar.multiselect(
    "Choose coins (max 20 recommended):",
    options=DEFAULT_COINS,
    default=DEFAULT_COINS[:10],
    key="coin_selector"
)

# 2. Run Analysis Button
st.sidebar.subheader("Run Analysis")
run_analysis = st.sidebar.button("🚀 Run Analysis", type="primary", key="run_analysis_btn")

# 3. Date Range Selection
st.sidebar.subheader("Data Range")
today = date.today()
end_date = st.sidebar.date_input("End Date:", value=today)
start_date = st.sidebar.date_input("Start Date:", value=end_date - timedelta(days=365))
if start_date >= end_date:
    st.sidebar.error("Start date must be before end date.")

# 3. Factor Calculation Parameters
st.sidebar.subheader("3. Factor Calculation Parameters")
volatility_window = st.sidebar.slider("Volatility Window (days):", 7, 90, 30, help="Period for calculating price volatility.", key="volatility_window")
momentum_window = st.sidebar.slider("Momentum Window (days):", 7, 180, 60, help="Period for calculating price momentum (returns).", key="momentum_window")
volume_window = st.sidebar.slider("Volume Score Window (days):", 7, 90, 30, help="Period for calculating average volume for scoring.", key="volume_window")

# 4. Machine Learning Parameters
st.sidebar.subheader("4. Machine Learning Parameters")
dr_method = st.sidebar.selectbox(
    "Dimensionality Reduction Method:",
    options=["PCA", "t-SNE"],
    help="PCA (Principal Component Analysis) is faster, t-SNE (t-Distributed Stochastic Neighbor Embedding) is better for visualizing complex non-linear relationships.",
    key="dr_method"
)
n_components = st.sidebar.slider(
    "Number of Components (for DR):",
    min_value=2,
    max_value=3,
    value=2,
    help="Number of dimensions to reduce factors to (2D or 3D for visualization).",
    key="n_components"
)
clustering_method = st.sidebar.selectbox(
    "Clustering Method:",
    options=["K-Means", "DBSCAN"],
    help="K-Means requires a predefined number of clusters. DBSCAN finds clusters based on density and can identify noise.",
    key="clustering_method"
)

n_clusters = None
eps = None
min_samples = None

if clustering_method == "K-Means":
    n_clusters = st.sidebar.slider("Number of Clusters (K-Means):", 2, 10, 4, help="The number of clusters K-Means will try to form.", key="n_clusters")
elif clustering_method == "DBSCAN":
    eps = st.sidebar.slider("DBSCAN Epsilon (eps):", 0.1, 2.0, 0.5, 0.1, help="The maximum distance between two samples for one to be considered as in the neighborhood of the other.", key="eps")
    min_samples = st.sidebar.slider("DBSCAN Min Samples:", 2, 10, 5, help="The number of samples (or total weight) in a neighborhood for a point to be considered as a core point.", key="min_samples")

# --- Results Display ---
if not selected_coins:
    st.sidebar.warning("Please select at least one cryptocurrency to analyze.")
elif run_analysis:  # Only run analysis when the button is clicked
    with st.spinner("Fetching data and running analysis... This might take a moment."):
        try:
            # Initialize the CryptoAnalyzer instance
            analyzer = CryptoAnalyzer(
                coin_ids=selected_coins,
                start_date=start_date,
                end_date=end_date,
                volatility_window=volatility_window,
                momentum_window=momentum_window,
                volume_window=volume_window,
                dr_method=dr_method,
                n_components=n_components,
                clustering_method=clustering_method,
                n_clusters=n_clusters,
                eps=eps,
                min_samples=min_samples
            )
            # Run the full analysis pipeline
            analysis_success = analyzer.run_analysis()
            if analysis_success:
                st.session_state.crypto_analyzer = analyzer
                st.session_state.df_data = analyzer.df_data
                st.session_state.df_factors = analyzer.df_factors
                st.session_state.df_clustered = analyzer.df_clustered
                st.success("Analysis completed successfully!")
            else:
                st.error("Analysis failed. Please check the parameters and try again.")
        except Exception as e:
            st.error(f"An unexpected error occurred during analysis: {str(e)}")
            st.session_state.df_data = None
            st.session_state.df_factors = None
            st.session_state.df_clustered = None

# Always show visuals if analysis results are available
if st.session_state.df_data is not None and st.session_state.df_clustered is not None:
    st.subheader("📊 Data Fetching & Cleaning")
    st.write("Raw Data Sample (last 5 days):")
    st.dataframe(st.session_state.df_data.tail())

    st.subheader("📈 Factor Calculation")
    st.write("Calculated Factors Sample:")
    st.dataframe(st.session_state.df_factors.head())
    st.write("Factor Statistics:")
    st.dataframe(st.session_state.df_factors[FACTOR_COLUMNS].describe())

    st.subheader("✨ Cluster Visualization")
    st.info("Visualizing clusters. Click on points in the plot to see asset details.")
    if n_components == 2:
        fig = plot_clusters_2d(
            st.session_state.df_clustered,
            'DR_Component_1',
            'DR_Component_2',
            'Cluster',
            f'{clustering_method} Clusters ({dr_method} 2D)'
        )
    else:  # n_components == 3
        fig = plot_clusters_3d(
            st.session_state.df_clustered,
            'DR_Component_1',
            'DR_Component_2',
            'DR_Component_3',
            'Cluster',
            f'{clustering_method} Clusters ({dr_method} 3D)'
        )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📈 Price Analysis & Investment Metrics")
    fig_prices = plot_price_trends(st.session_state.df_data, selected_coins)
    st.plotly_chart(fig_prices, use_container_width=True)

    fig_returns = plot_cumulative_returns(st.session_state.df_data, selected_coins)
    st.plotly_chart(fig_returns, use_container_width=True)

    fig_drawdown = plot_drawdown_analysis(st.session_state.df_data, selected_coins)
    st.plotly_chart(fig_drawdown, use_container_width=True)

    st.write("**Investment Metrics**")
    metrics_df = calculate_investment_metrics(st.session_state.df_data)
    st.dataframe(metrics_df.style.format({
        'Annual Return': '{:.2%}',
        'Annualized Volatility': '{:.2%}',
        'Sharpe Ratio': '{:.2f}',
        'Max Drawdown': '{:.2%}'
    }))

    st.write("**Factor Correlation Analysis**")
    fig_corr = plot_correlation_matrix(st.session_state.df_factors, FACTOR_COLUMNS)
    st.plotly_chart(fig_corr, use_container_width=True)

    st.subheader("🔍 Interactive Cluster Exploration")
    cluster_options = sorted(st.session_state.df_clustered['Cluster'].unique().tolist())
    if -1 in cluster_options:  # DBSCAN noise cluster
        cluster_options.remove(-1)
        cluster_options.insert(0, -1)  # Put noise at the beginning

    selected_cluster = st.selectbox(
        "Select a Cluster to Explore:",
        options=cluster_options,
        index=0,
        format_func=lambda x: f"Cluster {x}" if x != -1 else "Noise (-1)"
    )
    st.session_state.selected_cluster_id = selected_cluster

    if st.session_state.selected_cluster_id is not None:
        cluster_data = st.session_state.df_clustered[
            st.session_state.df_clustered['Cluster'] == st.session_state.selected_cluster_id
        ].sort_values(by='Asset')

        st.write(f"**Assets in Cluster {st.session_state.selected_cluster_id}:**")
        st.dataframe(cluster_data[['Asset'] + FACTOR_COLUMNS])

        st.write(f"**Average Factor Contributions for Cluster {st.session_state.selected_cluster_id}:**")
        fig_factors = plot_factor_contributions(cluster_data, st.session_state.selected_cluster_id, FACTOR_COLUMNS)
        st.plotly_chart(fig_factors, use_container_width=True)


# --- Portfolio Builder ---
st.subheader("💼 Portfolio Builder")
if st.session_state.df_clustered is not None:
    all_assets = st.session_state.df_clustered['Asset'].unique().tolist()
    st.session_state.portfolio_assets = st.multiselect(
        "Select assets for your portfolio:",
        options=all_assets,
        default=st.session_state.portfolio_assets,
        help="Choose cryptocurrencies to build a hypothetical portfolio.",
        key="portfolio_asset_selector"
    )

    if st.session_state.portfolio_assets:
        st.markdown(calculate_portfolio_risk_summary(st.session_state.df_clustered, st.session_state.portfolio_assets))
    else:
        st.info("Select assets above to see a basic portfolio risk summary.")
else:
    st.info("Run the analysis first to enable the portfolio builder.")


# --- Export Data ---
st.subheader("📥 Export Results")
if st.session_state.df_clustered is not None:
    csv_clustered = st.session_state.df_clustered.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Clustered Data as CSV",
        data=csv_clustered,
        file_name="crypto_clusters.csv",
        mime="text/csv",
        help="Download the full dataset with calculated factors and cluster assignments."
    )

    if st.session_state.portfolio_assets:
        portfolio_df_export = st.session_state.df_clustered[st.session_state.df_clustered['Asset'].isin(st.session_state.portfolio_assets)]
        csv_portfolio = portfolio_df_export.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Portfolio Assets as CSV",
            data=csv_portfolio,
            file_name="crypto_portfolio.csv",
            mime="text/csv",
            help="Download details of assets selected for your portfolio."
        )
else:
    st.info("Run the analysis to enable data export.")

st.markdown("---")
st.markdown("Developed with ❤️ and coffee by Reda akdim.")


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
    def _fetch_data_cached(self, _self): # Added _self to ignore hashing
        """Caches the data fetching process."""
        return fetch_crypto_data(_self.coin_ids, _self.start_date, _self.end_date)

    @st.cache_data(ttl=3600) # Cache this method's results
    def _calculate_factors_cached(self, _self, df): # Added _self to ignore hashing
        """Caches the factor calculation process."""
        return calculate_all_factors(df, _self.volatility_window, _self.momentum_window, _self.volume_window)

    @st.cache_data(ttl=3600) # Cache this method's results
    def _run_ml_pipeline_cached(self, _self, df_factors_copy): # Added _self to ignore hashing
        """Caches the ML pipeline execution."""
        return run_ml_pipeline(
            df_factors_copy,
            _self.dr_method,
            _self.n_components,
            _self.clustering_method,
            _self.n_clusters,
            _self.eps,
            _self.min_samples
        )

    def run_analysis(self):
        """
        Executes the full analysis pipeline: fetch data, calculate factors, run ML pipeline.

        Returns:
            bool: True if analysis was successful, False otherwise.
        """
        st.info(f"Fetching data for {len(self.coin_ids)} coins from {self.start_date} to {self.end_date}...")
        # Pass self as _self to the cached method
        self.df_data = self._fetch_data_cached(self)

        if self.df_data is None or self.df_data.empty:
            st.error("No data fetched. Please check coin selections or date range. CoinGecko API might have rate limits or specific coin data might not be available for the selected period.")
            return False

        st.info("Calculating Volatility, Momentum, and Volume Score...")
        # Pass self as _self to the cached method
        self.df_factors = self._calculate_factors_cached(self, self.df_data)

        if self.df_factors is None or self.df_factors.empty:
            st.error("Factor calculation failed. This might happen if there's not enough historical data for the selected windows.")
            return False

        st.info(f"Applying {self.dr_method} and {self.clustering_method} clustering...")
        # Pass a copy to _run_ml_pipeline_cached to ensure caching works correctly
        # as st.cache_data requires immutable inputs for cache hit.
        # Also pass self as _self to the cached method
        self.df_clustered = self._run_ml_pipeline_cached(self, self.df_factors.copy())

        if self.df_clustered is None or self.df_clustered.empty:
            st.error("ML pipeline failed. Check parameters or data quality.")
            return False

        return True