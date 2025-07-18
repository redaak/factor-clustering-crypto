# app.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, timedelta

# Import custom modules and the new CryptoAnalyzer class
from crypto_analyzer import CryptoAnalyzer
from visualization import plot_clusters_2d, plot_clusters_3d, plot_factor_contributions # Still need these for direct plotting

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="Crypto Factor Screener & Cluster Explorer",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Constants ---
DEFAULT_COINS = ["bitcoin", "ethereum", "ripple", "solana", "cardano", "dogecoin", "polkadot", "litecoin", "chainlink", "avalanche-2"]
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
Welcome to the Crypto Factor Screener & Cluster Explorer! This application helps you analyze and group cryptocurrencies based on their financial factors and behavioral patterns using machine learning.
""")

# --- Sidebar for User Inputs ---
st.sidebar.header("⚙️ Configuration")

# Coin Selection
st.sidebar.subheader("1. Select Cryptocurrencies")
selected_coins = st.sidebar.multiselect(
    "Choose coins (max 20 recommended for performance):",
    options=DEFAULT_COINS,
    default=DEFAULT_COINS[:10],
    help="Select the cryptocurrencies you want to analyze. Data is fetched from CoinGecko."
)

# Date Range Selection
st.sidebar.subheader("2. Data Range")
today = date.today()
end_date = st.sidebar.date_input("End Date:", value=today)
start_date = st.sidebar.date_input("Start Date:", value=end_date - timedelta(days=365))
if start_date >= end_date:
    st.sidebar.error("Start date must be before end date.")

# Factor Calculation Parameters
st.sidebar.subheader("3. Factor Calculation Parameters")
volatility_window = st.sidebar.slider("Volatility Window (days):", 7, 90, 30, help="Period for calculating price volatility.")
momentum_window = st.sidebar.slider("Momentum Window (days):", 7, 180, 60, help="Period for calculating price momentum (returns).")
volume_window = st.sidebar.slider("Volume Score Window (days):", 7, 90, 30, help="Period for calculating average volume for scoring.")

# Machine Learning Parameters
st.sidebar.subheader("4. Machine Learning Parameters")

dr_method = st.sidebar.selectbox(
    "Dimensionality Reduction Method:",
    options=["PCA", "t-SNE"],
    help="PCA (Principal Component Analysis) is faster, t-SNE (t-Distributed Stochastic Neighbor Embedding) is better for visualizing complex non-linear relationships."
)

n_components = st.sidebar.slider(
    "Number of Components (for DR):",
    min_value=2,
    max_value=3,
    value=2,
    help="Number of dimensions to reduce factors to (2D or 3D for visualization)."
)

clustering_method = st.sidebar.selectbox(
    "Clustering Method:",
    options=["K-Means", "DBSCAN"],
    help="K-Means requires a predefined number of clusters. DBSCAN finds clusters based on density and can identify noise."
)

n_clusters = None
eps = None
min_samples = None

if clustering_method == "K-Means":
    n_clusters = st.sidebar.slider("Number of Clusters (K-Means):", 2, 10, 4, help="The number of clusters K-Means will try to form.")
elif clustering_method == "DBSCAN":
    eps = st.sidebar.slider("DBSCAN Epsilon (eps):", 0.1, 2.0, 0.5, 0.1, help="The maximum distance between two samples for one to be considered as in the neighborhood of the other.")
    min_samples = st.sidebar.slider("DBSCAN Min Samples:", 2, 10, 5, help="The number of samples (or total weight) in a neighborhood for a point to be considered as a core point.")


# --- Run Analysis Button ---
if st.sidebar.button("🚀 Run Analysis"):
    if not selected_coins:
        st.sidebar.warning("Please select at least one cryptocurrency to analyze.")
    else:
        with st.spinner("Fetching data and running analysis... This might take a moment."):
            try:
                # Initialize the CryptoAnalyzer instance
                st.session_state.crypto_analyzer = CryptoAnalyzer(
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
                analysis_success = st.session_state.crypto_analyzer.run_analysis()

                if analysis_success:
                    st.success("Analysis completed successfully!")
                    st.session_state.df_data = st.session_state.crypto_analyzer.df_data
                    st.session_state.df_factors = st.session_state.crypto_analyzer.df_factors
                    st.session_state.df_clustered = st.session_state.crypto_analyzer.df_clustered

                    # Display raw data sample
                    st.subheader("📊 Data Fetching & Cleaning")
                    st.write("Raw Data Sample (last 5 days):")
                    st.dataframe(st.session_state.df_data.tail())

                    # Display calculated factors sample
                    st.subheader("📈 Factor Calculation")
                    st.write("Calculated Factors Sample:")
                    st.dataframe(st.session_state.df_factors.head())
                    st.write("Factor Statistics:")
                    st.dataframe(st.session_state.df_factors[FACTOR_COLUMNS].describe())

                    # 4. Visualize Clusters
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
                    else: # n_components == 3
                        fig = plot_clusters_3d(
                            st.session_state.df_clustered,
                            'DR_Component_1',
                            'DR_Component_2',
                            'DR_Component_3',
                            'Cluster',
                            f'{clustering_method} Clusters ({dr_method} 3D)'
                        )
                    st.plotly_chart(fig, use_container_width=True)

                    # --- Interactive Cluster Exploration ---
                    st.subheader("🔍 Interactive Cluster Exploration")
                    if st.session_state.df_clustered is not None:
                        cluster_options = sorted(st.session_state.df_clustered['Cluster'].unique().tolist())
                        if -1 in cluster_options: # DBSCAN noise cluster
                            cluster_options.remove(-1)
                            cluster_options.insert(0, -1) # Put noise at the beginning

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

                            # Plot factor contributions for the selected cluster
                            st.write(f"**Average Factor Contributions for Cluster {st.session_state.selected_cluster_id}:**")
                            fig_factors = plot_factor_contributions(cluster_data, st.session_state.selected_cluster_id, FACTOR_COLUMNS)
                            st.plotly_chart(fig_factors, use_container_width=True)
                    else:
                        st.info("Run analysis first to explore clusters.")

                else:
                    st.error("Analysis failed. Please check the parameters and try again.")
                    # Clear session state data if analysis failed
                    st.session_state.df_data = None
                    st.session_state.df_factors = None
                    st.session_state.df_clustered = None

            except Exception as e:
                st.error(f"An unexpected error occurred during analysis: {e}")
                st.exception(e) # Display full traceback for debugging
                st.session_state.df_data = None
                st.session_state.df_factors = None
                st.session_state.df_clustered = None


# --- Portfolio Builder ---
st.subheader("💼 Portfolio Builder")
if st.session_state.df_clustered is not None:
    all_assets = st.session_state.df_clustered['Asset'].unique().tolist()
    st.session_state.portfolio_assets = st.multiselect(
        "Select assets for your portfolio:",
        options=all_assets,
        default=st.session_state.portfolio_assets,
        help="Choose cryptocurrencies to build a hypothetical portfolio."
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
st.markdown("Developed with ❤️ using Streamlit, Plotly, Pandas, and Scikit-learn.")







