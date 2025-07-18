# README.md

# Crypto Factor Screener & Cluster Explorer

## Overview

The **Crypto Factor Screener & Cluster Explorer** is a Streamlit web application designed for cryptocurrency enthusiasts, investors, and data scientists. It provides a powerful tool to analyze cryptocurrency assets by calculating key financial factors, applying machine learning for behavioral clustering, and visualizing these clusters in an interactive 2D or 3D space. Users can explore individual clusters, understand their factor contributions, build hypothetical portfolios, and export their analysis results.

## Features

- **Data Fetching:** Fetches historical price and volume data for various cryptocurrencies from the CoinGecko API.
- **Factor Calculation:** Computes essential financial factors such as:
  - **Volatility:** Measures price fluctuation over a defined period.
  - **Momentum:** Indicates the rate of price change over a defined period.
  - **Volume Score:** A normalized measure of trading activity.
- **Machine Learning Pipeline:**
  - **Normalization:** Scales factor data to a common range.
  - **Dimensionality Reduction:** Uses Principal Component Analysis (PCA) or t-Distributed Stochastic Neighbor Embedding (t-SNE) to reduce factors to 2D or 3D for visualization.
  - **Clustering:** Groups cryptocurrencies with similar behavior using K-Means or DBSCAN algorithms.
- **Interactive Visualization:**
  - Plotly-powered 2D/3D scatter plots to visualize asset clusters.
  - Hover over points to see asset details.
  - Select clusters to view assets within them and their average factor contributions.
- **Portfolio Builder:** Allows users to select assets across different clusters and provides a basic, simplified risk summary for the chosen portfolio.
- **Data Export:** Export clustered data and portfolio details to CSV files.
- **User-Friendly Interface:** Built with Streamlit for an intuitive and interactive user experience.

## Installation

To run this application locally, follow these steps:

1.  **Clone the repository:**

    ```bash
    git clone [https://github.com/yourusername/crypto-factor-screener.git](https://github.com/yourusername/crypto-factor-screener.git)
    cd crypto-factor-screener
    ```

2.  **Install Pipenv:**
    If you don't have Pipenv installed, you can install it via pip:

    ```bash
    pip install pipenv
    ```

3.  **Create and activate a virtual environment with Pipenv:**
    ```bash
    pipenv install
    pipenv shell
    ```

## Usage

1.  **Run the Streamlit application:**

    ```bash
    streamlit run app.py
    ```

2.  **Access the App:**
    The app will open in your default web browser (usually `http://localhost:8501`).

3.  **Interact with the App:**
    - **Sidebar Configuration:**
      - **Select Cryptocurrencies:** Choose the assets you want to analyze.
      - **Data Range:** Define the historical period for data fetching.
      - **Factor Calculation Parameters:** Adjust the rolling windows for volatility, momentum, and volume score.
      - **Machine Learning Parameters:** Select your preferred dimensionality reduction method (PCA/t-SNE) and clustering algorithm (K-Means/DBSCAN), along with their respective parameters.
    - **Run Analysis:** Click the "Run Analysis" button to fetch data, calculate factors, and perform clustering.
    - **Explore Clusters:** Use the "Select a Cluster to Explore" dropdown to view assets within each cluster and their average factor contributions.
    - **Build Portfolio:** Select assets from the "Portfolio Builder" section to get a basic risk summary.
    - **Export Data:** Download the clustered data or your selected portfolio assets as CSV files.

## Deployment on Streamlit Cloud

To deploy this app on Streamlit Cloud, follow these general steps:

1.  **Push to GitHub:** Ensure your entire project (including `app.py`, `data_fetcher.py`, `factor_calculator.py`, `ml_pipeline.py`, `visualization.py`, and `requirements.txt`) is pushed to a public GitHub repository.
2.  **Sign in to Streamlit Cloud:** Go to [share.streamlit.io](https://share.streamlit.io/) and sign in with your GitHub account.
3.  **New App:** Click "New app" in your Streamlit Cloud dashboard.
4.  **Connect Repository:** Select your GitHub repository, choose the branch (e.g., `main`), and specify `app.py` as the main file path.
5.  **Deploy:** Click "Deploy!" Streamlit Cloud will automatically install dependencies from `requirements.txt` and run your app.

## Project Structure

```
.
├── app.py                      # Main Streamlit application
├── data_fetcher.py             # Module for fetching cryptocurrency data
├── factor_calculator.py        # Module for calculating financial factors
├── ml_pipeline.py              # Module for data normalization, DR, and clustering
├── visualization.py            # Module for Plotly visualizations
├── requirements.txt            # List of Python dependencies
└── README.md                   # Project documentation
```

## Future Enhancements (Pro Tips)

- **Cluster Comparison Dashboard:** Add a dedicated section to compare key factors and characteristics across different clusters side-by-side.
- **Backtesting Module:** Implement a simple backtesting framework to simulate portfolio performance based on cluster assignments over historical periods.
- **Mobile Responsiveness & Modern UI:** Enhance the app's visual appeal and ensure full responsiveness for mobile devices using custom CSS or Streamlit's theming options.
- **Sentiment/On-chain Data:** Integrate sentiment data from news/social media or on-chain metrics (e.g., active addresses, transaction count) as additional factors.
- **Advanced Portfolio Optimization:** Incorporate more sophisticated portfolio optimization techniques (e.g., Markowitz, Black-Litterman).
- **User Authentication:** For private data or user-specific portfolios, implement a user authentication system.

## Contributing

Contributions are welcome! If you have suggestions for improvements or new features, please open an issue or submit a pull request.

## License

This project is open-source and available under the [MIT License](LICENSE).
