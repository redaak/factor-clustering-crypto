# factor_calculator.py
import pandas as pd
import numpy as np

def calculate_volatility(df, window=30):
    """
    Calculates the rolling volatility (standard deviation of daily returns).

    Args:
        df (pd.DataFrame): DataFrame with 'Asset' and 'Returns' columns.
        window (int): The rolling window in days.

    Returns:
        pd.Series: A Series of volatility values, indexed by 'Asset' and 'Date'.
    """
    # Ensure returns are numeric
    df['Returns'] = pd.to_numeric(df['Returns'], errors='coerce')
    # Calculate rolling standard deviation of returns
    volatility = df.groupby('Asset')['Returns'].rolling(window=window).std().reset_index(level=0, drop=True)
    return volatility.rename('Volatility')

def calculate_momentum(df, window=60):
    """
    Calculates the rolling momentum (cumulative returns over a period).

    Args:
        df (pd.DataFrame): DataFrame with 'Asset' and 'Price' columns.
        window (int): The rolling window in days.

    Returns:
        pd.Series: A Series of momentum values, indexed by 'Asset' and 'Date'.
    """
    # Calculate rolling percentage change of price
    momentum = df.groupby('Asset')['Price'].pct_change(periods=window).reset_index(level=0, drop=True)
    return momentum.rename('Momentum')

def calculate_volume_score(df, window=30):
    """
    Calculates a normalized volume score (e.g., log of rolling average volume).
    This helps to normalize volume differences across assets.

    Args:
        df (pd.DataFrame): DataFrame with 'Asset' and 'Volume' columns.
        window (int): The rolling window in days for average volume.

    Returns:
        pd.Series: A Series of volume scores, indexed by 'Asset' and 'Date'.
    """
    # Ensure volume is numeric
    df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')
    # Calculate rolling average volume
    rolling_avg_volume = df.groupby('Asset')['Volume'].rolling(window=window).mean().reset_index(level=0, drop=True)
    # Apply log transformation to normalize scale, handle zero/negative volumes
    volume_score = np.log1p(rolling_avg_volume) # log1p(x) = log(1+x) to handle small values
    return volume_score.rename('Volume_Score')

def calculate_all_factors(df_data, volatility_window=30, momentum_window=60, volume_window=30):
    """
    Calculates all specified factors for the given cryptocurrency data.
    The function ensures that factors are calculated for the latest available date for each asset.

    Args:
        df_data (pd.DataFrame): DataFrame with 'Asset', 'Date', 'Price', 'Volume', 'Returns' columns.
        volatility_window (int): Window for volatility calculation.
        momentum_window (int): Window for momentum calculation.
        volume_window (int): Window for volume score calculation.

    Returns:
        pd.DataFrame: A DataFrame with 'Asset' and the calculated factor columns,
                      representing the latest factor values for each asset.
                      Returns None if input df_data is empty or invalid.
    """
    if df_data is None or df_data.empty:
        print("Input DataFrame for factor calculation is empty or None.")
        return None

    # Ensure 'Date' is datetime for proper sorting
    df_data['Date'] = pd.to_datetime(df_data['Date'])
    df_data = df_data.sort_values(by=['Asset', 'Date'])

    # Calculate factors
    df_data['Volatility'] = calculate_volatility(df_data.copy(), window=volatility_window)
    df_data['Momentum'] = calculate_momentum(df_data.copy(), window=momentum_window)
    df_data['Volume_Score'] = calculate_volume_score(df_data.copy(), window=volume_window)

    # Drop rows with NaN values that result from rolling calculations (e.g., initial rows)
    df_data.dropna(subset=['Volatility', 'Momentum', 'Volume_Score'], inplace=True)

    if df_data.empty:
        print("No data remaining after dropping NaNs from factor calculation. Adjust windows or data range.")
        return None

    # Get the latest factor values for each asset
    latest_factors = df_data.groupby('Asset').last().reset_index()

    # Select relevant columns for the output
    factor_cols = ['Asset', 'Volatility', 'Momentum', 'Volume_Score']
    return latest_factors[factor_cols]

#not like us