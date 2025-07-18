# data_fetcher.py
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import streamlit as st

def fetch_crypto_data(coin_ids, start_date, end_date):
    """
    Fetches historical price and volume data for a list of Yahoo Finance crypto tickers (e.g., 'BTC-USD').
    Calculates daily returns.

    Args:
        coin_ids (list): A list of Yahoo Finance tickers (e.g., ["BTC-USD", "ETH-USD"]).
        start_date (datetime.date): The start date for data fetching.
        end_date (datetime.date): The end date for data fetching.

    Returns:
        pd.DataFrame: A DataFrame with 'Asset', 'Date', 'Price', 'Volume', 'Returns' columns,
                      or None if data fetching fails or no data is available.
    """
    all_data = []
    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')

    for ticker in coin_ids:
        try:
            # Validate ticker format
            if not ticker.endswith('-USD'):
                st.warning(f"Ticker {ticker} may not be in the correct format. Expected format: XXX-USD")
            
            # Create a yfinance Ticker object and fetch data
            crypto = yf.Ticker(ticker)
            df = crypto.history(start=start_str, end=end_str, interval='1d')
            
            if df.empty:
                st.warning(f"No data available for {ticker} in the specified range.")
                continue
                
            df = df.reset_index()
            df['Asset'] = ticker
            df = df.rename(columns={
                'Date': 'Date',
                'Close': 'Price',
                'Volume': 'Volume'
            })
            
            df = df[['Asset', 'Date', 'Price', 'Volume']]
            df = df.drop_duplicates(subset=['Date'])
            df = df.sort_values(by='Date')
            df = df.reset_index(drop=True)
            
            # Calculate returns
            df['Returns'] = df.groupby('Asset')['Price'].pct_change()
            
            # Validate data quality
            if df['Price'].isnull().any() or df['Volume'].isnull().any():
                st.warning(f"Some price or volume data is missing for {ticker}")
                df = df.fillna(method='ffill')  # Forward fill missing values
            
            all_data.append(df)
            st.success(f"Successfully fetched data for {ticker}")
            
        except Exception as e:
            st.error(f"Error fetching data for {ticker}: {str(e)}")
            continue

    if not all_data:
        return None

    df_final = pd.concat(all_data, ignore_index=True)
    df_final['Returns'] = df_final['Returns'].fillna(0)
    return df_final
