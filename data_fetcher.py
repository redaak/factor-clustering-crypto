# data_fetcher.py
import pandas as pd
import requests
from datetime import datetime, timedelta

def fetch_crypto_data(coin_ids, start_date, end_date):
    """
    Fetches historical price and volume data for a list of coin IDs from CoinGecko API.
    Calculates daily returns.

    Args:
        coin_ids (list): A list of CoinGecko coin IDs (e.g., ["bitcoin", "ethereum"]).
        start_date (datetime.date): The start date for data fetching.
        end_date (datetime.date): The end date for data fetching.

    Returns:
        pd.DataFrame: A DataFrame with 'Asset', 'Date', 'Price', 'Volume', 'Returns' columns,
                      or None if data fetching fails or no data is available.
    """
    base_url = "https://api.coingecko.com/api/v3/coins"
    all_data = []

    # Convert dates to Unix timestamps (milliseconds for CoinGecko)
    start_timestamp = int(datetime.combine(start_date, datetime.min.time()).timestamp())
    end_timestamp = int(datetime.combine(end_date, datetime.max.time()).timestamp())

    for coin_id in coin_ids:
        url = f"{base_url}/{coin_id}/market_chart/range"
        params = {
            "vs_currency": "usd",
            "from": start_timestamp,
            "to": end_timestamp
        }
        try:
            response = requests.get(url, params=params)
            response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
            data = response.json()

            prices = data.get('prices', [])
            volumes = data.get('total_volumes', [])

            if not prices or not volumes:
                print(f"No data available for {coin_id} in the specified range.")
                continue

            # Create DataFrame for current coin
            df_coin = pd.DataFrame({
                'Date': [datetime.fromtimestamp(ts[0] / 1000).date() for ts in prices],
                'Price': [p[1] for p in prices],
                'Volume': [v[1] for v in volumes]
            })
            df_coin['Asset'] = coin_id
            df_coin = df_coin[['Asset', 'Date', 'Price', 'Volume']] # Reorder columns

            # Ensure unique dates and sort
            df_coin = df_coin.drop_duplicates(subset=['Date']).sort_values(by='Date').reset_index(drop=True)

            # Calculate daily returns
            df_coin['Returns'] = df_coin.groupby('Asset')['Price'].pct_change()

            all_data.append(df_coin)

        except requests.exceptions.RequestException as e:
            print(f"Error fetching data for {coin_id}: {e}")
        except ValueError as e:
            print(f"Error parsing JSON for {coin_id}: {e}")
        except Exception as e:
            print(f"An unexpected error occurred for {coin_id}: {e}")

    if not all_data:
        return None

    df_final = pd.concat(all_data, ignore_index=True)

    # Handle missing returns (first day of each asset will have NaN)
    df_final['Returns'] = df_final['Returns'].fillna(0) # Or drop rows, depending on strategy

    return df_final

