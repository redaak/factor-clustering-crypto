import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def calculate_investment_metrics(df):
    """Calculate investment metrics for cryptocurrencies including alpha and beta."""
    metrics = {}
    
    # Calculate market (BTC) returns first
    btc_data = df[df['Asset'] == 'BTC-USD'].sort_values('Date')
    market_returns = btc_data['Returns'].fillna(0)
    risk_free_rate = 0.02  # Assuming 2% risk-free rate
    
    for asset in df['Asset'].unique():
        asset_data = df[df['Asset'] == asset].sort_values('Date')
        daily_returns = asset_data['Returns'].fillna(0)

        # Calculate annual return
        annual_return = (1 + daily_returns).prod() ** (252/len(daily_returns)) - 1

        # Calculate volatility (annualized)
        volatility = daily_returns.std() * np.sqrt(252)

        # Calculate beta (using BTC as market)
        if asset != 'BTC-USD':
            # Align dates for asset and market returns
            merged = pd.merge(
                asset_data[['Date', 'Returns']],
                btc_data[['Date', 'Returns']],
                on='Date',
                suffixes=('_asset', '_market')
            )
            asset_aligned = merged['Returns_asset'].fillna(0)
            market_aligned = merged['Returns_market'].fillna(0)
            if len(asset_aligned) > 1 and len(market_aligned) > 1:
                covariance = np.cov(asset_aligned, market_aligned)[0][1]
                market_variance = np.var(market_aligned)
                beta = covariance / market_variance if market_variance != 0 else 0
            else:
                beta = 0
            # Calculate alpha
            asset_avg_return = asset_aligned.mean() * 252
            market_avg_return = market_aligned.mean() * 252
            alpha = asset_avg_return - (risk_free_rate + beta * (market_avg_return - risk_free_rate))
        else:
            beta = 1.0
            alpha = 0.0

        # Calculate Sharpe Ratio (assuming risk-free rate of 0.02)
        rf = 0.02
        sharpe_ratio = (annual_return - rf) / volatility if volatility != 0 else 0

        # Calculate Maximum Drawdown
        cumulative_returns = (1 + daily_returns).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdowns = cumulative_returns/rolling_max - 1
        max_drawdown = drawdowns.min()

        # Store metrics
        metrics[asset] = {
            'Annual Return': annual_return,
            'Annualized Volatility': volatility,
            'Sharpe Ratio': sharpe_ratio,
            'Max Drawdown': max_drawdown
        }
    
    return pd.DataFrame(metrics).T

def plot_correlation_matrix(df_factors, factor_columns):
    """Create correlation matrix plot for factors."""
    corr_matrix = df_factors[factor_columns].corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix,
        x=factor_columns,
        y=factor_columns,
        text=np.round(corr_matrix, 2),
        texttemplate='%{text}',
        textfont={'size': 12},
        colorscale='RdBu',
        zmid=0
    ))
    
    fig.update_layout(
        title='Factor Correlation Matrix',
        width=600,
        height=500
    )
    
    return fig

def plot_price_trends(df, selected_assets):
    """Create line chart for price trends ."""
    fig = go.Figure()
    
    # Sort assets to ensure BTC-USD is first if present
    sorted_assets = sorted(selected_assets, key=lambda x: (x != "BTC-USD", x))
    
    for i, asset in enumerate(sorted_assets):
        asset_data = df[df['Asset'] == asset].sort_values('Date')
        price_c = asset_data['Price'] 
        
        fig.add_trace(go.Scatter(
            x=asset_data['Date'],
            y=price_c,
            name=f"{asset}",
            mode='lines',
            visible='legendonly' if i > 0 else True
        ))
    
    fig.update_layout(
        height=600,
        title='Price Trends Over Time',
        hovermode='x unified',
        showlegend=True,
        yaxis=dict(
            title="Price (USD)",
            gridcolor="lightgray"
        ),
        legend=dict(
            title="Click to Show/Hide",
            itemclick="toggle",
            itemdoubleclick="toggleothers"
        ),
        xaxis=dict(
            gridcolor="lightgray"
        )
    )
    
    return fig

def plot_cumulative_returns(df, selected_assets):
    """Create cumulative returns chart showing relative performance."""
    fig = go.Figure()
    
    for asset in selected_assets:
        asset_data = df[df['Asset'] == asset].sort_values('Date')
        cumulative_returns = ((1 + asset_data['Returns'].fillna(0)).cumprod() - 1) * 100
        
        fig.add_trace(go.Scatter(
            x=asset_data['Date'],
            y=cumulative_returns,
            name=asset,
            mode='lines'
        ))
    
    fig.update_layout(
        title='Cumulative Returns',
        xaxis_title='Date',
        yaxis_title='Cumulative Return',
        hovermode='x unified',
        showlegend=True
    )
    
    return fig

def plot_drawdown_analysis(df, selected_assets):
    """Create drawdown analysis chart."""
    fig = go.Figure()
    
    for asset in selected_assets:
        asset_data = df[df['Asset'] == asset].sort_values('Date')
        returns = asset_data['Returns'].fillna(0)
        cumulative_returns = (1 + returns).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdowns = cumulative_returns/rolling_max - 1
        
        fig.add_trace(go.Scatter(
            x=asset_data['Date'],
            y=drawdowns,
            name=asset,
            mode='lines'
        ))
    
    fig.update_layout(
        title='Drawdown Analysis',
        xaxis_title='Date',
        yaxis_title='Drawdown',
        hovermode='x unified',
        showlegend=True
    )
    
    return fig
