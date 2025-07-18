import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def calculate_investment_metrics(df):
    """Calculate investment metrics for cryptocurrencies."""
    metrics = {}
    
    for asset in df['Asset'].unique():
        asset_data = df[df['Asset'] == asset].sort_values('Date')
        
        # Calculate daily and annual returns
        daily_returns = asset_data['Returns'].fillna(0)
        annual_return = (1 + daily_returns).prod() ** (252/len(daily_returns)) - 1
        
        # Calculate volatility (annualized)
        volatility = daily_returns.std() * np.sqrt(252)
        
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
    """Create line charts for price trends."""
    fig = go.Figure()
    
    for asset in selected_assets:
        asset_data = df[df['Asset'] == asset].sort_values('Date')
        fig.add_trace(go.Scatter(
            x=asset_data['Date'],
            y=asset_data['Price'],
            name=asset,
            mode='lines'
        ))
    
    fig.update_layout(
        title='Price Trends Over Time (Log Scale)',
        xaxis_title='Date',
        yaxis_title='Price (USD)',
        yaxis_type="log",
        hovermode='x unified',
        showlegend=True
    )
    
    return fig

def plot_cumulative_returns(df, selected_assets):
    """Create cumulative returns chart."""
    fig = go.Figure()
    
    for asset in selected_assets:
        asset_data = df[df['Asset'] == asset].sort_values('Date')
        cumulative_returns = (1 + asset_data['Returns'].fillna(0)).cumprod()
        
        fig.add_trace(go.Scatter(
            x=asset_data['Date'],
            y=cumulative_returns,
            name=asset,
            mode='lines'
        ))
    
    fig.update_layout(
        title='Cumulative Returns (Log Scale)',
        xaxis_title='Date',
        yaxis_title='Cumulative Return',
        yaxis_type="log",
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
