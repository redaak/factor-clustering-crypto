import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def calculate_cumulative_returns(returns):
    """Calculate cumulative returns."""
    return (1 + returns).cumprod() - 1

def split_positive_negative_returns(returns):
    """Split returns into positive and negative components."""
    positive_returns = returns.clip(lower=0)
    negative_returns = returns.clip(upper=0)
    return positive_returns, negative_returns

def add_trace(fig, x, y, name, color, row, col, visible):
    """Add a trace to the figure."""
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            name=name,
            mode='lines',
            line=dict(color=color, width=1.5),
            visible=visible
        ),
        row=row, col=col
    )

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
            covariance = np.cov(daily_returns, market_returns)[0][1]
            market_variance = np.var(market_returns)
            beta = covariance / market_variance
            
            # Calculate alpha
            asset_avg_return = daily_returns.mean() * 252
            market_avg_return = market_returns.mean() * 252
            alpha = asset_avg_return - (risk_free_rate + beta * (market_avg_return - risk_free_rate))
        else:
            beta = 1.0
            alpha = 0.0
        
        # Calculate Sharpe Ratio
        sharpe_ratio = (annual_return - risk_free_rate) / volatility if volatility != 0 else 0
        
        # Calculate Maximum Drawdown
        cumulative_returns = calculate_cumulative_returns(daily_returns)
        rolling_max = cumulative_returns.expanding().max()
        drawdowns = (cumulative_returns - rolling_max) / rolling_max
        max_drawdown = drawdowns.min()
        
        metrics[asset] = {
            'Annual Return': annual_return,
            'Annualized Volatility': volatility,
            'Alpha': alpha,
            'Beta': beta,
            'Sharpe Ratio': sharpe_ratio,
            'Max Drawdown': max_drawdown
        }
    
    return pd.DataFrame.from_dict(metrics, orient='index')

def plot_price_trends(df, selected_assets):
    """Create line chart for price trends with better legend interaction."""
    fig = go.Figure()
    
    for asset in selected_assets:
        asset_data = df[df['Asset'] == asset].sort_values('Date')
        add_trace(
            fig,
            x=asset_data['Date'],
            y=asset_data['Price'],
            name=f"{asset}",
            color='blue',
            row=1, col=1,
            visible=True if asset == selected_assets[0] else 'legendonly'
        )
    
    fig.update_layout(
        height=600,
        title='Price Trends Over Time',
        hovermode='x unified',
        showlegend=True,
        yaxis=dict(
            title="Price (USD)",
            gridcolor="lightgray"
        ),
        xaxis=dict(
            gridcolor="lightgray"
        ),
        legend=dict(
            title="Click to Show/Hide",
            itemclick="toggle",
            itemdoubleclick="toggleothers"
        )
    )
    
    return fig

def plot_cumulative_returns(df, selected_assets):
    """Create separate charts for cumulative gains and drawdowns."""
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Positive Cumulative Returns', 'Negative Cumulative Returns'),
        vertical_spacing=0.15,
        row_heights=[0.5, 0.5]
    )
    
    for asset in selected_assets:
        asset_data = df[df['Asset'] == asset].sort_values('Date')
        clean_returns = asset_data['Returns'].fillna(0).clip(lower=-0.5, upper=0.5)
        
        positive_returns, negative_returns = split_positive_negative_returns(clean_returns)
        cum_positive = calculate_cumulative_returns(positive_returns) * 100
        cum_negative = calculate_cumulative_returns(negative_returns) * 100
        
        add_trace(
            fig,
            x=asset_data['Date'],
            y=cum_positive,
            name=f"{asset} (Positive)",
            color='green',
            row=1, col=1,
            visible=True if asset == selected_assets[0] else 'legendonly'
        )
        
        add_trace(
            fig,
            x=asset_data['Date'],
            y=cum_negative,
            name=f"{asset} (Negative)",
            color='red',
            row=2, col=1,
            visible=True if asset == selected_assets[0] else 'legendonly'
        )
    
    fig.update_layout(
        height=800,
        showlegend=True,
        hovermode='x unified',
        legend=dict(
            title="Click to Show/Hide Assets",
            itemclick="toggle",
            itemdoubleclick="toggleothers"
        )
    )
    
    fig.update_yaxes(
        title_text="Positive Returns (%)", 
        gridcolor="lightgray",
        zeroline=True,
        zerolinecolor="black",
        zerolinewidth=1,
        row=1, col=1
    )
    fig.update_yaxes(
        title_text="Negative Returns (%)",
        gridcolor="lightgray",
        zeroline=True,
        zerolinecolor="black",
        zerolinewidth=1,
        row=2, col=1
    )
    fig.update_xaxes(gridcolor="lightgray")
    
    return fig

def plot_drawdown_analysis(df, selected_assets):
    """Create drawdown analysis chart."""
    fig = go.Figure()
    
    for asset in selected_assets:
        asset_data = df[df['Asset'] == asset].sort_values('Date')
        returns = asset_data['Returns'].fillna(0)
        cumulative_returns = (1 + returns).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - rolling_max) / rolling_max * 100
        
        fig.add_trace(go.Scatter(
            x=asset_data['Date'],
            y=drawdown,
            name=asset,
            mode='lines',
            visible=True if asset == selected_assets[0] else 'legendonly'
        ))
    
    fig.update_layout(
        height=400,
        title='Drawdown Analysis',
        hovermode='x unified',
        showlegend=True,
        yaxis=dict(
            title="Drawdown (%)",
            zeroline=True,
            zerolinecolor="black",
            zerolinewidth=1,
            gridcolor="lightgray"
        ),
        xaxis=dict(gridcolor="lightgray"),
        legend=dict(
            title="Click to Show/Hide",
            itemclick="toggle",
            itemdoubleclick="toggleothers"
        )
    )
    
    return fig

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
        zmid=0,
        zmin=-1,
        zmax=1
    ))
    
    fig.update_layout(
        title='Factor Correlation Matrix',
        width=600,
        height=500,
        yaxis=dict(autorange="reversed")
    )
    
    return fig
