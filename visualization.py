# visualization.py
import plotly.express as px
import pandas as pd

def plot_clusters_2d(df, x_col, y_col, cluster_col, title):
    """
    Generates a 2D scatter plot of clusters using Plotly Express.

    Args:
        df (pd.DataFrame): DataFrame containing DR components and cluster assignments.
        x_col (str): Column name for the X-axis (e.g., 'DR_Component_1').
        y_col (str): Column name for the Y-axis (e.g., 'DR_Component_2').
        cluster_col (str): Column name for cluster labels (e.g., 'Cluster').
        title (str): Title of the plot.

    Returns:
        plotly.graph_objects.Figure: A Plotly figure object.
    """
    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        color=cluster_col,
        hover_name="Asset",
        title=title,
        labels={x_col: "Component 1", y_col: "Component 2", cluster_col: "Cluster"},
        color_continuous_scale=px.colors.qualitative.Plotly if df[cluster_col].dtype == 'object' else px.colors.sequential.Plasma,
        template="plotly_white"
    )
    fig.update_traces(marker=dict(size=10, opacity=0.8, line=dict(width=1, color='DarkSlateGrey')))
    fig.update_layout(
        hovermode="closest",
        title_x=0.5,
        height=600,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig

def plot_clusters_3d(df, x_col, y_col, z_col, cluster_col, title):
    """
    Generates a 3D scatter plot of clusters using Plotly Express.

    Args:
        df (pd.DataFrame): DataFrame containing DR components and cluster assignments.
        x_col (str): Column name for the X-axis (e.g., 'DR_Component_1').
        y_col (str): Column name for the Y-axis (e.g., 'DR_Component_2').
        z_col (str): Column name for the Z-axis (e.g., 'DR_Component_3').
        cluster_col (str): Column name for cluster labels (e.g., 'Cluster').
        title (str): Title of the plot.

    Returns:
        plotly.graph_objects.Figure: A Plotly figure object.
    """
    fig = px.scatter_3d(
        df,
        x=x_col,
        y=y_col,
        z=z_col,
        color=cluster_col,
        hover_name="Asset",
        title=title,
        labels={x_col: "Component 1", y_col: "Component 2", z_col: "Component 3", cluster_col: "Cluster"},
        color_continuous_scale=px.colors.qualitative.Plotly if df[cluster_col].dtype == 'object' else px.colors.sequential.Plasma,
        template="plotly_white"
    )
    fig.update_traces(marker=dict(size=8, opacity=0.8, line=dict(width=1, color='DarkSlateGrey')))
    fig.update_layout(
        hovermode="closest",
        title_x=0.5,
        height=700,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig

def plot_factor_contributions(df_cluster, cluster_id, factor_columns):
    """
    Generates a bar chart showing the average factor contributions for a specific cluster.

    Args:
        df_cluster (pd.DataFrame): DataFrame containing assets belonging to a specific cluster.
        cluster_id (int or str): The ID of the cluster being analyzed.
        factor_columns (list): A list of column names representing the factors.

    Returns:
        plotly.graph_objects.Figure: A Plotly figure object.
    """
    if df_cluster.empty:
        fig = px.bar(title=f"No data for Cluster {cluster_id}")
        return fig

    # Calculate average factor values for the cluster
    avg_factors = df_cluster[factor_columns].mean().reset_index()
    avg_factors.columns = ['Factor', 'Average Value']

    fig = px.bar(
        avg_factors,
        x='Factor',
        y='Average Value',
        title=f"Average Factor Values for Cluster {cluster_id}",
        labels={'Factor': 'Factor', 'Average Value': 'Average Normalized Value'},
        template="plotly_white",
        color='Average Value', # Color bars based on their value
        color_continuous_scale=px.colors.sequential.Viridis
    )
    fig.update_layout(
        title_x=0.5,
        height=400,
        margin=dict(l=20, r=20, t=50, b=20),
        yaxis_title="Average Normalized Value",
        xaxis_title="Factor"
    )
    fig.update_traces(marker_line_width=1, marker_line_color='DarkSlateGrey')
    return fig
