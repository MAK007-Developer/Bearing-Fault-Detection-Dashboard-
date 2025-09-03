import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO
import base64
 

def plot_time_series(data):
    """
    Plot time series data using Plotly.
    
    Args:
        data: DataFrame with DE and FE columns
        
    Returns:
        Plotly figure
    """
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        subplot_titles=("Drive End (DE) Vibration Signal", "Fan End (FE) Vibration Signal"))
    
    # Add DE signal trace
    fig.add_trace(
        go.Scatter(x=list(range(len(data))), y=data['DE'], name="DE Signal", line=dict(color="blue")),
        row=1, col=1
    )
    
    # Add FE signal trace
    fig.add_trace(
        go.Scatter(x=list(range(len(data))), y=data['FE'], name="FE Signal", line=dict(color="red")),
        row=2, col=1
    )
    
    # Update layout
    fig.update_layout(
        height=600,
        title_text="Vibration Signals Time Series",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis2_title="Time Point",
        yaxis_title="Amplitude (DE)",
        yaxis2_title="Amplitude (FE)",
    )
    
    return fig

def plot_pca(pca_result, labels):
    """
    Create a 2D scatter plot of PCA-reduced features.
    
    Args:
        pca_result: PCA-transformed features
        labels: Labels for each data point
        
    Returns:
        Plotly figure
    """
    # Create a DataFrame for plotting
    df = pd.DataFrame({
        'PC1': pca_result[:, 0],
        'PC2': pca_result[:, 1],
        'Label': labels
    })
    
    # Create the scatter plot
    fig = px.scatter(
        df, x='PC1', y='PC2', color='Label',
        title="PCA Visualization of Bearing Fault Features",
        labels={'PC1': 'Principal Component 1', 'PC2': 'Principal Component 2'},
        color_discrete_sequence=px.colors.qualitative.Bold
    )
    
    # Add confidence ellipses for each class
    for label in set(labels):
        subset = df[df['Label'] == label]
        if len(subset) > 2:  # Need at least 3 points for an ellipse
            x = subset['PC1']
            y = subset['PC2']
            
            # Calculate the center and covariance
            center_x = x.mean()
            center_y = y.mean()
            cov = np.cov(x, y)
            
            # Calculate eigenvalues and eigenvectors
            eigenvalues, eigenvectors = np.linalg.eig(cov)
            
            # Sort eigenvalues and eigenvectors
            order = eigenvalues.argsort()[::-1]
            eigenvalues = eigenvalues[order]
            eigenvectors = eigenvectors[:, order]
            
            # Angle between the x-axis and the largest eigenvector
            angle = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])
            
            # 95% confidence interval
            chisquare_val = 5.991  # 95% confidence for 2 degrees of freedom
            theta = np.linspace(0, 2 * np.pi, 100)
            
            # Parametric equation of the ellipse
            ellipse_x = center_x + np.sqrt(chisquare_val * eigenvalues[0]) * np.cos(theta) * np.cos(angle) - \
                        np.sqrt(chisquare_val * eigenvalues[1]) * np.sin(theta) * np.sin(angle)
            ellipse_y = center_y + np.sqrt(chisquare_val * eigenvalues[0]) * np.cos(theta) * np.sin(angle) + \
                        np.sqrt(chisquare_val * eigenvalues[1]) * np.sin(theta) * np.cos(angle)
            
            # Add ellipse to the plot
            fig.add_trace(
                go.Scatter(
                    x=ellipse_x, y=ellipse_y,
                    mode='lines',
                    line=dict(dash='dash'),
                    showlegend=False,
                    hoverinfo='skip'
                )
            )
    
    # Update layout
    fig.update_layout(
        height=600,
        legend_title="Fault Type",
        xaxis_title="Principal Component 1",
        yaxis_title="Principal Component 2"
    )
    
    return fig

def plot_explained_variance(explained_variance_ratio):
    """
    Create a bar plot of explained variance for each principal component.
    
    Args:
        explained_variance_ratio: Array of explained variance ratios
        
    Returns:
        Plotly figure
    """
    # Calculate cumulative explained variance
    cumulative = np.cumsum(explained_variance_ratio)
    
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add bar chart for explained variance
    fig.add_trace(
        go.Bar(
            x=[f"PC{i+1}" for i in range(len(explained_variance_ratio))],
            y=explained_variance_ratio,
            name="Explained Variance",
            marker_color="blue"
        ),
        secondary_y=False
    )
    
    # Add line chart for cumulative explained variance
    fig.add_trace(
        go.Scatter(
            x=[f"PC{i+1}" for i in range(len(cumulative))],
            y=cumulative,
            name="Cumulative Explained Variance",
            marker_color="red",
            mode="lines+markers"
        ),
        secondary_y=True
    )
    
    # Update layout
    fig.update_layout(
        title_text="Explained Variance by Principal Components",
        xaxis_title="Principal Component",
        barmode="group",
        height=500
    )
    
    # Update y-axes
    fig.update_yaxes(title_text="Explained Variance", secondary_y=False)
    fig.update_yaxes(title_text="Cumulative Explained Variance", secondary_y=True)
    
    return fig

def plot_confusion_matrix(confusion_matrix, class_names):
    """
    Create a heatmap visualization of the confusion matrix.
    
    Args:
        confusion_matrix: Confusion matrix array
        class_names: Names of the classes
        
    Returns:
        Plotly figure
    """
    # Create a dataframe from the confusion matrix
    df = pd.DataFrame(confusion_matrix, index=class_names, columns=class_names)
    
    # Create the heatmap
    fig = px.imshow(
        df,
        text_auto=True,
        labels=dict(x="Predicted", y="Actual", color="Count"),
        color_continuous_scale="Blues",
        aspect="auto"
    )
    
    # Update layout
    fig.update_layout(
        title="Confusion Matrix",
        height=500,
        xaxis_title="Predicted",
        yaxis_title="Actual"
    )
    
    return fig

def plot_anomaly_scores(scores, labels, indices):
    """
    Create a scatter plot of anomaly scores.
    
    Args:
        scores: Array of anomaly scores
        labels: Labels for each data point (normal/anomaly)
        indices: Indices of the data points
        
    Returns:
        Plotly figure
    """
    # Create a dataframe for plotting
    df = pd.DataFrame({
        'Index': indices,
        'Score': scores,
        'Type': labels
    })
    
    # Create the scatter plot
    fig = go.Figure()
    
    # Add scatter plot for anomaly scores
    for label, color in zip(['Normal', 'Anomaly'], ['blue', 'red']):
        subset = df[df['Type'] == label]
        fig.add_trace(
            go.Scatter(
                x=subset['Index'],
                y=subset['Score'],
                mode='markers',
                name=label,
                marker=dict(color=color, size=8)
            )
        )
    
    # Add a horizontal line for a default threshold
    fig.add_shape(
        type="line",
        x0=min(indices),
        x1=max(indices),
        y0=0.5,
        y1=0.5,
        line=dict(
            color="green",
            width=2,
            dash="dash"
        )
    )
    
    # Add annotation for the threshold line
    fig.add_annotation(
        x=max(indices),
        y=0.5,
        text="Default Threshold",
        showarrow=False,
        yshift=10
    )
    
    # Update layout
    fig.update_layout(
        title="Anomaly Scores",
        xaxis_title="Data Point Index",
        yaxis_title="Anomaly Score",
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig



