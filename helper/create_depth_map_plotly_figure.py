
import plotly.express as px
import numpy as np
import plotly.graph_objects as go

def create_depth_map_plotly_figure(depth_map: np.ndarray, title: str = 'Plotly Depth Map', 
                                   cmap: str = 'viridis') -> go.Figure:
    """
    Creates and returns a Plotly figure object visualizing a depth map as a heatmap.
    
    Args:
        depth_map (np.ndarray): Depth map (2D array) to visualize.
        title (str): Title for the plot.
        cmap (str): Colormap to use (Plotly supports many Matplotlib names).
        
    Returns:
        go.Figure: The Plotly figure object ready for display.
    """
    # Use Plotly Express for a simple heatmap/image display
    fig = px.imshow(
        depth_map, 
        color_continuous_scale=cmap,
        title=title
    )
    
    # Customize layout for better readability (optional)
    fig.update_layout(
        xaxis_title="X-Axis", 
        yaxis_title="Y-Axis",
        title_x=0.5 # Center the title
    )
    
    # Explicitly set the colorbar label
    fig.update_coloraxes(colorbar_title_text='Depth Value')
    
    # Return the figure object
    return fig
