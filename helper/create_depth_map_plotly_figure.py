
import plotly.express as px
import numpy as np
import plotly.graph_objects as go
from PIL import Image
import torch


def _to_2d_float_array(depth_input) -> np.ndarray:
    """
    Normalize various depth input types to a 2D numpy float array.
    Accepts: dict with keys ('depth','depth_map'), PIL Image, torch tensor,
    numpy array, or nested lists. Non-numeric entries are coerced to NaN.
    """
    # Extract from dict if necessary
    if isinstance(depth_input, dict):
        for k in ("depth", "depth_map"):
            if k in depth_input:
                depth_input = depth_input[k]
                break

    # Torch tensor
    if hasattr(depth_input, 'cpu') and hasattr(depth_input, 'numpy'):
        arr = depth_input.cpu().numpy()
    elif isinstance(depth_input, Image.Image):
        arr = np.array(depth_input)
    else:
        arr = np.array(depth_input)

    # Attempt to coerce to float; on failure map bad entries to NaN
    try:
        arr = arr.astype(float)
    except Exception:
        # safe vectorized conversion
        def _safe(x):
            try:
                return float(x)
            except Exception:
                return np.nan
        vec = np.vectorize(_safe, otypes=[float])
        arr = vec(arr)

    # If RGB(A), convert to luminance
    if arr.ndim == 3 and arr.shape[2] in (3, 4):
        arr = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]

    # Ensure 2D
    if arr.ndim == 1:
        # turn into a row
        arr = arr[np.newaxis, :]
    if arr.ndim > 2:
        # collapse extra dimensions by taking the first channel
        arr = arr[..., 0]

    return arr.astype(float)


def create_depth_map_plotly_figure(depth_map, title: str = 'Plotly Depth Map', cmap: str = 'viridis') -> go.Figure:
    """
    Creates and returns a Plotly figure object visualizing a depth map as a heatmap.
    
    Args:
        depth_map (np.ndarray): Depth map (2D array) to visualize.
        title (str): Title for the plot.
        cmap (str): Colormap to use (Plotly supports many Matplotlib names).
        
    Returns:
        go.Figure: The Plotly figure object ready for display.
    """
    # Normalize input to a numeric 2D array
    numeric = _to_2d_float_array(depth_map)

    # Use Plotly Express for a simple heatmap/image display
    fig = px.imshow(
        numeric,
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
