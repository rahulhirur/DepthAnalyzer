import streamlit as st
import numpy as np
import plotly.graph_objects as go
import io
from typing import IO, Any, Tuple
from PIL import Image


# ====================================================================
# UTILITY FUNCTIONS
# ====================================================================

def _to_2d_float_array(depth_input: Any) -> np.ndarray:
    """
    Normalize various depth input types to a 2D numpy float array.
    Accepts: dict, PIL Image, torch tensor, numpy array, or nested lists. 
    Non-numeric entries are coerced to NaN.
    """
    if isinstance(depth_input, dict):
        for k in ("depth", "depth_map"):
            if k in depth_input:
                depth_input = depth_input[k]
                break

    if hasattr(depth_input, 'cpu') and hasattr(depth_input, 'numpy'):
        arr = depth_input.cpu().numpy()
    elif isinstance(depth_input, Image.Image):
        arr = np.array(depth_input)
    else:
        arr = np.array(depth_input)

    try:
        arr = arr.astype(float)
    except Exception:
        def _safe(x):
            try:
                return float(x)
            except Exception:
                return np.nan
        vec = np.vectorize(_safe, otypes=[float])
        arr = vec(arr)

    if arr.ndim == 3 and arr.shape[2] in (3, 4):
        arr = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]

    if arr.ndim == 1:
        arr = arr[np.newaxis, :]
    if arr.ndim > 2:
        arr = arr[..., 0]

    return arr.astype(float)

import numpy as np
import plotly.graph_objects as go
from typing import Any, Tuple
# import streamlit as st # Assuming Streamlit might be used for st.error calls

def create_depth_map_plotly_figure(depth_map: np.ndarray, title: str = 'Plotly Depth Map', cmap: str = 'viridis') -> Tuple[go.Figure, dict]:
    """
    Creates a Plotly Heatmap figure from a NumPy depth map array using plotly.graph_objects.Heatmap.
    
    This function explicitly sets the axis ranges based on the array dimensions to ensure
    the heatmap fills the container correctly and removes extraneous axis space.

    Args:
        depth_map (np.ndarray): The 2D NumPy array containing depth values (Z data).
        title (str): The title for the chart.
        cmap (str): The colormap name (e.g., 'viridis') to use for the heatmap.

    Returns:
        Tuple[go.Figure, dict]: A tuple containing the Plotly Figure and the Plotly configuration dictionary.
    """
    
    try:
        # Get the dimensions of the depth map array (Height is Y, Width is X)
        height, width = depth_map.shape
        
        # Create the Heatmap trace
        heatmap_trace = go.Heatmap(
            z=depth_map,           # Pass the NumPy array directly as the Z data
            colorscale=cmap,       # Use the specified colormap
            colorbar_title='Depth Value'
        )

        fig = go.Figure(data=[heatmap_trace])
        
        # --- FIX: Update layout with explicit axis ranges and settings ---
        fig.update_layout(
            title=title,
            # X-axis configuration
            xaxis=dict(
                range=[0, width], # Explicitly set range from 0 to array width
                constrain='domain', # Prevents stretching
                showgrid=False,
                zeroline=False,
            ),
            # Y-axis configuration
            yaxis=dict(
                autorange='reversed',  # Keep reversed for typical image orientation (0,0 top-left)
                scaleanchor='x',       # Ensure the aspect ratio is square/correct
                scaleratio=1,          # Ensure 1:1 scaling
                range=[0, height],     # Explicitly set range from 0 to array height
                showgrid=False,
                zeroline=False,
            ),
            margin=dict(l=20, r=20, t=40, b=20), # Adjust margins for better fit
            autosize=True,
        )
        # -----------------------------------------------------------
        
        # --- Define Plotly Configuration (Kept as is) ---
        plotly_config = {
            'displayModeBar': True,
            'scrollZoom': True,       
            'displaylogo': False,
            'locale': 'en'
        }
        
        return fig, plotly_config
        
    except Exception as e:
        # Error handling block
        try:
            import streamlit as st
            st.error(f"Error creating Plotly figure: {e}")
            st.error(f"Type of depth_map: {type(depth_map)}")
        except ImportError:
            print(f"Error creating Plotly figure: {e}")
            print(f"Type of depth_map: {type(depth_map)}")
            
        return go.Figure(), {}

def colorize_depth_map_pil(depth_image_pil: Image.Image, cmap_name: str = 'magma') -> Image.Image:
    """
    Colorizes a raw grayscale PIL depth image using a custom NumPy colormap approximation,
    avoiding external dependencies like Matplotlib.

    Args:
        depth_image_pil (Image.Image): The single-channel (e.g., 'L' or 'I;16') raw PIL depth image.
        cmap_name (str): The requested colormap name (currently ignored in favor of custom logic).

    Returns:
        Image.Image: A color-mapped PIL image ready for display (RGB mode).
    """
    
    # 1. Convert PIL Image to NumPy array
    data_numpy = np.array(depth_image_pil).astype(np.float32)
    
    # 2. Normalize the data to the range [0, 1]
    depth_min = data_numpy.min()
    depth_max = data_numpy.max()
    
    if depth_max == depth_min: # Handle cases of uniform depth
        normalized_data = np.zeros_like(data_numpy, dtype=np.float32)
    else:
        normalized_data = (data_numpy - depth_min) / (depth_max - depth_min)
    
    # 3. Apply a custom colormap approximation using pure NumPy math.
    # This is designed to provide a high-contrast depth visualization
    # (e.g., Blue/Purple for close depth, Yellow/White for far depth, similar to Magma/Turbo).
    
    # R channel: increases steadily (0.1 -> 1.0)
    R = np.clip(1.5 * normalized_data - 0.5, 0, 1) 
    # G channel: increases in mid-range
    G = np.clip(1.2 * normalized_data, 0, 1)        
    # B channel: starts high, decreases (1.0 -> 0.0)
    B = np.clip(1.0 - 1.5 * normalized_data, 0, 1) 

    # 4. Stack the channels and scale to 0-255 (uint8)
    colored_data_rgb = np.stack([R, G, B], axis=-1)
    colored_data_rgb = (colored_data_rgb * 255).astype(np.uint8)
    
    # 5. Convert the NumPy array back to a PIL Image
    return Image.fromarray(colored_data_rgb)
