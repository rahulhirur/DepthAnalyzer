import streamlit as st
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import io
from typing import IO, Any, Tuple

from PIL import Image

import pandas as pd

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


def create_depth_map_plotly_figure(depth_map: Any, title: str = 'Plotly Depth Map', cmap: str = 'viridis') -> Tuple[go.Figure, dict]:
    
    try:
        # numeric = _to_2d_float_array(depth_map)
        df = pd.DataFrame(depth_map)
        fig = px.imshow(
            df,
            color_continuous_scale=cmap,
            title=title
        )
        fig.update_coloraxes(colorbar_title_text='Depth Value')
        
        # --- Define Plotly Configuration (Moved Here as Requested) ---
        plotly_config = {
            'displayModeBar': True,
            'scrollZoom': True,       
            'displaylogo': False,
            'locale': 'en'
        }
        
        return fig, plotly_config
    except Exception as e:
        st.error(f"Error creating Plotly figure: {e}")
        st.error(f"Type of depth_map: {type(depth_map)}")


def convert_plotly_to_downloadable_bytes(fig: go.Figure, format: str = 'png') -> IO[bytes]:
    """
    Converts a Plotly Figure object into an in-memory BytesIO buffer 
    for use with st.download_button. Requires 'kaleido'.
    """
    img_buffer = io.BytesIO()
    
    # FUTURE-PROOF: Using fig.write_image() without the deprecated 'engine' argument.
    fig.write_image(file=img_buffer, format=format)
    
    img_buffer.seek(0) 
    
    return img_buffer