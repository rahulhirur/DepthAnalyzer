import streamlit as st
import numpy as np
import io
from PIL import Image
from transformers import pipeline
from helper.create_depth_map_plotly_figure import create_depth_map_plotly_figure, colorize_depth_map_pil

#plotly go import
import plotly.graph_objects as go



#add session state for generated point cloud x,y,z
if 'point_cloud_x' not in st.session_state:
    st.session_state.point_cloud_x = None
if 'point_cloud_y' not in st.session_state:
    st.session_state.point_cloud_y = None
if 'point_cloud_z' not in st.session_state:
    st.session_state.point_cloud_z = None


def clean_point_cloud(x, y, z):
        """
        Remove points from point cloud where z coordinate equals 0.
        
        Args:
            x: numpy array of x coordinates
            y: numpy array of y coordinates
            z: numpy array of z coordinates
        
        Returns:
            tuple: (cleaned_x, cleaned_y, cleaned_z, num_removed)
        """
        mask = z != 0
        cleaned_x = x[mask]
        cleaned_y = y[mask]
        cleaned_z = z[mask]
        num_removed = len(z) - len(cleaned_z)
        
        return cleaned_x, cleaned_y, cleaned_z, num_removed

def plot_scatter3d(x=None, y=None,z=None, selected_colorscale = 'viridis'):
     
    try:
        if x is None or y is None or z is None:
            raise ValueError("X, Y, and Z coordinates must be provided.")
            return None
        else:
                

            # Color by Z-coordinate (height) if no explicit colors
            marker_config = dict(size=2, color=z, colorscale= selected_colorscale, colorbar=dict(title='Z'))

            fig = go.Figure(data=[go.Scatter3d(x=x, y=y, z=z,
                                                mode='markers',
                                                marker=marker_config)])

            fig.update_layout(
                scene=dict(
                    xaxis_title='X',
                    yaxis_title='Y',
                    zaxis_title='Z',
                    aspectmode='data',
                    xaxis=dict(
                        visible=True, # Ensure X-axis is visible
                        showbackground=True,
                        backgroundcolor="rgba(0, 0, 0, 0.05)", # Light background for the grid
                        gridcolor="lightgrey", # Color of grid lines
                        linecolor="black", # Color of the axis line
                        zerolinecolor="black" # Color of the zero line
                    ),
                    yaxis=dict(
                        visible=True, # Ensure Y-axis is visible
                        showbackground=True,
                        backgroundcolor="rgba(0, 0, 0, 0.05)",
                        gridcolor="lightgrey",
                        linecolor="black",
                        zerolinecolor="black"
                    ),
                    zaxis=dict(
                        visible=True, # Ensure Z-axis is visible
                        showbackground=True,
                        backgroundcolor="rgba(0, 0, 0, 0.05)",
                        gridcolor="lightgrey",
                        linecolor="black",
                        zerolinecolor="black"
                    )
                
                ),
                autosize=False,
                width=1200,  # Adjust overall width
                height=800,   # Adjust overall height
                margin=dict(l=40, r=40, b=40, t=40)  # Adjust margins
            )
            return fig
    
    except Exception as e:
        st.error(f"Error creating point cloud figure: {e}")
        return None

def visualize_pcd_from_depth(depth_data: np.ndarray, metric: bool = False, threshold: float = 3.5, scale_factor: float = 100.0, selected_colorscale: str = None):

    """
    Create and display a 3D point cloud from a 2D depth map.

    Args:
        depth_data: 2D numpy array of depth (Z) values.
        metric: If True, apply metric-based thresholding (assumes depth in meters).
        threshold: Threshold (in same units as depth_data) used to zero out far values when metric=True.
        selected_colorscale: Plotly colorscale name.
    """

    # downsample for faster visualization if too large
    max_points = 200000  # Maximum number of points to visualize
    st.write(f"Original depth data size: {depth_data.size} points.")
    if depth_data.size > max_points:
        factor = int(np.ceil(np.sqrt(depth_data.size / max_points)))
        depth_data = depth_data[::factor, ::factor]
        st.info(f"Depth data downsampled by a factor of {factor} for visualization.")

    h, w = depth_data.shape
    xx, yy = np.meshgrid(np.arange(w), np.arange(h))
    x = xx.flatten()
    y = yy.flatten()

    z = depth_data.flatten().astype(float)

    # If metric scaling is requested, use the provided threshold to remove far/invalid depths
    if metric:
        # Metric mode: zero out values beyond the threshold (commonly invalid/noisy lidar measurements)
        z = np.where(z > threshold, 0.0, z)
    else:
        # Non-metric mode: reverse threshold logic — zero out values below the threshold
        # This helps remove very-close/noise pixels when working with relative (unitless) outputs
        z = np.where(z < threshold, 0.0, z)

    # Scale for visibility in both modes
    z = z * scale_factor

    # Additional clamp to avoid extreme outliers dominating the visualization
    z = np.where(z > 500.0, 500.0, z)

    if selected_colorscale is None:
        selected_colorscale = 'viridis'

    if x is not None:
        fig = plot_scatter3d(x, y, z, selected_colorscale)

        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("Failed to create figure from point cloud data.")
    
    st.success("3D Point Cloud visualization generated successfully.")
    return x,y,z



# Set page config once at the top
st.set_page_config(layout="wide", page_title="Point Cloud Analyzer")

st.title('Point Cloud Analyzer')

# File uploaders
cols= st.columns(2)

with cols[0]:
    st.subheader("Generated Point Cloud")
    generated_file = st.file_uploader(
        "Upload Generated Point Cloud (.npy)",
        type=['npy'],
        key='generated'
    )
    
with cols[1]:
    st.subheader("Ground Truth Point Cloud")
    ground_truth_file = st.file_uploader(
        "Upload Ground Truth Point Cloud (.npy)",
        type=['npy'],
        key='ground_truth'
    )

# Read and process the uploaded files
generated_pc = None
ground_truth_pc = None


if generated_file is not None:
    generated_pc = np.load(io.BytesIO(generated_file.read()))
    st.success(f"Generated point cloud loaded: {generated_pc.shape}")

if ground_truth_file is not None:
    ground_truth_pc = np.load(io.BytesIO(ground_truth_file.read()))
    st.success(f"Ground truth point cloud loaded: {ground_truth_pc.shape}")


tabs = st.tabs(["3D Point Cloud Visualization", 'Point Cloud Comparison'])

with tabs[0]:

    st.subheader("3D Point Cloud Visualization")

    st.write('Point cloud')
    
    metric_default = False
    model_title = st.session_state['last_model_title']

    try:
        if model_title and "Metric" in model_title:
            metric_default = True
    except Exception:
        metric_default = False

    metric_toggle = st.checkbox(
        "Treat depth as metric",
        value=metric_default,
        help="If checked, values beyond the threshold will be removed before visualization.",
    )

    # Add a scale factor int number input
    scale_factor = st.number_input(
        "Scale Factor",
        min_value=1,
        max_value=1000,
        value=100,
        step=5,
        help="Scale factor to multiply depth values for visualization."
    )

    # Compute a robust median (ignore non-finite and zero values where appropriate) to suggest a default threshold
    try:
        flat = np.asarray(generated_pc).flatten()
        finite_mask = np.isfinite(flat)
        pos_mask = flat > 0
        combined_mask = finite_mask & pos_mask
        if combined_mask.any():
            median_val = float(np.median(flat[combined_mask]))
        else:
            # Fall back to median of finite values (could be zeros if all zeros)
            finite_only = flat[finite_mask]
            median_val = float(np.median(finite_only)) if finite_only.size > 0 else 0.0
    except Exception:
        median_val = 0.0

    st.write(f"Median depth (robust, non-zero): {median_val:.4f}")

    # Suggest a default threshold relative to the median (user can override)
    suggested_default = float(np.clip(median_val * 1.5 if median_val > 0 else 3.5, 0.0, 50.0))
    st.write(f"Suggested default threshold: {suggested_default:.4f}")

    if median_val <= 10.0:
        suggested_step = 0.1
    elif median_val <= 50.0:
        suggested_step = 0.5
    else:
        suggested_step = 1.0


    threshold = st.slider(
        "Depth threshold (units same as model output; meters if metric)",
        min_value=0.0,
        max_value=suggested_default*2,
        value=suggested_default,
        step=suggested_step,
        help=(
            "Threshold behaviour depends on the 'metric' toggle: "
            "If metric=True, values ABOVE this threshold are removed. "
            "If metric=False, values BELOW this threshold are removed (useful for relative maps)."
        ),
    )
    if st.button(
        "Generate 3D Point Cloud",
        key="generate_point_cloud",
        type="primary",
        help="Click to generate and visualize the 3D point cloud from the depth map."
    ):
        
        # x, y, z = visualize_pcd_from_depth(generated_pc, metric=metric_toggle, threshold=threshold)
        st.session_state.point_cloud_x, st.session_state.point_cloud_y, st.session_state.point_cloud_z = visualize_pcd_from_depth(
            generated_pc,
            metric=metric_toggle,
            threshold=threshold,
            scale_factor=scale_factor
        )


with tabs[1]:
    st.subheader("Point Cloud Comparison")

    if st.session_state.point_cloud_x is not None:

        


        # Clean generated point cloud
        cleaned_x, cleaned_y, cleaned_z, removed_count = clean_point_cloud(
            st.session_state.point_cloud_x,
            st.session_state.point_cloud_y,
            st.session_state.point_cloud_z
        )

        st.write(f"Original points: {len(st.session_state.point_cloud_z)}")
        st.write(f"Cleaned points (z ≠ 0): {len(cleaned_z)}")
        st.write(f"Removed {removed_count} points with z = 0")

        # Do the same cleaning for ground truth if available
        if ground_truth_pc is not None:
            gt_x, gt_y, gt_z, gt_removed_count = clean_point_cloud(
                ground_truth_pc[:, 0],
                ground_truth_pc[:, 1],
                ground_truth_pc[:, 2]
            )

            st.write(f"Ground Truth Original points: {len(ground_truth_pc)}")
            st.write(f"Ground Truth Cleaned points (z ≠ 0): {len(gt_z)}")
            st.write(f"Ground Truth Removed {gt_removed_count} points with z = 0")

        #Add a st slider to limit max points to plot
        max_plot_points = st.slider(
            "Max Points to Plot",
            min_value=10000,
            max_value=500000,
            value=50000,
            step=10000,
            help="Maximum number of points to plot for each point cloud to ensure performance."
        )

        if len(cleaned_z) > max_plot_points:

            sample_factor = int(np.ceil(len(cleaned_z) / max_plot_points))
            cleaned_x = cleaned_x[::sample_factor]
            cleaned_y = cleaned_y[::sample_factor]
            cleaned_z = cleaned_z[::sample_factor]

            st.info(f"Subsampled generated point cloud by factor {sample_factor} for plotting.")

        if ground_truth_pc is not None and len(gt_z) > max_plot_points:

            gt_sample_factor = int(np.ceil(len(gt_z) / max_plot_points))
            gt_x = gt_x[::gt_sample_factor]
            gt_y = gt_y[::gt_sample_factor]
            gt_z = gt_z[::gt_sample_factor]

            st.info(f"Subsampled ground truth point cloud by factor {gt_sample_factor} for plotting.")


        #Min value and max value for z in both point clouds
        st.write(f"Generated Point Cloud Z range: min={np.min(cleaned_z):.4f}, max={np.max(cleaned_z):.4f}")
        if ground_truth_pc is not None:
            st.write(f"Ground Truth Point Cloud Z range: min={np.min(gt_z):.4f}, max={np.max(gt_z):.4f}")

        # Normalize both point clouds to fit within the same scale for better comparison

        all_z = np.concatenate([cleaned_z, gt_z]) if ground_truth_pc is not None else cleaned_z
        z_min, z_max = np.min(all_z), np.max(all_z)
        cleaned_z = (cleaned_z - z_min) / (z_max - z_min) * 500.0  # Scale to [0, 500]
        if ground_truth_pc is not None:
            gt_z = (gt_z - z_min) / (z_max - z_min) * 500.0
        

        #Add a Button to plot point clouds
        if st.button(
            "Plot Point Cloud Comparison",
            key="plot_point_cloud_comparison",
            type="primary",
            help="Click to plot and compare the generated and ground truth point clouds."
        ):
        # Plot both point clouds together for comparison in streamlit
            fig = go.Figure()
            fig.add_trace(go.Scatter3d(
                x=cleaned_x,
                y=cleaned_y,
                z=cleaned_z,
                mode='markers',
                marker=dict(
                    size=2,
                    color='blue',
                    opacity=0.8
                ),
                name='Generated Point Cloud'
            ))
            if ground_truth_pc is not None:
                fig.add_trace(go.Scatter3d(
                    x=gt_x,
                    y=gt_y,
                    z=gt_z,
                    mode='markers',
                    marker=dict(
                        size=2,
                        color='red',
                        opacity=0.8
                    ),
                    name='Ground Truth Point Cloud'
                ))
                
                st.plotly_chart(fig)
