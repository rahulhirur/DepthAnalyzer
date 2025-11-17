import streamlit as st
import numpy as np
import io
from PIL import Image
from transformers import pipeline
import torch
from helper.create_depth_map_plotly_figure import create_depth_map_plotly_figure, colorize_depth_map_pil
from typing import Any, Tuple
from transformers import PromptDepthAnythingForDepthEstimation, PromptDepthAnythingImageProcessor

from transformers import AutoImageProcessor, AutoModelForDepthEstimation

#plotly go import
import plotly.graph_objects as go

# Set page config once at the top
st.set_page_config(layout="wide", page_title="Depth-Anything V2 Estimator")

def convert_depth_map_to_bytes(depth_image: Image.Image) -> bytes:
    """Converts the PIL depth image to a byte buffer for download."""
    buf = io.BytesIO()
    # Save the 16-bit depth map to the buffer
    depth_image.save(buf, format='PNG')
    return buf.getvalue()


# --- Configuration: Model Mappings ---
# Maps the user-friendly name to the Hugging Face Model ID
MODEL_MAP = {
    "Relative (Small - 25M)": "depth-anything/Depth-Anything-V2-Small-hf",
    "Relative (Base - 97M)": "depth-anything/Depth-Anything-V2-Base-hf",
    "Relative (Large - 335M)": "depth-anything/Depth-Anything-V2-Large-hf",
    "Metric (Indoor Large)": "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf",
    "Metric (Outdoor Large)": "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf",
    "Prompt (Large - 340M)": "depth-anything/prompt-depth-anything-vitl-hf",
    "Camera Depth Model d405": "depth-anything/camera-depth-model-d405",
    "Camera Depth Model d435": "depth-anything/camera-depth-model-d435",
    "Camera Depth Model Base": "depth-anything/camera-depth-model-base",
}

# --- Session State Initialization ---
if 'depth_pipeline' not in st.session_state:
    st.session_state['depth_pipeline'] = None

if 'img_processor_pipeline' not in st.session_state:
    st.session_state['img_processor_pipeline'] = None

if "run_estimation_btn_clicked" not in st.session_state:
    st.session_state["run_estimation_btn_clicked"] = False

if 'loaded_model_id' not in st.session_state:
    st.session_state['loaded_model_id'] = None
# State variables to store results after inference
if 'last_depth_data' not in st.session_state:
    st.session_state['last_depth_data'] = None
if 'last_original_image' not in st.session_state:
    st.session_state['last_original_image'] = None
if 'last_depth_image_pil' not in st.session_state:
    st.session_state['last_depth_image_pil'] = None
if 'last_model_title' not in st.session_state:
    st.session_state['last_model_title'] = None
# State flag to control heatmap generation
if 'heatmap_requested' not in st.session_state:
    st.session_state['heatmap_requested'] = False

def run_button_callback():
    st.session_state["run_estimation_btn_clicked"] = True


def tensor_to_pil_np(tensor_depth):
    """
    Converts a torch.Tensor depth map to a PIL Image.

    Args:
        tensor_depth (torch.Tensor): The input depth tensor.

    Returns:
        PIL.Image.Image: The converted PIL Image.
    """
    # Convert the tensor to a NumPy array
    depth_np = tensor_depth.squeeze().cpu().numpy()

    # Normalize the depth values to the range 0-255
    depth_normalized = (depth_np - depth_np.min()) / (depth_np.max() - depth_np.min()) * 255
    depth_normalized = depth_normalized.astype('uint8')

    # Convert the NumPy array to a PIL Image
    depth_image_pil = Image.fromarray(depth_normalized)
    
    return depth_image_pil, depth_np




def run_model_get_depth(original_image, lidar_depth_image=None):

    if lidar_depth_image is not None:
        st.write('1')
        img_processor = st.session_state['img_processor_pipeline']
        st.write('2')
        depth_pipeline = st.session_state['depth_pipeline']
        st.write('3')
        
        inputs = img_processor(
            images=original_image,
            depth_images=lidar_depth_image,
            return_tensors="pt"
        )
        st.write('4')
        with torch.no_grad():
            outputs = depth_pipeline(**inputs)
        st.write('5')

        post_processed_output = img_processor.post_process_depth_estimation(
            outputs,
            target_sizes=[(original_image.height, original_image.width)],
        )
        st.write('6')
        predicted_depth = post_processed_output[0]["predicted_depth"]
        st.write('7')
        if predicted_depth is None:
            raise KeyError(f"Predicted depth not found in model output keys, check prompt and depth image compatibility.")
            return None, None
            
        else:
            return tensor_to_pil_np(predicted_depth), predicted_depth.squeeze().cpu().numpy()

    else:


        depth_pipeline = st.session_state['depth_pipeline']

        result = depth_pipeline(original_image)

        if isinstance(result, list) and result:
            output = result[0]
        elif isinstance(result, dict):
            output = result
        else:
            raise RuntimeError(f"Unexpected model output type: {type(result)}")

        depth_image_pil = output.get("depth") 


        if depth_image_pil is None:

            raise KeyError(f"Depth image not found in model output keys: {list(output.keys())}")
            return None, None
        else:
            st.success("Inference complete!")
            data_numpy = output.get("predicted_depth").cpu().detach().numpy()
            return depth_image_pil, data_numpy
    

def min_max_scale_and_get_variance(data_2d: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Scales a 2D NumPy array using Min-Max normalization (range [0, 1]) 
    and returns the scaled array along with its variance.

    Args:
        data_2d: A 2D NumPy array containing numerical data.

    Returns:
        A tuple containing:
        1. normalized_data (np.ndarray): The 2D array scaled between 0 and 1.
        2. normalized_variance (float): The variance of the scaled data.
    
    Raises:
        ValueError: If the input array is not 2D.
    """
    
    if data_2d.ndim != 2:
        raise ValueError("Input array must be 2-dimensional.")
    
    # 1. Calculate the overall min and max for scaling
    data_min = np.min(data_2d)
    data_max = np.max(data_2d)
    data_range = data_max - data_min
    
    # 2. Handle the edge case where all values are the same
    if data_range == 0:
        # If max equals min, the array cannot be scaled. Return zeros or 
        # a neutral value (like 0.5) and zero variance.
        return np.zeros_like(data_2d), 0.0

    # 3. Apply Min-Max Normalization: X_scaled = (X - Min) / (Max - Min)
    normalized_data = (data_2d - data_min) / data_range
    
    # 4. Calculate the variance of the resulting scaled data
    normalized_variance = np.var(normalized_data)
    
    return normalized_variance    



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

def visualize_pcd_from_depth(depth_data: np.ndarray, metric: bool = False, threshold: float = 3.5, selected_colorscale: str = None):

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
    z = z * 100.0

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


# --- Utility Function: Model Loading (Cached) ---
@st.cache_resource
def load_model(model_id: str):
    """Loads the Hugging Face depth estimation pipeline."""
    if "prompt" in model_id.lower():

        try:
            image_processor = AutoImageProcessor.from_pretrained(model_id)
            model = AutoModelForDepthEstimation.from_pretrained(model_id)

            return model, image_processor
        except Exception as e:
            # Re-raise the exception for the main app's error handler
            st.write('Something went wrong')
            raise e
    else:

        try:
            depth_pipe = pipeline(
                task="depth-estimation", 
                model=model_id, 
                trust_remote_code=True,
                device="cuda:0" if torch.cuda.is_available() else "cpu"
            )
            return depth_pipe, None
        except Exception as e:
            # Re-raise the exception for the main app's error handler
            raise e
        
# --- Callback function to handle model loading on button click ---
def handle_model_load(model_id, model_selection):


    with st.status(f"Downloading and loading model: **{model_id}**...", expanded=True) as status:
        try:
            pipeline_instance, image_processor_instance = load_model(model_id)
            st.session_state['depth_pipeline'] = pipeline_instance
            st.session_state['img_processor_pipeline'] = image_processor_instance
            st.session_state['loaded_model_id'] = model_selection
            status.update(label=f"Model {model_id.split('/')[-1]} loaded successfully!", state="complete", expanded=False)
        except Exception as e:
            status.update(label=f"Failed to load model {model_id.split('/')[-1]}", state="error", expanded=True)
            st.error(f"Error during model loading: {e}")
            st.session_state['depth_pipeline'] = None
            st.session_state['loaded_model_id'] = None


def _load_single_input(source):
    """
    (Internal helper) Loads an image or numpy array from a Streamlit UploadedFile.

    Args:
        source: A Streamlit UploadedFile object.

    Returns:
        PIL.Image.Image or np.ndarray: The loaded data.
    """
    if source is None:
        return None

    # Check if source is file-like (Streamlit UploadedFile)
    if hasattr(source, 'read'):
        file_name = getattr(source, 'name', '').lower()
        
        if file_name.endswith('.npy'):
            # np.load can read file-like objects
            return np.load(source)
        elif file_name.endswith(('.png', '.jpg', '.jpeg')):
            # Open as an image file
            return Image.open(source)
        else:
            # Try to open as image by default
            try:
                return Image.open(source)
            except Exception:
                raise TypeError(
                    f"Unsupported file type: {file_name}. "
                    "Please upload a .npy, .png, .jpg, or .jpeg."
                )
    
    raise TypeError("Invalid input source. Expected Streamlit UploadedFile.")

def load_inputs(image_source, depth_source):
    """
    Loads the source image and the depth prompt.

    Args:
        image_source: The Streamlit UploadedFile for the source image.
        depth_source: The Streamlit UploadedFile for the depth prompt.

    Returns:
        tuple: (original_image, lidar_depth_image)
               (PIL.Image.Image or np.ndarray, PIL.Image.Image or np.ndarray)
    """
    original_image = _load_single_input(image_source)
    lidar_depth_image = _load_single_input(depth_source)
    return original_image, lidar_depth_image


def main():
    st.title("Depth-Anything V2 Estimator")
    st.markdown("### Zero-Shot & Metric Depth Estimation for Images")

    # --- Sidebar for Configuration and Model Loading ---
    st.sidebar.header("Model Selection")
    
    # 1. Model Selection Dropdown
    model_selection = st.sidebar.selectbox(
        "Choose Model Variant:",
        options=list(MODEL_MAP.keys()),
        index=0,
        key='model_selector'
    )
    

    model_id = MODEL_MAP[model_selection]
    
    st.sidebar.caption(f"HF ID: `{model_id}`")
    
    # Model Status and Load Button Logic
    is_model_ready = st.session_state['depth_pipeline'] and st.session_state['loaded_model_id'] == model_selection
    
    if is_model_ready:

        st.sidebar.success(f"Model **{model_selection}** is ready.")
        load_label = "Model Loaded - Click to Reload"
    else:

        st.sidebar.warning(f"Model **{model_selection}** is not loaded.")
        load_label = f"Load Model"

    # 2. Load Model Button
    st.sidebar.button(
        load_label,
        on_click=handle_model_load,
        args=(model_id, model_selection),
        type="secondary",
        width="stretch"
    )
    
    st.sidebar.markdown("---")
    
    # 3. File Uploader
    uploaded_file = st.sidebar.file_uploader(
        "Upload an Image (.jpg, .png)", 
        type=["jpg", "jpeg", "png"]
    )

    if st.session_state['img_processor_pipeline'] is not None:

        uploaded_lidar_file = st.sidebar.file_uploader(
            "Upload a depth/LiDAR image (.png, .npy)",
            type=["png", "npy"]
        )

    # 4. Run Inference Button (Conditionally Enabled)
    run_button = st.sidebar.button(
        "Run Depth Estimation", 
        type="primary", 
        use_container_width=True,
        on_click=run_button_callback,
        disabled=not is_model_ready or uploaded_file is None
    )


    # 5. Clear Output Button
    if st.sidebar.button(
        "Clear All Output",
        type="secondary",
        use_container_width=True,
        disabled=st.session_state['last_depth_data'] is None
    ):
        # Clear all result-related session state
        st.session_state['last_depth_data'] = None
        st.session_state['last_original_image'] = None
        st.session_state['last_depth_image_pil'] = None
        st.session_state['last_model_title'] = None
        st.session_state['heatmap_requested'] = False
        st.session_state["run_estimation_btn_clicked"] = False
        st.rerun()
    
    st.sidebar.markdown("---")
    st.sidebar.caption("The **Relative** models provide unitless depth indicating distance order. The **Metric** models are fine-tuned to output depth in meters.")

    # --- Main Content Area for Inference ---
    if run_button and uploaded_file and is_model_ready:
        try:

            if "prompt" in model_id.lower():
                
                original_image, lidar_depth_image = load_inputs(uploaded_file, uploaded_lidar_file)
                st.write('Prompt Data Loaded')
                depth_image_pil, data_numpy = run_model_get_depth(original_image, lidar_depth_image)
                st.write('Prompt model done')
            else:

                original_image = Image.open(uploaded_file).convert("RGB")
                
                depth_image_pil, data_numpy = run_model_get_depth(original_image)
            

            # --- Store Results in Session State ---
            st.session_state['last_original_image'] = original_image
            st.session_state['last_depth_image_pil'] = depth_image_pil
            st.session_state['last_model_title'] = model_selection
            # Extract and store NumPy array (the Z data for Plotly)
            st.session_state['last_depth_data'] = data_numpy
            # Reset heatmap requested state for the new data
            st.session_state['heatmap_requested'] = False 
            
            # Trigger a rerun to display the visualization section which is outside this block
            st.rerun()

        except Exception as e:
            st.error(f"An error occurred during processing: {e}")
    

    if st.session_state["run_estimation_btn_clicked"]:

        # --- Visualization and Download Section (runs if data is available) ---
        if st.session_state['last_depth_data'] is not None:
            
            # Access stored data
            original_image = st.session_state['last_original_image']
            depth_image_pil = st.session_state['last_depth_image_pil']
            data_numpy = st.session_state['last_depth_data']
            model_title = st.session_state['last_model_title']

            st.info(f"The variance of the estimated depth map is: {min_max_scale_and_get_variance(data_numpy):.6f}")
            
            # --- Data Preparation for Downloads ---
            file_name_prefix = f"depth_data_{model_title.replace(' ', '_')}"
            
            # 2. Prepare 16-bit PNG (Raw Data)
            raw_png_bytes = convert_depth_map_to_bytes(colorize_depth_map_pil(depth_image_pil))
            
            # 3. Prepare NumPy Data (.npy)
            npy_buffer = io.BytesIO()
            np.save(npy_buffer, data_numpy)
            npy_buffer.seek(0) # IMPORTANT: Reset buffer position before reading for download

            # --- Download Buttons (Updated Layout) ---
            st.markdown("---")
            st.subheader("Download Depth Data")

            # Use columns for a better layout, especially on mobile
            col1, col2 = st.columns(2)
            
            
            with col1:
                st.download_button(
                    label="Download RAW Depth PNG (16-bit)",
                    data=raw_png_bytes,
                    file_name=f"{file_name_prefix}_raw.png",
                    mime="image/png",
                    key="download_raw_png_v2",
                    type="secondary",
                    help="The PNG file containing raw 16-bit depth values for quantitative use."
                )

            with col2:
                st.download_button(
                    label="Download NumPy Array (.npy)",
                    data=npy_buffer,
                    file_name=f"{file_name_prefix}.npy",
                    mime="application/octet-stream",
                    key="download_npy_v2",
                    type="secondary",
                    help="Download the depth map as a raw NumPy array file."
                )
            
            # --- Heatmap Trigger Button ---
            st.markdown("---")
            st.subheader("Visualization Tabs")
            
            if not st.session_state['heatmap_requested']:
                st.button(
                    "Generate Interactive Heatmap (Plotly)",
                    key="trigger_plotly_heatmap",
                    on_click=lambda: st.session_state.update(heatmap_requested=True),
                    type="primary",
                    help="Click to generate the interactive Plotly visualization, which can be resource-intensive."
                )
            else:
                st.success("Interactive Plotly Heatmap is ready to view in the tab below.")


            # --- Visualization Tabs ---
            tabs = st.tabs(['Analytics Point Cloud', "Depth Map (PNG)", "Original Image", "Depth Heatmap (Plotly)"])
        # Tab 1: Original Image
            
            
            
            
            with tabs[2]:
                st.subheader("Original Image")
                st.image(original_image, width="stretch")

            # Tab 2: Depth Map PNG
            with tabs[1]:
                st.subheader("Depth Map Output (PNG Visualization)")
                st.image(colorize_depth_map_pil(depth_image_pil), width="stretch")
                if "Relative" in model_title:
                    st.caption("Output values are unitless (closer/further relative order).")
                else:
                    st.caption("Output values are absolute metric depth (in meters).")

            # Tab 3: Depth Heatmap Plotly (Conditional Generation)
            with tabs[3]:
                
                st.subheader("Interactive Depth Heatmap (Z-Values)")
                st.write(f"Visualization button clicked-{st.session_state.get('heatmap_requested')}")

                if st.session_state.get('heatmap_requested', False):
                    with st.status("Generating resource-intensive Plotly chart...", expanded=True) as status:
                        try:
                            plotly_figure, plotly_config = create_depth_map_plotly_figure(
                                data_numpy, 
                                title=f"Depth Heatmap - {model_title}",
                            )
                            status.update(label="Plotly Heatmap ready!", state="complete", expanded=True)
                            st.plotly_chart(plotly_figure, config=plotly_config)
                        except Exception as e:
                            status.update(label="Heatmap generation failed.", state="error", expanded=True)
                            st.error(f"Error generating Plotly figure: {e}")
                            st.session_state['heatmap_requested'] = False
                else:
                    st.info("Click 'Generate Interactive Heatmap (Plotly)' above to render the visualization.")
                
            # Tab 4: Point Cloud Visualization
            with tabs[0]:
                st.subheader("3D Point Cloud Visualization")

                st.write('Point cloud')
                # Controls for point cloud visualization
                # Default metric mode inferred from the model title when available
                metric_default = False
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

                # Compute a robust median (ignore non-finite and zero values where appropriate) to suggest a default threshold
                try:
                    flat = np.asarray(data_numpy).flatten()
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
                    
                    visualize_pcd_from_depth(data_numpy, metric=metric_toggle, threshold=threshold)
            
      
            # st.session_state["run_estimation_btn_clicked"] = False  # Reset the flag\
        else:
            # st.session_state["run_estimation_btn_clicked"] = False

            # Check if data is present, if not, show instructions
            if not is_model_ready and uploaded_file is None:
                st.info("Start by selecting a model and clicking 'Load Model', then upload an image.")
            elif is_model_ready and uploaded_file is None:
                st.info(f"Model **{model_selection}** is loaded. Please upload an image to run inference.")
            elif not is_model_ready and uploaded_file:
                st.info(f"Image uploaded. Click 'Load Model' to proceed with **{model_selection}**.")


if __name__ == "__main__":
    main()
