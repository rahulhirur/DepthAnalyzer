import streamlit as st
import numpy as np
import io
from PIL import Image
from transformers import pipeline
import torch
from helper.create_depth_map_plotly_figure import create_depth_map_plotly_figure, colorize_depth_map_pil
from typing import Any, Tuple
from transformers import PromptDepthAnythingForDepthEstimation, PromptDepthAnythingImageProcessor

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
    depth_image_pil = Image.fromarray(depth_normalized, 'L')
    
    return depth_image_pil, depth_np



def run_model_get_depth(original_image, lidar_depth_image=None):

    if lidar_depth_image is not None:
        img_processor = st.session_state['img_processor_pipeline']
        depth_pipeline = st.session_state['depth_pipeline']

        inputs = img_processor(
            images=original_image,
            depth_images=lidar_depth_image,
            return_tensors="pt"
        )

        with torch.no_grad():
            outputs = depth_pipeline(**inputs)

        post_processed_output = img_processor.post_process_depth_estimation(
            outputs,
            target_sizes=[(original_image.height, original_image.width)],
        )
        
        predicted_depth = post_processed_output[0]["predicted_depth"]

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


# --- Utility Function: Model Loading (Cached) ---
@st.cache_resource
def load_model(model_id: str):
    """Loads the Hugging Face depth estimation pipeline."""
    if "prompt" in model_id.lower():

        try:
            image_processor = PromptDepthAnythingImageProcessor.from_pretrained(model_id)
            model = PromptDepthAnythingForDepthEstimation.from_pretrained(model_id)

            return model, image_processor
        except Exception as e:
            # Re-raise the exception for the main app's error handler
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
        disabled=not is_model_ready or uploaded_file is None
    )

    st.sidebar.markdown("---")
    st.sidebar.caption("The **Relative** models provide unitless depth indicating distance order. The **Metric** models are fine-tuned to output depth in meters.")

    # --- Main Content Area for Inference ---
    if run_button and uploaded_file and is_model_ready:
        try:

            if "prompt" in model_id.lower():
                
                original_image, lidar_depth_image = load_inputs(uploaded_file, uploaded_lidar_file)

                depth_image_pil, data_numpy = run_model_get_depth(original_image, lidar_depth_image)
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
    
    # --- Visualization and Download Section (runs if data is available) ---
    if st.session_state['last_depth_data'] is not None:
        
        # Access stored data
        original_image = st.session_state['last_original_image']
        depth_image_pil = st.session_state['last_depth_image_pil']
        data_numpy = st.session_state['last_depth_data']
        model_title = st.session_state['last_model_title']
        
        # --- Data Preparation for Downloads ---
        file_name_prefix = f"depth_data_{model_title.replace(' ', '_')}"
        
        # 2. Prepare 16-bit PNG (Raw Data)
        raw_png_bytes = convert_depth_map_to_bytes(depth_image_pil)
        
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
        tabs = st.tabs(["Depth Map (PNG)", "Original Image", "Depth Heatmap (Plotly)"])
        
        # Tab 1: Original Image
        with tabs[1]:
            st.subheader("Original Image")
            st.image(original_image, width="stretch")

        # Tab 2: Depth Map PNG
        with tabs[0]:
            st.subheader("Depth Map Output (PNG Visualization)")
            st.image(colorize_depth_map_pil(depth_image_pil), width="stretch")
            if "Relative" in model_title:
                st.caption("Output values are unitless (closer/further relative order).")
            else:
                st.caption("Output values are absolute metric depth (in meters).")

        # Tab 3: Depth Heatmap Plotly (Conditional Generation)
        with tabs[2]:
            
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
        
    # --- Initial Prompts/Instruction Blocks ---
    else:
        # Check if data is present, if not, show instructions
        if not is_model_ready and uploaded_file is None:
            st.info("Start by selecting a model and clicking 'Load Model', then upload an image.")
        elif is_model_ready and uploaded_file is None:
            st.info(f"Model **{model_selection}** is loaded. Please upload an image to run inference.")
        elif not is_model_ready and uploaded_file:
            st.info(f"Image uploaded. Click 'Load Model' to proceed with **{model_selection}**.")


if __name__ == "__main__":
    main()
