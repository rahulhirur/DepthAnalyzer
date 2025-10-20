import streamlit as st
import numpy as np
import io
from PIL import Image
from transformers import pipeline
import torch
from helper.create_depth_map_plotly_figure import create_depth_map_plotly_figure, convert_plotly_to_downloadable_bytes

st.set_page_config(layout="wide")

# --- Configuration: Model Mappings ---
# Maps the user-friendly name to the Hugging Face Model ID
MODEL_MAP = {
    "Relative (Small - 25M)": "depth-anything/Depth-Anything-V2-Small-hf",
    "Relative (Base - 97M)": "depth-anything/Depth-Anything-V2-Base-hf",
    "Relative (Large - 335M)": "depth-anything/Depth-Anything-V2-Large-hf",
    "Metric (Indoor Large)": "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf",
    "Metric (Outdoor Large)": "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf",
}

# --- Session State Initialization ---
if 'depth_pipeline' not in st.session_state:
    st.session_state['depth_pipeline'] = None
if 'loaded_model_id' not in st.session_state:
    st.session_state['loaded_model_id'] = None

# --- Utility Function: Model Loading (Cached) ---
@st.cache_resource
def load_model(model_id: str):
    """
    Loads the Hugging Face depth estimation pipeline.
    Uses st.cache_resource to load the model only once, speeding up the app.
    Uses st.status to show the loading process.
    """
    # START: Using st.status for model loading progress
    # with st.status(f"Downloading and loading model: **{model_id}**...", expanded=True) as status:
    try:
        # The pipeline handles model loading, preprocessing, and inference steps.
        depth_pipe = pipeline(
            task="depth-estimation", 
            model=model_id, 
            trust_remote_code=True,
            device="cuda:0" if torch.cuda.is_available() else "cpu"
        )
        
        # Update status on success
        # status.update(label=f"Model {model_id.split('/')[-1]} loaded successfully!", state="complete", expanded=False)
        return depth_pipe
    except Exception as e:
        # Update status on error
        # status.update(label=f"Failed to load model {model_id.split('/')[-1]}", state="error", expanded=True)
        # Re-raise the exception for the main app's error handler
        raise e
    # END: Using st.status

# --- Callback function to handle model loading on button click ---
def handle_model_load(model_id, model_selection):
    """Callback to store the loaded model in session state."""
    with st.status(f"Downloading and loading model: **{model_id}**...", expanded=True) as status:
        
        try:
            pipeline_instance = load_model(model_id)
            st.session_state['depth_pipeline'] = pipeline_instance
            st.session_state['loaded_model_id'] = model_selection
            status.update(label=f"Model {model_id.split('/')[-1]} loaded successfully!", state="complete", expanded=False)
        except Exception as e:
            status.update(label=f"Failed to load model {model_id.split('/')[-1]}", state="error", expanded=True)
            st.error(f"Error during model loading: {e}")
            st.session_state['depth_pipeline'] = None
            st.session_state['loaded_model_id'] = None

def main():
    st.set_page_config(layout="wide", page_title="Depth-Anything V2 Estimator")
    st.title("Depth-Anything V2 Estimator")
    st.markdown("### Zero-Shot & Metric Depth Estimation for Images")

    # --- Sidebar for Configuration and Model Loading ---
    st.sidebar.header("Model Selection")
    
    # 1. Model Selection Dropdown
    model_selection = st.sidebar.selectbox(
        "Choose Model Variant:",
        options=list(MODEL_MAP.keys()),
        index=0,
        key='model_selector' # Ensure selection persists
    )
    
    model_id = MODEL_MAP[model_selection]
    st.sidebar.caption(f"HF ID: `{model_id}`")
    
    # Display status if model is loaded
    if st.session_state['depth_pipeline'] and st.session_state['loaded_model_id'] == model_selection:
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

    # 4. Run Inference Button (Conditionally Enabled)
    is_model_ready = st.session_state['depth_pipeline'] is not None and st.session_state['loaded_model_id'] == model_selection
    
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
            # Load the original image
            original_image = Image.open(uploaded_file).convert("RGB")
            
            # --- Run Model ---
            with st.spinner(f"Running inference with {model_selection}..."):
                depth_pipeline = st.session_state['depth_pipeline']

                # Run inference
                result = depth_pipeline(original_image)

                # Normalize output to extract depth image
                if isinstance(result, list) and result:
                    st.write("Model output is a list.") 
                    output = result[0]
                elif isinstance(result, dict):
                    st.write("Model output is a dict.")
                    keyname = list(result.keys())

                    output = result
                else:
                    raise RuntimeError(f"Unexpected model output type: {type(result)}")

                # Extract depth image
                depth_image_pil = output.get("depth") 
                
                if depth_image_pil is None:
                    raise KeyError(f"Depth image not found in model output keys: {list(output.keys())}")
                
            st.success("Inference complete!")

            # --- Visualization ---
            tabs = st.tabs(["Depth map","Original Image", "Depth heat Map"])
            
            # Display Original Image
            with tabs[0]:

                st.subheader("Depth Map Output (Plotly Chart)")
                st.image(depth_image_pil, width="stretch")
                # plotly_figure, plotly_config = create_depth_map_plotly_figure(depth_image_pil)
                
                
                if "Relative" in model_selection:
                    st.caption("Output values are unitless (closer/further relative order).")
                else:
                    st.caption("Output values are absolute metric depth (in meters).")
    
                
            # Display Depth Map using Plotly for better visualization/interactivity
            with tabs[1]:

                st.subheader("Original Image")
                st.image(original_image, width="stretch")
            
            with tabs[2]:
                st.write("Generating Plotly figure...")
                data_numpy = output.get("predicted_depth").cpu().detach().numpy()
                plotly_figure, plotly_config = create_depth_map_plotly_figure(data_numpy)
                st.plotly_chart(plotly_figure, config=plotly_config)

        except Exception as e:
            st.error(f"An error occurred during processing: {e}")
            # st.warning("Please ensure all required Python libraries are installed.")

    elif not is_model_ready and uploaded_file is None:
        st.info("Start by selecting a model and clicking 'Load Model', then upload an image.")
    elif is_model_ready and uploaded_file is None:
        st.info(f"Model **{model_selection}** is loaded. Please upload an image to run inference.")
    elif not is_model_ready and uploaded_file:
        st.info(f"Image uploaded. Click 'Load Model' to proceed with **{model_selection}**.")


if __name__ == "__main__":
    main()

