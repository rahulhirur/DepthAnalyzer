import streamlit as st
import numpy as np
import io
from PIL import Image
from transformers import pipeline
import torch # Moved to top to ensure availability in the cached function
from helper.create_depth_map_plotly_figure import create_depth_map_plotly_figure

# --- Configuration: Model Mappings ---
# Maps the user-friendly name to the Hugging Face Model ID
MODEL_MAP = {
    "Relative (Small - 25M)": "depth-anything/Depth-Anything-V2-Small-hf",
    "Relative (Base - 97M)": "depth-anything/Depth-Anything-V2-Base-hf",
    "Relative (Large - 335M)": "depth-anything/Depth-Anything-V2-Large-hf",
    "Metric (Indoor Large)": "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf",
    "Metric (Outdoor Large)": "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf",
}

# --- Utility Function: Model Loading (Cached) ---
@st.cache_resource
def load_model(model_id: str):
    """
    Loads the Hugging Face depth estimation pipeline.
    Uses st.cache_resource to load the model only once, speeding up the app.
    Uses st.status to show the loading process.
    """
    # START: Using st.status for model loading progress
    with st.status(f"Downloading and loading model: **{model_id}**...", expanded=True) as status:
        try:
            # The pipeline handles model loading, preprocessing, and inference steps.
            depth_pipe = pipeline(
                task="depth-estimation", 
                model=model_id, 
                # Some Hugging Face model repos include custom Python code that
                # registers TorchScript/torch.classes or custom model classes.
                # Enabling `trust_remote_code=True` allows loading that code so
                # those registrations are executed and the model can be instantiated.
                # SECURITY: This executes model code from the model repo – only
                # set this to True for trusted model sources (e.g. official HF
                # repos or repositories you trust).
                trust_remote_code=True,
                device="cuda:0" if st.session_state.get('use_gpu', False) and torch.cuda.is_available() else "cpu"
            )
            
            # Update status on success
            status.update(label=f"Model {model_id.split('/')[-1]} loaded successfully!", state="complete", expanded=False)
            return depth_pipe
        except Exception as e:
            # Update status on error
            status.update(label=f"Failed to load model {model_id.split('/')[-1]}", state="error", expanded=True)
            # Re-raise the exception for the main app's error handler
            raise e
    # END: Using st.status
    

# --- Utility Function: Image Processing & Saving ---
def convert_depth_map_to_bytes(depth_image: Image.Image) -> bytes:
    """Converts the PIL depth image to a byte buffer for download."""
    buf = io.BytesIO()
    # Save the 16-bit depth map to the buffer
    depth_image.save(buf, format='PNG')
    return buf.getvalue()

# --- Streamlit App Layout ---
def main():
    st.set_page_config(layout="wide", page_title="Depth-Anything V2 Estimator")
    st.title("Depth-Anything V2 Estimator")
    st.markdown("### Zero-Shot & Metric Depth Estimation for Images")

    # --- Sidebar for Configuration ---
    st.sidebar.header("Configuration")
    
    # 1. Model Selection
    model_selection = st.sidebar.selectbox(
        "Choose Model Variant:",
        options=list(MODEL_MAP.keys()),
        index=2 # Default to Relative (Large)
    )
    
    model_id = MODEL_MAP[model_selection]
    st.sidebar.caption(f"HF ID: `{model_id}`")
    
    # 2. File Uploader
    uploaded_file = st.sidebar.file_uploader(
        "Upload an Image (.jpg, .png)", 
        type=["jpg", "jpeg", "png"]
    )

    # 3. Run Button
    run_button = st.sidebar.button("Run Depth Estimation", type="primary", use_container_width=True)

    st.sidebar.markdown("---")
    st.sidebar.caption("The **Relative** models provide unitless depth indicating distance order. The **Metric** models are fine-tuned to output depth in meters.")

    # --- Main Content Area ---
    if run_button and uploaded_file:
        try:
            # Load the original image
            original_image = Image.open(uploaded_file).convert("RGB")
            
            # --- Load and Run Model ---
            with st.spinner(f"Running inference with {model_selection}..."):
                # Load the cached pipeline
                depth_pipeline = load_model(model_id)

                # Run inference
                result = depth_pipeline(original_image)

                # The pipeline can return either a list of dicts or a single dict
                # depending on transformers version / model implementation. Normalize
                # both cases to a single dict (`output`) and then extract the
                # depth image robustly.
                if isinstance(result, list) and result:
                    output = result[0]
                elif isinstance(result, dict):
                    output = result
                else:
                    raise RuntimeError(f"Unexpected model output type: {type(result)}")

                # Common keys may be 'depth' or 'depth_map' depending on model.
                depth_image_pil = None
                for k in ("depth", "depth_map"):
                    if k in output:
                        depth_image_pil = output[k]
                        break
                if depth_image_pil is None:
                    raise KeyError(f"Depth image not found in model output keys: {list(output.keys())}")
                
            st.success("Inference complete!")

            # --- Visualization ---
            col1, col2 = st.columns(2)
            
            # Display Original Image
            with col1:
                st.subheader("Original Image")
                st.image(original_image, use_container_width=True)
            
            # Display Depth Map
            with col2:
                st.subheader("Depth Map Output (Relative or Metric)")
                # The PIL depth map is often grayscale. Streamlit uses a default colormap, but 
                # we'll display the image and provide a specific visual clue.
                figure_to_display = create_depth_map_plotly_figure(depth_image_pil)
                st.plotly_chart(figure_to_display, use_container_width=True)
                
                if "Relative" in model_selection:
                    st.caption("Visualization uses a color map (darker = closer, lighter = further). Output values are unitless.")
                else:
                    st.caption("Visualization uses a color map. Output values are absolute metric depth (in meters).")


            # --- Save Depth Map (.png) ---
            st.markdown("---")
            st.subheader("Download Depth Map")

            # Convert the depth image (which is 16-bit PIL image) to bytes
            depth_bytes = convert_depth_map_to_bytes(depth_image_pil)
            
            # Use the input filename with a new suffix
            file_name = uploaded_file.name.split('.')[0]
            download_filename = f"{file_name}_{model_selection.replace(' ', '_')}.png"

            st.download_button(
                label="Download Depth Map (16-bit PNG)",
                data=depth_bytes,
                file_name=download_filename,
                mime="image/png",
                key="download_button",
                type="secondary",
                help="The downloaded PNG file contains raw 16-bit depth values for quantitative analysis."
            )

        except Exception as e:
            st.error(f"An error occurred during processing: {e}")
            st.warning("Ensure all required Python libraries (`streamlit`, `transformers`, `torch`, `Pillow`) are installed.")

    elif not uploaded_file and not run_button:
        st.info("Upload an image in the sidebar and click 'Run Depth Estimation' to start.")

if __name__ == "__main__":
    # The torch import has been moved to the top of the file.
    main()
