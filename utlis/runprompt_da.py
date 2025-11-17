import requests
from PIL import Image
import numpy as np
from transformers import PromptDepthAnythingForDepthEstimation, PromptDepthAnythingImageProcessor
import torch

url = "https://github.com/DepthAnything/PromptDA/blob/main/assets/example_images/image.jpg?raw=true"
url = "/home/raptor/hirurrahul_prjs/output/PSP/Input/cam1_1_0.png"

image = Image.open(url)


image_processor = PromptDepthAnythingImageProcessor.from_pretrained("depth-anything/prompt-depth-anything-vitl-hf")
model = PromptDepthAnythingForDepthEstimation.from_pretrained("depth-anything/prompt-depth-anything-vitl-hf")

prompt_depth_url = "/home/raptor/hirurrahul_prjs/output/PSP/Depth_Image/Depth_Map_12-11_19-07.png"
prompt_depth = Image.open(prompt_depth_url)


def to_pil(img):
    """Normalize various image types to a PIL.Image in RGB (when appropriate).

    Accepts PIL.Image, numpy arrays (H,W,C or H,W), and torch tensors.
    Returns a PIL.Image.
    """
    # PIL Image
    if isinstance(img, Image.Image):
        try:
            return img.convert("RGB")
        except Exception:
            return img

    # numpy array
    if isinstance(img, np.ndarray):
        if img.ndim == 2:  # single channel
            return Image.fromarray(img).convert("L")
        if img.ndim == 3:
            # if channels first, transpose
            if img.shape[0] in (1, 3, 4):
                arr = np.moveaxis(img, 0, -1)
            else:
                arr = img
            # drop alpha if present
            if arr.shape[2] == 4:
                arr = arr[..., :3]
            return Image.fromarray(arr).convert("RGB")

    # torch tensor
    try:
        import torch as _torch

        if _torch.is_tensor(img):
            arr = img.detach().cpu().numpy()
            return to_pil(arr)
    except Exception:
        pass

    raise TypeError(f"Unsupported image type: {type(img)}")


def to_pil_depth(img):
    """Convert input to a single-channel (grayscale) PIL image for prompt_depth."""
    # PIL Image
    if isinstance(img, Image.Image):
        try:
            return img.convert("L")
        except Exception:
            return img

    # numpy array
    if isinstance(img, np.ndarray):
        if img.ndim == 2:
            return Image.fromarray(img).convert("L")
        if img.ndim == 3:
            # if channels first, transpose
            if img.shape[0] in (1, 3, 4):
                arr = np.moveaxis(img, 0, -1)
            else:
                arr = img
            # if RGB, convert to grayscale
            if arr.shape[2] == 4:
                arr = arr[..., :3]
            pil = Image.fromarray(arr)
            return pil.convert("L")

    # torch tensor
    try:
        import torch as _torch

        if _torch.is_tensor(img):
            arr = img.detach().cpu().numpy()
            return to_pil_depth(arr)
    except Exception:
        pass

    raise TypeError(f"Unsupported prompt_depth type: {type(img)}")


# Ensure the processor receives a batch (list) of PIL images.
images_batch = image if isinstance(image, (list, tuple)) else [image]

prompt_depth_batch = prompt_depth if isinstance(prompt_depth, (list, tuple)) else [prompt_depth]

# Normalize all entries to PIL images and convert to RGB where appropriate
images_batch = [to_pil(x) for x in images_batch]
# prompt_depth should be single-channel depth map

prompt_depth_batch = [to_pil_depth(x) for x in prompt_depth_batch]

print("DEBUG: images_batch type:", type(images_batch), "first element type:", type(images_batch[0]))
print("DEBUG: prompt_depth_batch type:", type(prompt_depth_batch), "first element type:", type(prompt_depth_batch[0]))

try:
    inputs = image_processor(images=images_batch, return_tensors="pt", prompt_depth=prompt_depth_batch)
except Exception as e:
    print("ERROR while calling image_processor:", repr(e))
    # print more diagnostics
    first = images_batch[0]
    print("First image instance:", first)
    try:
        print("First image mode/size:", getattr(first, 'mode', None), getattr(first, 'size', None))
    except Exception:
        pass
    raise


print("DEBUG: inputs keys:", list(inputs.keys()))
for k, v in inputs.items():
    try:
        # some items may be lists (e.g., pixel values) or tensors
        if hasattr(v, 'shape'):
            print(f" - {k}: type={type(v)}, shape={v.shape}, dtype={getattr(v, 'dtype', None)}")
        else:
            print(f" - {k}: type={type(v)}")
    except Exception as _:
        print(f" - {k}: (failed to get shape) type={type(v)}")

# Normalize prompt_depth tensor dimensionality for interpolation expectations
if 'prompt_depth' in inputs:
    pd = inputs['prompt_depth']
    if isinstance(pd, torch.Tensor):
        print("DEBUG: prompt_depth tensor before norm ndim=", pd.ndim, "shape=", pd.shape)
        # remove any singleton spatial dims
        while pd.ndim > 4 and 1 in pd.shape[2:]:
            # squeeze the first singleton spatial dim we find (from left)
            for i in range(2, pd.ndim):
                if pd.shape[i] == 1:
                    pd = pd.squeeze(dim=i)
                    break

        # If channels are last (N, H, W, C), move to channels-first
        if pd.ndim == 4 and pd.shape[-1] in (1, 3, 4):
            pd = pd.permute(0, 3, 1, 2)
            print("DEBUG: prompt_depth permuted to channels-first ->", pd.shape)

        # If missing channel dim (N, H, W), add channel dim
        if pd.ndim == 3:
            pd = pd.unsqueeze(1)
            print("DEBUG: prompt_depth unsqueezed to add channel dim ->", pd.shape)

        # Reduce multi-channel prompt_depth to single channel (depth) by averaging channels
        if pd.ndim == 4 and pd.shape[1] != 1:
            pd = pd.mean(dim=1, keepdim=True)
            print("DEBUG: prompt_depth reduced to single channel ->", pd.shape)

        # if still not 4D, warn
        if pd.ndim != 4:
            print("WARNING: prompt_depth has unexpected ndim after normalization:", pd.ndim)

        inputs['prompt_depth'] = pd
    else:
        print("DEBUG: prompt_depth is not a tensor; type:", type(pd))

with torch.no_grad():
    outputs = model(**inputs)
post_processed_output = image_processor.post_process_depth_estimation(
    outputs,
    target_sizes=[(image.height, image.width)],
)

predicted_depth = post_processed_output[0]["predicted_depth"]

print(predicted_depth.size())  # torch.Size([1, H, W])

print("Depth estimation completed.")

print('Type of predicted_depth:', type(predicted_depth))


print('Visualize the predicted depth map using matplotlib:')



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
        print(f"Error creating point cloud figure: {e}")
        return None







def visualize_pcd_from_depth(depth_data:np.ndarray, selected_colorscale:str=None):

    # downsample for faster visualization if too large
    max_points = 200000  # Maximum number of points to visualize
    print(f"Original depth data size: {depth_data.size} points.")
    if depth_data.size > max_points:
        factor = int(np.ceil(np.sqrt(depth_data.size / max_points)))
        depth_data = depth_data[::factor, ::factor]
        print(f"Depth data downsampled by a factor of {factor} for visualization.")

    '#Meshgrid Creation for Point Cloud'
    h, w = depth_data.shape
    xx, yy = np.meshgrid(np.arange(w), np.arange(h))
    x = xx.flatten()
    y = yy.flatten()
    z = depth_data.flatten()

    #z value is below 3.9 make it zero to remove floor noise and double the z values for better visibility
    #z = np.where(z < 3.5, 0, z*100)    

    #Absolute Scale Inverted Filter
    z = np.where(z > 0, z*100,0)
    z = np.where(z > 5.8, z*5,0)

    if selected_colorscale is None:
        selected_colorscale = 'viridis'
    
    if x is not None:

        fig = plot_scatter3d(x,y,z, selected_colorscale)

        if fig is not None:
            fig.show()
        else:
            print("Failed to create figure from PLY data.")




data_numpy = predicted_depth.squeeze().cpu().numpy()


# Number of points in the sample of depth map
print('Depth map shape:', data_numpy.shape)
print('Number of points in depth map:', data_numpy.size)





import matplotlib.pyplot as plt
plt.imshow(data_numpy, cmap='plasma')
plt.colorbar()
plt.title('Predicted Depth Map')
plt.show()

# Visualize the predicted depth a 3d scatter plot using plotly
print('3D Scatter Plot of Predicted Depth:')
import numpy as np

import plotly.graph_objs as go
visualize_pcd_from_depth(data_numpy, selected_colorscale='plasma')