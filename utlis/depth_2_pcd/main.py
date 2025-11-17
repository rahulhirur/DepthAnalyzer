import numpy as np
import json
import open3d as o3d
import os

import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.io as pio


pio.renderers.default = "browser"

import datetime
from tqdm import tqdm

# This function is a wrapper to use Open3D's efficient point cloud creation
def create_o3d_point_cloud_from_numpy(points):
    """
    Converts a NumPy array of 3D points into an Open3D point cloud object.
    
    Args:
        points (np.ndarray): A NumPy array of shape (N, 3) containing the X, Y, Z coordinates.
    
    Returns:
        o3d.geometry.PointCloud: The Open3D point cloud object.
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd

def convert_depth_to_point_cloud(depth_map, fx, fy, cx, cy):
    """
    Converts a depth map to a 3D point cloud using camera intrinsics.
    This version is more efficient and handles invalid points better.

    Args:
        depth_map (np.ndarray): The 2D depth map.
        fx (float): Focal length in x.
        fy (float): Focal length in y.
        cx (float): Principal point in x.
        cy (float): Principal point in y.

    Returns:
        o3d.geometry.PointCloud: The generated point cloud.
    """
    height, width = depth_map.shape
    
    # Create coordinate grids
    # This vectorized approach is much faster than a nested loop
    u_map, v_map = np.meshgrid(np.arange(width), np.arange(height))
    
    # Flatten the maps and the depth map
    u = u_map.flatten()
    v = v_map.flatten()
    depth = depth_map.flatten()
    
    # Filter out invalid points (zero and NaN depths)
    valid_indices = np.where((depth > 0) & (~np.isnan(depth)))
    u_valid = u[valid_indices]
    v_valid = v[valid_indices]
    depth_valid = depth[valid_indices]
    
    # Calculate 3D points using the camera intrinsics
    X = (u_valid - cx) * depth_valid / fx
    Y = (v_valid - cy) * depth_valid / fy
    Z = depth_valid
    
    # Stack coordinates to form a point array
    points = np.stack((X, Y, Z), axis=1)
    
    # Create and return the Open3D point cloud object
    return create_o3d_point_cloud_from_numpy(points)

def save_point_cloud(point_cloud, output_base_dir, name_timestamp=True):
    """
    Saves the point cloud to a binary PLY file using Open3D for efficiency.

    Args:
        point_cloud (o3d.geometry.PointCloud): The point cloud to save.
        output_base_dir (str): The directory to save the file.
        name_timestamp (bool): If True, appends a timestamp to the filename.

    Returns:
        str: The path to the saved file, or None if an error occurs.
    """
    try:
        # Define a custom color class for console output
        class bcolors:
            OKGREEN = '\033[92m'
            ENDC = '\033[0m'
        
        if not os.path.exists(output_base_dir):
            os.makedirs(output_base_dir)
        
        if name_timestamp:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            outpath = os.path.join(output_base_dir, f'point_cloud_{timestamp}.ply')
        else:
            outpath = os.path.join(output_base_dir, 'point_cloud.ply')

        # Use o3d.io.write_point_cloud for efficient binary saving
        o3d.io.write_point_cloud(outpath, point_cloud, write_ascii=False)
        print(f"{bcolors.OKGREEN}Point cloud saved to {outpath}{bcolors.ENDC}")
        return outpath
    except Exception as e:
        print(f"Error saving point cloud: {e}")
        raise e
        return None

def visualize_point_cloud(pcd):
    """
    Visualizes a point cloud using Plotly.

    Args:
        pcd (open3d.geometry.PointCloud): The point cloud to visualize.
    """
    points = np.asarray(pcd.points)
    fig = go.Figure(data=[go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode='markers',
        marker=dict(size=2, color=points[:, 2], colorscale='Viridis', opacity=0.8)
    )])
    fig.update_layout(scene=dict(
        xaxis_title='X',
        yaxis_title='Y',
        zaxis_title='Z'
    ))
    return fig


# filepath1 = "/home/hirur/MasterThesis/3d_Reconstruction/Something.npy"
# #Load a npy file
# data = np.load(filepath1)

#filepath2 = "/home/hirur/MasterThesis/3d_Reconstruction/output/batch_1/ModelOutput/depthanything_v2/depth.npy"
#depth_data = np.load(filepath2)


#plotly heatmap


#plotly heatmap
# fig = px.imshow(depth_data, color_continuous_scale='Viridis')
# fig.update_layout(title="Numpy Array Heatmap", xaxis_title="X-axis", yaxis_title="Y-axis")
# fig.show()

#get the shape of the data
# print("Shape:", data.shape)
# print("Data type:", data.dtype)


def load_scaled_calibration_parameters(json_file_path):
    """
    Reads scaled camera calibration parameters from a JSON file and returns them as NumPy arrays.

    Args:
        json_file_path (str): The full path to the JSON file containing the parameters.

    Returns:
        dict or None: A dictionary containing the loaded parameters as NumPy arrays,
                      or None if an error occurs during loading.
                      The dictionary will contain keys:
                      'K1_scaled', 'D1_scaled', 'K2_scaled', 'D2_scaled',
                      'R_scaled', 'T_scaled', 'image_size_resized', 'scale_factor_applied'.
    """
    
    

    loaded_params = None
    try:
        streamlit_json_file = False
        if isinstance(json_file_path, str):
            
            with open(json_file_path, 'r') as f:
                loaded_data = json.load(f)
        
        elif hasattr(json_file_path, 'read'):
            streamlit_json_file = True
            loaded_data = json.load(json_file_path)
            # loaded_data = json.load(json_file_path)

        # Extract the required variables, converting lists back to numpy arrays
        K1_scaled = np.array(loaded_data["camera_matrix_1"])
        D1_scaled = np.array(loaded_data["dist_coeff_1"])
        K2_scaled = np.array(loaded_data["camera_matrix_2"])
        D2_scaled = np.array(loaded_data["dist_coeff_2"])
        R_scaled = np.array(loaded_data["Rot_mat"])
        T_scaled = np.array(loaded_data["Trans_vect"])

        # These might be lists/tuples or basic types, no need for np.array conversion
        image_size_resized = loaded_data.get("image_size_resized")
        scale_factor_applied = loaded_data.get("scale_factor_applied")
        image_size_actual = loaded_data.get("image_size_actual")

        loaded_params = {
            "K1": K1_scaled,
            "D1": D1_scaled,
            "K2": K2_scaled,
            "D2": D2_scaled,
            "R": R_scaled,
            "T": T_scaled,
            "image_size_actual": image_size_actual,
            "image_size_resized": image_size_resized,
            "scale_factor_applied": scale_factor_applied
        }

        if streamlit_json_file:
            print(f"Successfully loaded scaled calibration parameters from: {json_file_path.name}")
        else:
            print(f"Successfully loaded scaled calibration parameters from: {json_file_path}")


    except FileNotFoundError:
        print(f"Error: The file '{json_file_path}' was not found.")
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from the file '{json_file_path}'. Check file integrity.")
    except KeyError as e:
        print(f"Error: Missing expected key '{e}' in the JSON data. File format might be incorrect.")
    except Exception as e:
        print(f"An unexpected error occurred while reading the JSON file: {e}")

    return loaded_params

def depth_to_point_cloud(depth_map, fx, fy, cx, cy):

    height, width = depth_map.shape
    points = []
    
    #tqdm can be used for progress bar
    for v in tqdm(range(height)):
        for u in range(width):
            Z = depth_map[v, u]
            if Z == 0:
                continue
            X = (u - cx) * Z / fx
            Y = (v - cy) * Z / fy

            points.append((X, Y, Z))
    return np.array(points)

#point cloud to ply file
def save_point_cloud_to_ply(points, filename):
    with open(filename, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")
        for p in points:
            f.write(f"{p[0]} {p[1]} {p[2]}\n")


# plyFile = "/home/hirur/MasterThesis/3d_Reconstruction/output/batch_1/ModelOutput/depthanything_v2/depth_point_cloud.ply"

def depth_to_ply_file(depth_file, scaled_parameters_file, output_dir,tag):

    # depth_file = "/home/hirur/MasterThesis/3d_Reconstruction/output/batch_1/ModelOutput/depthanything_v2/depth.npy"
    # output_dir = "/home/hirur/MasterThesis/3d_Reconstruction/output/batch_1/ModelOutput/depthanything_v2"
    # scaled_parameters_file = "/home/hirur/MasterThesis/3d_Reconstruction/output/batch_1/scaled_calibration_parameters.json"
    
    scaled_params = load_scaled_calibration_parameters(scaled_parameters_file)

    if scaled_params is not None:

        fx = scaled_params["K1"][0, 0]
        fy = scaled_params["K1"][1, 1]
        cx = scaled_params["K1"][0, 2]
        cy = scaled_params["K1"][1, 2]
        
        print(f"Using fx: {fx}, fy: {fy}, cx: {cx}, cy: {cy}")

        depth_map = np.load(depth_file)

        intrinsic = o3d.camera.PinholeCameraIntrinsic()
        intrinsic.set_intrinsics(width=depth_map.shape[1], height=depth_map.shape[0],
                                fx=fx, fy=fy, cx=cx, cy=cy)

        depth_o3d = o3d.geometry.Image(depth_map.astype(np.uint16))
        # Create point cloud from depth
        pcd = o3d.geometry.PointCloud.create_from_depth_image(
            depth_o3d,
            intrinsic,
            depth_scale=1000.0,  # if depth is in millimeters, scale to meters
            depth_trunc=3.0,     # cut points farther than 3m
            stride=1             # use every pixel
        )

        print(f"Point cloud has {len(pcd.points)} points.")

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        outpath_ply_file = os.path.join(output_dir, f'{tag}_point_cloud_{timestamp}.ply')
        
        # Save point cloud to file

        o3d.io.write_point_cloud(outpath_ply_file, pcd)
        print(f"Point cloud saved to {outpath_ply_file}")