import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

import config


# Apply Iterative Closest Point (ICP) to align two point clouds
def apply_icp(source, target, max_iterations=50, threshold=0.02):
    reg_icp = o3d.pipelines.registration.registration_icp(
        source, target, threshold, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
    return reg_icp.transformation

# Align all point clouds to the first point cloud (or any chosen base)
def align_point_clouds(point_clouds):
    aligned_clouds = [point_clouds[0]]  # Keep the first one as reference
    for i in range(1, len(point_clouds)):
        source = point_clouds[i]
        target = aligned_clouds[-1]

        # Apply ICP to align the source to the target
        transformation = apply_icp(source, target)

        # Transform the source point cloud
        source.transform(transformation)

        # Add the aligned point cloud to the list
        aligned_clouds.append(source)

    return aligned_clouds


# Fuse all point clouds into one
def fuse_point_clouds(point_clouds):
    fused_point_cloud = o3d.geometry.PointCloud()
    for pcd in point_clouds:
        fused_point_cloud += pcd
    return fused_point_cloud


def generate_point_cloud(depth_map, K, min_depth=1e-6):
    height, width = depth_map.shape
    u, v = np.meshgrid(np.arange(width), np.arange(height))

    # Extract intrinsic parameters from the camera matrix K
    fx = K[0, 0]  # Focal length in x direction
    fy = K[1, 1]  # Focal length in y direction
    cx = K[0, 2]  # Principal point in x direction
    cy = K[1, 2]  # Principal point in y direction

    # # Apply depth thresholding to remove invalid depth values
    max_depth = np.percentile(depth_map, 99)
    mask = (depth_map <= 240) | (depth_map >= 250)
    depth_map = np.where(mask, depth_map, np.nan)  # Replace invalid depth values with NaN

    # Compute 3D coordinates only for valid points
    X = (u - cx) * depth_map / fx
    Y = (v - cy) * depth_map / fy
    Z = depth_map

    # Stack into (N, 3) format and remove NaN values
    points = np.stack((X.flatten(), Y.flatten(), Z.flatten()), axis=-1)
    points = points[~np.isnan(points).any(axis=1)]  # Remove NaN values

    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # light_gray = np.tile([0.78, 0.78, 0.78], (len(points), 1))
    # pcd.colors = o3d.utility.Vector3dVector(light_gray)

    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

    # Extract the inlier (good) point cloud
    pcd_cleaned = pcd.select_by_index(ind)

    return pcd_cleaned

def generate_point_cloud_from_3d_grid(pts_3d, max_depth=None):
    X = pts_3d[:, :, 0]
    Y = pts_3d[:, :, 1]
    Z = pts_3d[:, :, 2]

    mask = (Z > 1e-6)
    if max_depth is not None:
        mask &= (Z < max_depth)

    points = np.stack((X[mask], Y[mask], Z[mask]), axis=-1)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    if len(pcd.points) > 20:
        cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        pcd = pcd.select_by_index(ind)

    return pcd


def visualize_depth_map(depth_map, title):
    """Visualize the depth map using matplotlib."""
    plt.figure()
    plt.imshow(depth_map, cmap="jet")
    plt.colorbar(label="Depth (in mm)")
    plt.title(title)
    plt.show()

def compute_depth_map_from_unwrapped_phase(unwrapped_phase, fx):

    # Avoid division by zero and very small values
    min_phase_value = 1e-3  # Adjust based on your data
    unwrapped_phase = np.where(unwrapped_phase < min_phase_value, min_phase_value, unwrapped_phase)

    # Compute depth using the formula
    distance_from_ref = 1600 # in mm
    depth_map = (
        distance_from_ref  * unwrapped_phase
    ) / config.CAMERA_PROJECTOR_BASELINE

    # # Handle outliers
    # min_depth = 180  # Minimum expected depth
    # max_depth = 1000  # Maximum expected depth (adjust based on your application)
    # depth_map = np.clip(depth_map, min_depth, max_depth)

    return depth_map


def visualize_point_cloud(pcd, title):
    """Visualize the point cloud using open3d."""
    print(f"Visualizing {title}")
    o3d.visualization.draw_geometries([pcd], window_name=title)
