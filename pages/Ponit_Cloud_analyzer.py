import streamlit as st
import numpy as np
import io
from PIL import Image
from transformers import pipeline
from helper.create_depth_map_plotly_figure import create_depth_map_plotly_figure, colorize_depth_map_pil
from scipy.spatial import KDTree
import open3d as o3d
import copy

#plotly go import
import plotly.graph_objects as go
from sklearn.neighbors import KDTree


#add session state for generated point cloud x,y,z
if 'point_cloud_x' not in st.session_state:
    st.session_state.point_cloud_x = None
if 'point_cloud_y' not in st.session_state:
    st.session_state.point_cloud_y = None
if 'point_cloud_z' not in st.session_state:
    st.session_state.point_cloud_z = None

if 'ground_truth_x' not in st.session_state:
    st.session_state.ground_truth_x = None
if 'ground_truth_y' not in st.session_state:
    st.session_state.ground_truth_y = None
if 'ground_truth_z' not in st.session_state:
    st.session_state.ground_truth_z = None

if 'point_cloud_x_final' not in st.session_state:
    st.session_state.point_cloud_x_final = None
if 'point_cloud_y_final' not in st.session_state:
    st.session_state.point_cloud_y_final = None
if 'point_cloud_z_final' not in st.session_state:
    st.session_state.point_cloud_z_final = None


if 'ground_truth_x_final' not in st.session_state:
    st.session_state.ground_truth_x_final = None
if 'ground_truth_y_final' not in st.session_state:
    st.session_state.ground_truth_y_final = None
if 'ground_truth_z_final' not in st.session_state:
    st.session_state.ground_truth_z_final = None
    




def normalize_to_unit_cube(x, y, z):
            """
            Normalize point cloud coordinates to the range [0, 1] independently per axis.

            Args:
                x: numpy array of x coordinates
                y: numpy array of y coordinates
                z: numpy array of z coordinates

            Returns:
                tuple: (normalized_x, normalized_y, normalized_z) where each component
                       is scaled to the [0, 1] range. If an axis has zero range (constant
                       values) the returned normalized values for that axis will be 0.0.
            """
            x = np.asarray(x)
            y = np.asarray(y)
            z = np.asarray(z)

            # Per-axis min/max
            min_x, max_x = np.min(x), np.max(x)
            min_y, max_y = np.min(y), np.max(y)
            min_z, max_z = np.min(z), np.max(z)

            range_x = max_x - min_x
            range_y = max_y - min_y
            range_z = max_z - min_z

            # Avoid division by zero: if range is zero, return zeros for that axis
            if range_x == 0:
                normalized_x = np.zeros_like(x, dtype=float)
            else:
                normalized_x = (x - min_x) / range_x

            if range_y == 0:
                normalized_y = np.zeros_like(y, dtype=float)
            else:
                normalized_y = (y - min_y) / range_y

            if range_z == 0:
                normalized_z = np.zeros_like(z, dtype=float)
            else:
                normalized_z = (z - min_z) / range_z

            return normalized_x, normalized_y, normalized_z



def icp_alignment(source_x, source_y, source_z, target_x, target_y, target_z, max_iterations=100, tolerance=1e-6):
            """
            Perform Iterative Closest Point (ICP) algorithm to align source point cloud to target point cloud.
            
            Args:
                source_x, source_y, source_z: Source point cloud coordinates (to be aligned)
                target_x, target_y, target_z: Target point cloud coordinates (reference)
                max_iterations: Maximum number of ICP iterations
                tolerance: Convergence tolerance for mean squared error
            
            Returns:
                tuple: (aligned_x, aligned_y, aligned_z, rotation_matrix, translation_vector, final_error)
            """
            
            # Stack points into (N, 3) arrays
            source_points = np.vstack((source_x, source_y, source_z)).T
            target_points = np.vstack((target_x, target_y, target_z)).T
            
            # Initialize transformation
            R = np.eye(3)  # Rotation matrix
            t = np.zeros(3)  # Translation vector
            
            transformed_source = source_points.copy()
            prev_error = float('inf')
            
            for iteration in range(max_iterations):
                # Build KD-tree for fast nearest neighbor search
                tree = KDTree(target_points)
                
                # Find closest points in target for each source point
                distances, indices = tree.query(transformed_source)
                # Ensure 1D arrays (KDTree.query may return shape (N,1) if k=1)
                distances = np.atleast_1d(distances).reshape(-1)
                indices = np.atleast_1d(indices).reshape(-1)
                closest_points = target_points[indices]
                
                # Compute centroids
                source_centroid = np.mean(transformed_source, axis=0)
                target_centroid = np.mean(closest_points, axis=0)
                
                # Center the point clouds
                source_centered = transformed_source - source_centroid
                target_centered = closest_points - target_centroid
                
                # Compute cross-covariance matrix
                H = source_centered.T @ target_centered
                
                # SVD for optimal rotation
                U, _, Vt = np.linalg.svd(H)
                R_iter = Vt.T @ U.T
                
                # Ensure proper rotation (det(R) = 1)
                if np.linalg.det(R_iter) < 0:
                    Vt[-1, :] *= -1
                    R_iter = Vt.T @ U.T
                
                # Compute translation
                t_iter = target_centroid - R_iter @ source_centroid
                
                # Apply transformation
                transformed_source = (R_iter @ transformed_source.T).T + t_iter
                
                # Update cumulative transformation
                R = R_iter @ R
                t = R_iter @ t + t_iter
                
                # Compute mean squared error
                error = np.mean(distances**2)
                
                # Check convergence
                if abs(prev_error - error) < tolerance:
                    st.info(f"ICP converged at iteration {iteration + 1} with error {error:.6f}")
                    break
                
                prev_error = error
            else:
                st.warning(f"ICP reached max iterations ({max_iterations}) with error {error:.6f}")
            
            # Extract final aligned coordinates
            aligned_x = transformed_source[:, 0]
            aligned_y = transformed_source[:, 1]
            aligned_z = transformed_source[:, 2]
            
            return aligned_x, aligned_y, aligned_z, R, t, error


def get_pca_transform(source_points, target_points):
    """
    Calculates a coarse alignment using Principal Component Analysis (PCA).
    Aligns centroids and principal axes.
    """
    # 1. Center both clouds
    source_centroid = np.mean(source_points, axis=0)
    target_centroid = np.mean(target_points, axis=0)
    
    source_centered = source_points - source_centroid
    target_centered = target_points - target_centroid
    
    # 2. Compute Covariance Matrices
    cov_source = np.cov(source_centered.T)
    cov_target = np.cov(target_centered.T)
    
    # 3. Eigen Decomposition (SVD is stable)
    U_s, _, _ = np.linalg.svd(cov_source)
    U_t, _, _ = np.linalg.svd(cov_target)
    
    # 4. Compute Rotation R = U_t * U_s^T
    R = U_t @ U_s.T
    
    # Handle reflection case (det(R) = -1)
    if np.linalg.det(R) < 0:
        U_t[:, -1] *= -1
        R = U_t @ U_s.T
        
    # 5. Compute Translation
    t = target_centroid - R @ source_centroid
    
    return R, t


def get_random_rotations(n=10):
    """Generates n random 3x3 rotation matrices."""
    rotations = []
    for _ in range(n):
        # QR decomposition of a random matrix guarantees a random orthogonal matrix
        H = np.random.randn(3, 3)
        Q, R = np.linalg.qr(H)

        # Ensure determinant is +1 (proper rotation, not reflection)
        if np.linalg.det(Q) < 0:
            Q[:, -1] *= -1
        rotations.append(Q)
    return rotations


def icp_alignment_robust(source_x, source_y, source_z, target_x, target_y, target_z, 
                        max_iterations=100, tolerance=1e-6, outlier_rejection_ratio=0.9,
                        init_method='pca', num_restarts=10):
    """
    Robust ICP with options for PCA or Random Restart initialization.
    
    Args:
        init_method (str): 'pca' for principal axes alignment, 'random' for random restart.
        num_restarts (int): Number of random rotations to try if init_method='random'.
    """
    # Stack points
    source_points = np.vstack((source_x, source_y, source_z)).T
    target_points = np.vstack((target_x, target_y, target_z)).T
    
    # Build KD-tree once (target doesn't move)
    tree = KDTree(target_points)
    
    # Calculate Centroids for initialization
    source_centroid = np.mean(source_points, axis=0)
    target_centroid = np.mean(target_points, axis=0)
    
    # --- STEP 1: COARSE ALIGNMENT ---
    best_R_init = np.eye(3)
    best_t_init = target_centroid - source_centroid # Default: just align centers
    
    if init_method == 'pca':
        st.info("Running PCA Coarse Alignment...")
        best_R_init, best_t_init = get_pca_transform(source_points, target_points)
        
    elif init_method == 'random':
        st.info(f"Running Random Restart Alignment ({num_restarts} attempts)...")
        best_score = float('inf')
        
        # Generate random rotations + always include Identity as a candidate
        rotations = get_random_rotations(num_restarts)
        rotations.append(np.eye(3))
        
        source_centered = source_points - source_centroid
        
        for R_try in rotations:
            # 1. Rotate source around its centroid
            # 2. Move source centroid to target centroid
            # Transformation: P_new = R_try @ (P_old - C_s) + C_t
            transformed_try = (R_try @ source_centered.T).T + target_centroid
            
            # Fast Check: Use a random subset of 500 points to score this rotation
            subset_indices = np.random.choice(len(transformed_try), min(500, len(transformed_try)), replace=False)
            distances, _ = tree.query(transformed_try[subset_indices])
            
            score = np.mean(distances**2)
            
            if score < best_score:
                best_score = score
                best_R_init = R_try
                # Calculate the implicit translation vector t for P_new = R*P + t
                # P_new = R*P - R*C_s + C_t  =>  t = C_t - R*C_s
                best_t_init = target_centroid - best_R_init @ source_centroid
        
        st.info(f"Best random init found with error score: {best_score:.4f}")

    # Apply the best initialization
    transformed_source = (best_R_init @ source_points.T).T + best_t_init
    R_accum = best_R_init
    t_accum = best_t_init
    
    prev_error = float('inf')
    
    # --- STEP 2: FINE ALIGNMENT (ICP) ---
    st.info("Starting Fine ICP...")
    for iteration in range(max_iterations):
        # Find closest points
        distances, indices = tree.query(transformed_source)
        # Ensure 1D arrays (KDTree.query may return shape (N,1) if k=1)
        distances = np.atleast_1d(distances).reshape(-1)
        indices = np.atleast_1d(indices).reshape(-1)
        
        # --- Outlier Rejection ---
        # Only keep the closest X% of points (rejects noise and non-overlapping regions)
        threshold = np.quantile(distances, outlier_rejection_ratio)
        valid_indices = distances < threshold
        
        if np.sum(valid_indices) < 10: # Safety check
            st.warning("Too few matching points, stopping.")
            break
            
        src_subset = transformed_source[valid_indices]
        tgt_subset = target_points[indices[valid_indices]]
        
        # Compute Centroids of the SUBSET
        source_centroid_iter = np.mean(src_subset, axis=0)
        target_centroid_iter = np.mean(tgt_subset, axis=0)
        
        # Center clouds
        source_centered_iter = src_subset - source_centroid_iter
        target_centered_iter = tgt_subset - target_centroid_iter
        
        # Cross-covariance
        H = source_centered_iter.T @ target_centered_iter
        
        # SVD
        U, _, Vt = np.linalg.svd(H)
        R_iter = Vt.T @ U.T
        
        # Reflection check
        if np.linalg.det(R_iter) < 0:
            Vt[-1, :] *= -1
            R_iter = Vt.T @ U.T
            
        t_iter = target_centroid_iter - R_iter @ source_centroid_iter
        
        # Update points
        transformed_source = (R_iter @ transformed_source.T).T + t_iter
        
        # Update cumulative transformation
        R_accum = R_iter @ R_accum
        t_accum = R_iter @ t_accum + t_iter
        
        # Error (on valid subset only)
        error = np.mean(distances[valid_indices]**2)
        
        if abs(prev_error - error) < tolerance:
            st.info(f"Converged at iteration {iteration}")
            break
        prev_error = error
        
    return transformed_source[:, 0], transformed_source[:, 1], transformed_source[:, 2], R_accum, t_accum, error


def ransac_global_registration(source_x, source_y, source_z, target_x, target_y, target_z, voxel_size=0.05):
    """
    Performs Global Registration using RANSAC and FPFH features via Open3D.
    
    Args:
        source_x, source_y, source_z: Source point cloud arrays
        target_x, target_y, target_z: Target point cloud arrays
        voxel_size: Downsampling scale (units dependent on your data, e.g., meters)
        
    Returns:
        aligned_source (numpy array), transformation_matrix (4x4)
    """
    
    # 1. Convert NumPy to Open3D PointClouds
    source = o3d.geometry.PointCloud()
    target = o3d.geometry.PointCloud()
    
    source_points = np.vstack((source_x, source_y, source_z)).T
    target_points = np.vstack((target_x, target_y, target_z)).T
    
    source.points = o3d.utility.Vector3dVector(source_points)
    target.points = o3d.utility.Vector3dVector(target_points)

    # 2. Preprocessing: Downsample + Compute Features (FPFH)
    # RANSAC needs features to know which points "look" similar
    def preprocess_point_cloud(pcd, voxel_size):
        pcd_down = pcd.voxel_down_sample(voxel_size)
        
        # Estimate normals (required for FPFH)
        radius_normal = voxel_size * 2
        pcd_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
        
        # Compute FPFH features
        radius_feature = voxel_size * 5
        pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcd_down,
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
        return pcd_down, pcd_fpfh

    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)

    # 3. Run RANSAC
    # Matches features and finds the best rigid transform
    distance_threshold = voxel_size * 1.5
    st.info(f"Running RANSAC with voxel_size={voxel_size}...")
    
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        True, # Mutual filter (checks consistency)
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, # Sample size (3 points define a rigid transform)
        [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
        ],
        o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999)
    )
    
    st.info(f"RANSAC Fitness Score: {result.fitness:.4f}")
    st.info(f"RANSAC Inlier RMSE: {result.inlier_rmse:.4f}")

    # 4. Apply Result
    transformation = result.transformation
    source.transform(transformation)
    
    # Convert back to NumPy
    aligned_points = np.asarray(source.points)
    
    return aligned_points, transformation



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

def visualize_pcd_from_depth(depth_data: np.ndarray, metric: bool = False, threshold: float = 3.5, scale_factor: float = 100.0, clean_point_cloud_bool = True, selected_colorscale: str = None):

    """
    Create and display a 3D point cloud from a 2D depth map.

    Args:
        depth_data: 2D numpy array of depth (Z) values.
        metric: If True, apply metric-based thresholding (assumes depth in meters).
        threshold: Threshold (in same units as depth_data) used to zero out far values when metric=True.
        selected_colorscale: Plotly colorscale name.
    """
    max_points = 200000  # Maximum number of points to visualize

    if depth_data.shape[1]==3:
        st.write('This is a processed point cloud data with x,y,z coordinates in the shape (N,3)')
        x = depth_data[:,0]
        y = depth_data[:,1]
        z = depth_data[:,2].astype(float)

        # figx = plot_scatter3d(x, y, z, selected_colorscale)
        # st.plotly_chart(figx, use_container_width=True)

        max_points = 200000  # Maximum number of points to visualize
        
        if depth_data.shape[0] > max_points:
            factor = int(np.ceil(depth_data.shape[0] / max_points))
            x = x[::factor]
            y = y[::factor]
            z = z[::factor]
            st.info(f"Point cloud downsampled by a factor of {factor} for visualization.")
    
    else:

        # downsample for faster visualization if too large
        
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

    if clean_point_cloud_bool:
        x,y,z, purge_pts = clean_point_cloud(x, y, z)

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


tabs = st.tabs(["Generated 3D Point Cloud Visualization", "Ground Truth Point Cloud Visualization", "Point Cloud Comparison", "Partial Cloud Aligment"])

with tabs[0]:
    if generated_file is not None:    
        st.subheader("3D Point Cloud Visualization")

        st.write('Point cloud')

        # Add st.checkbox to activate thresholding, scaling, and cleaning options in st columns

        colopts1 = st.columns(3)

        threshold_checkbox = colopts1[0].checkbox(
            "Enable Depth Thresholding",
            value=True,
            help="If checked, allows setting a depth threshold to filter points.",
            key="threshold_checkbox_generated2"
        )

        scaling_checkbox = colopts1[1].checkbox(
            "Enable Depth Scaling",
            value=True,
            help="If checked, allows setting a scale factor for depth values.",
            key="scaling_checkbox_generated2"
        )
        cleaning_checkbox = colopts1[2].checkbox(
            "Enable Point Cloud Cleaning",
            value=True,
            help="If checked, removes points with zero depth from the point cloud.",
            key="cleaning_checkbox_generated2"
        )

        
        if scaling_checkbox:
            # Add a scale factor int number input
            scale_factor = st.number_input(
                "Scale Factor",
                min_value=1,
                max_value=1000,
                value=100,
                step=5,
                help="Scale factor to multiply depth values for visualization."
            )
        else:
            scale_factor = 1

        if threshold_checkbox:

            metric_default = False

            try:
                if st.session_state['last_model_title'] and "Metric" in st.session_state['last_model_title']:
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
                key="threshold_generated1"
            )
        else:
            threshold = 0
            metric_toggle = False

        if cleaning_checkbox:
            clean_point_cloud_bool = True
        else:
            clean_point_cloud_bool = False

        colsub1 = st.columns(3)

        if colsub1[0].button(
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
                scale_factor=scale_factor,
                clean_point_cloud_bool=clean_point_cloud_bool
                )
        
        
        # A button to save this point cloud to .npy
        if st.session_state.point_cloud_x is not None:
            if colsub1[1].button(
                "Download Generated Point Cloud (.npy)",
                key="download_generated_point_cloud",
                type="secondary",
                help="Click to download the generated point cloud as a .npy file."
            ):
                generated_pcd_array = np.vstack((
                    st.session_state.point_cloud_x,
                    st.session_state.point_cloud_y,
                    st.session_state.point_cloud_z
                )).T  # Shape (N, 3)

                npy_bytes = io.BytesIO()
                np.save(npy_bytes, generated_pcd_array)
                npy_bytes.seek(0)

                colsub1[2].download_button(
                    label="Download Generated Point Cloud",
                    data=npy_bytes,
                    file_name="generated_point_cloud.npy",
                    mime="application/octet-stream"
                )

    else:
        st.info("Please upload a generated point cloud (.npy) file to visualize.")


with tabs[1]:

    if ground_truth_file is not None:    
        st.subheader("Ground Truth 3D Point Cloud Visualization")

        st.write('Point cloud')

        colopts2 = st.columns(3)

        threshold_checkbox2 = colopts2[0].checkbox(
            "Enable Depth Thresholding",
            value=True,
            help="If checked, allows setting a depth threshold to filter points."
        )

        scaling_checkbox2 = colopts2[1].checkbox(
            "Enable Depth Scaling",
            value=True,
            help="If checked, allows setting a scale factor for depth values."
        )
        cleaning_checkbox2 = colopts2[2].checkbox(
            "Enable Point Cloud Cleaning",
            value=True,
            help="If checked, removes points with zero depth from the point cloud."
        )

        
        if scaling_checkbox2:
            # Add a scale factor int number input
            
            scale_factor2 = st.number_input(
                "Scale Factor",
                min_value=1,
                max_value=1000,
                value=100,
                step=5,
                help="Scale factor to multiply depth values for visualization.",
                key="scale_factor_ground_truth2"
            )
        else:
            scale_factor2 = 1

        if threshold_checkbox2:

            metric_default2 = False

            try:
                if st.session_state['last_model_title'] and "Metric" in st.session_state['last_model_title']:
                    metric_default2 = True
            except Exception:
                metric_default2 = False

            metric_toggle2 = st.checkbox(
            "Treat depth as metric",
            value=metric_default2,
            help="If checked, values beyond the threshold will be removed before visualization.",
            key="metric_toggle_ground_truth2"
        )

        # Compute a robust median (ignore non-finite and zero values where appropriate) to suggest a default threshold
            try:
                flat = np.asarray(ground_truth_pc).flatten()
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


            threshold2 = st.slider(
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
                key="threshold_ground_truth2"
            )
        else:
            threshold2 = 0
            metric_toggle2 = False

        if cleaning_checkbox2:
            clean_point_cloud_bool2 = True
        else:
            clean_point_cloud_bool2 = False


        colsub2 = st.columns(3)
        if colsub2[0].button(
            "Generate 3D Point Cloud",
            key="generate_ground_truth_point_cloud",
            type="primary",
            help="Click to generate and visualize the 3D point cloud from the depth map."
        ):
            
            # x, y, z = visualize_pcd_from_depth(ground_truth_pc, metric=metric_toggle2, threshold=threshold2)
            

            st.session_state.ground_truth_x, st.session_state.ground_truth_y, st.session_state.ground_truth_z = visualize_pcd_from_depth(
                ground_truth_pc,
                metric=metric_toggle2,
                threshold=threshold2,
                scale_factor=scale_factor2,
                clean_point_cloud_bool=clean_point_cloud_bool2
                )
        
        # A button to save this point cloud to .npy
        if st.session_state.ground_truth_x is not None:
            if colsub2[1].button(
                "Download Ground Truth Point Cloud (.npy)",
                key="download_ground_truth_point_cloud",
                type="secondary",
                help="Click to download the ground truth point cloud as a .npy file."
            ):
                ground_truth_pcd_array = np.vstack((
                    st.session_state.ground_truth_x,
                    st.session_state.ground_truth_y,
                    st.session_state.ground_truth_z
                )).T  # Shape (N, 3)

                npy_bytes_gt = io.BytesIO()
                np.save(npy_bytes_gt, ground_truth_pcd_array)
                npy_bytes_gt.seek(0)

                colsub2[2].download_button(
                    label="Download Ground Truth Point Cloud",
                    data=npy_bytes_gt,
                    file_name="ground_truth_point_cloud.npy",
                    mime="application/octet-stream"
                )
    
    else:
        st.info("Please upload a ground truth point cloud (.npy) file to visualize.")


with tabs[2]:

    st.subheader("Point Cloud Comparison")

    if st.session_state.point_cloud_x is not None:

        # Add Button to plot point cloud comparison

        cleaned_x = st.session_state.point_cloud_x
        cleaned_y = st.session_state.point_cloud_y
        cleaned_z = st.session_state.point_cloud_z

        gt_x = st.session_state.ground_truth_x
        gt_y = st.session_state.ground_truth_y
        gt_z = st.session_state.ground_truth_z
        
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

        
        if st.button('Normalize Generated and Ground Truth Point Clouds to Unit Cube', key='normalize_point_clouds'):

            st.session_state.point_cloud_x_final, st.session_state.point_cloud_y_final, st.session_state.point_cloud_z_final = normalize_to_unit_cube(cleaned_x, cleaned_y, cleaned_z)

            st.session_state.ground_truth_x_final, st.session_state.ground_truth_y_final, st.session_state.ground_truth_z_final = normalize_to_unit_cube(gt_x, gt_y, gt_z) if ground_truth_pc is not None else (None, None, None)
            
            st.info(f'Point Cloud Normalized min Z: {np.min(st.session_state.point_cloud_z_final):.4f}, max Z: {np.max(st.session_state.point_cloud_z_final):.4f}')
            st.info(f'Ground Truth Point Cloud Normalized min Z: {np.min(st.session_state.ground_truth_z_final):.4f}, max Z: {np.max(st.session_state.ground_truth_z_final):.4f}' )
            st.success("Point clouds normalized to unit cube.")

        else:
            
            cleaned_x_unit, cleaned_y_unit, cleaned_z_unit = cleaned_x, cleaned_y, cleaned_z
            gt_x_unit, gt_y_unit, gt_z_unit = gt_x, gt_y, gt_z if ground_truth_pc is not None else (None, None, None)

        
        

        #Add a Button to plot point clouds
        if st.button(
            "Plot Point Cloud Comparison",
            key="plot_point_cloud_comparison",
            type="primary",
            help="Click to plot and compare the generated and ground truth point clouds."
        ):

            all_z = np.concatenate([st.session_state.point_cloud_z_final, st.session_state.ground_truth_z_final]) if ground_truth_pc is not None else st.session_state.ground_truth_z_final
            
            z_min, z_max = np.min(all_z), np.max(all_z)
            cleaned_z_unit = (cleaned_z_unit - z_min) / (z_max - z_min) * 1  # Scale to [0, 500]
            
            if ground_truth_pc is not None:
                gt_z_unit = (gt_z_unit - z_min) / (z_max - z_min) * 1
        # Plot both point clouds together for comparison in streamlit
            fig = go.Figure()

            fig.add_trace(go.Scatter3d(
                x=st.session_state.point_cloud_x_final,
                y=st.session_state.point_cloud_y_final,
                z=st.session_state.point_cloud_z_final,
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
                    x=st.session_state.ground_truth_x_final,
                    y=st.session_state.ground_truth_y_final,
                    z=st.session_state.ground_truth_z_final,
                    mode='markers',
                    marker=dict(
                        size=2,
                        color='red',
                        opacity=0.8
                    ),
                    name='Ground Truth Point Cloud'
                ))
                
                st.plotly_chart(fig)

    else:
        st.info("Please generate the point cloud in the '3D Point Cloud Visualization' tab first.")

with tabs[3]:
    
    st.subheader("Partial Cloud Alignment")

    # Rotate point cloud randomly for demonstration

    random_rotation_matrix = np.array([
        [0.866, -0.5, 0],
        [0.5, 0.866, 0],
        [0, 0, 1]
    ])

    if st.button('random Rotation'):

        cleaned_x = st.session_state.point_cloud_x_final
        cleaned_y = st.session_state.point_cloud_y_final
        cleaned_z = st.session_state.point_cloud_z_final

        points = np.vstack((cleaned_x, cleaned_y, cleaned_z)).T  # Shape (N, 3)
        rotated_points = points.dot(random_rotation_matrix.T)
        rotated_point_cloud_x = rotated_points[:, 0]
        rotated_point_cloud_y = rotated_points[:, 1]
        rotated_point_cloud_z = rotated_points[:, 2] 

        # Visualize rotated point cloud alongside ground truth if available
        fig = go.Figure()
        fig.add_trace(go.Scatter3d(
            x=rotated_point_cloud_x,
            y=rotated_point_cloud_y,
            z=rotated_point_cloud_z,
            mode='markers',
            marker=dict(
                    size=2,
                    color='blue',
                    opacity=0.8
                ),
            name='Rotated Generated Point Cloud'))
        
        if st.session_state.ground_truth_x is not None:
            fig.add_trace(go.Scatter3d(
                x=st.session_state.ground_truth_x,
                y=st.session_state.ground_truth_y,
                z=st.session_state.ground_truth_z,
                mode='markers',
                marker=dict(
                    size=2,
                    color='red',
                    opacity=0.8
                ),
                name='Ground Truth Point Cloud'
            ))
        
        st.plotly_chart(fig)

    # RANSAC Global Registration UI
    st.markdown("---")
    st.subheader("Global Alignment (RANSAC + FPFH)")
    voxel_size = st.number_input("Voxel size for downsampling (RANSAC)", min_value=0.001, max_value=1.0, value=0.05, step=0.001, format="%.3f")

    if st.button('Run RANSAC Global Registration'):
        if st.session_state.point_cloud_x_final is None or st.session_state.ground_truth_x_final is None:
            st.error("Both generated and ground-truth normalized point clouds are required (use 'Normalize' in Comparison tab first).")
        else:
            with st.spinner('Running RANSAC global registration...'):
                src_x = st.session_state.point_cloud_x_final
                src_y = st.session_state.point_cloud_y_final
                src_z = st.session_state.point_cloud_z_final

                tgt_x = st.session_state.ground_truth_x_final
                tgt_y = st.session_state.ground_truth_y_final
                tgt_z = st.session_state.ground_truth_z_final

                aligned_pts, transformation = ransac_global_registration(src_x, src_y, src_z, tgt_x, tgt_y, tgt_z, voxel_size=voxel_size)

                # Split and store aligned points back to session state
                st.session_state.point_cloud_x_final = aligned_pts[:, 0]
                st.session_state.point_cloud_y_final = aligned_pts[:, 1]
                st.session_state.point_cloud_z_final = aligned_pts[:, 2]

                st.success("RANSAC global registration finished.")
                st.write("Transformation matrix:")
                st.write(transformation)


        
    # Alignment method selection and button: three choices
    alignment_method = st.radio(
        "Alignment method",
        ("Basic ICP", "Robust ICP (PCA init + outlier rejection)", "Robust ICP (Random init + outlier rejection)", "RANSAC Global (FPFH + RANSAC)"),
        index=1,
        help="Choose Basic ICP, Robust ICP variants, or RANSAC global registration.",
    )

    # Common ICP parameters
    icp_max_iters = st.number_input("Max ICP iterations", min_value=1, max_value=1000, value=100, step=1)
    icp_tolerance = st.number_input("ICP convergence tolerance", min_value=1e-9, max_value=1e-2, value=1e-6, format="%.1e")

    # Parameters for robust ICP (shown when robust selected)
    if alignment_method != "Basic ICP":
        outlier_rejection_ratio = st.slider(
            "Outlier rejection quantile",
            min_value=0.5,
            max_value=0.99,
            value=0.9,
            step=0.01,
            help="Keep only points with distance below this quantile during each ICP iteration.",
        )
    else:
        outlier_rejection_ratio = None

    # If Random init selected, allow specifying number of restarts
    if alignment_method.startswith("Robust ICP (Random"):
        num_restarts = st.number_input("Random init restarts (num_restarts)", min_value=1, max_value=200, value=10, step=1)
    else:
        num_restarts = 0

    # RANSAC-specific params
    if alignment_method.startswith("RANSAC"):
        ransac_voxel_size = st.number_input("RANSAC voxel size", min_value=0.001, max_value=1.0, value=0.05, step=0.001, format="%.3f")
        ransac_max_iter = st.number_input("RANSAC max iterations", min_value=1000, max_value=500000, value=100000, step=1000)
    else:
        ransac_voxel_size = None
        ransac_max_iter = None

    # Add button to perform ICP alignment
    if st.button('Perform ICP Alignment', key='icp_alignment'):
        
        random_rotation_matrix = np.array([
            [0.866, -0.5, 0],
            [0.5, 0.866, 0],
            [0, 0, 1]
        ])
        cleaned_x = st.session_state.point_cloud_x_final
        cleaned_y = st.session_state.point_cloud_y_final
        cleaned_z = st.session_state.point_cloud_z_final

        points = np.vstack((cleaned_x, cleaned_y, cleaned_z)).T  # Shape (N, 3)
        rotated_points = points.dot(random_rotation_matrix.T)

        rotated_point_cloud_x = rotated_points[:, 0]
        rotated_point_cloud_y = rotated_points[:, 1]
        rotated_point_cloud_z = rotated_points[:, 2] 

        with st.spinner('Performing ICP alignment...'):
            if alignment_method == "Basic ICP":
                aligned_x, aligned_y, aligned_z, R, t, final_error = icp_alignment(
                    rotated_point_cloud_x, rotated_point_cloud_y, rotated_point_cloud_z,
                    st.session_state.ground_truth_x_final, st.session_state.ground_truth_y_final, st.session_state.ground_truth_z_final,
                    max_iterations=int(icp_max_iters),
                    tolerance=float(icp_tolerance),
                )

            else:
                # Choose init method based on user's selection
                if alignment_method.startswith("Robust ICP (PCA"):
                    init_method = 'pca'
                else:
                    init_method = 'random'

                if alignment_method.startswith("RANSAC"):
                    # Run RANSAC global registration (Open3D)
                    aligned_pts, transformation = ransac_global_registration(
                        rotated_point_cloud_x, rotated_point_cloud_y, rotated_point_cloud_z,
                        st.session_state.ground_truth_x_final, st.session_state.ground_truth_y_final, st.session_state.ground_truth_z_final,
                        voxel_size=float(ransac_voxel_size)
                    )

                    aligned_x = aligned_pts[:, 0]
                    aligned_y = aligned_pts[:, 1]
                    aligned_z = aligned_pts[:, 2]
                    R = transformation[:3, :3]
                    t = transformation[:3, 3]

                    # Compute a simple final_error as mean squared distance to nearest GT neighbors
                    from sklearn.neighbors import KDTree as SKKDTree
                    gt_pts = np.vstack((st.session_state.ground_truth_x_final, st.session_state.ground_truth_y_final, st.session_state.ground_truth_z_final)).T
                    tree_gt_tmp = SKKDTree(gt_pts)
                    distances_tmp, _ = tree_gt_tmp.query(np.vstack((aligned_x, aligned_y, aligned_z)).T)
                    final_error = float(np.mean(distances_tmp**2))

                else:
                    aligned_x, aligned_y, aligned_z, R, t, final_error = icp_alignment_robust(
                        rotated_point_cloud_x, rotated_point_cloud_y, rotated_point_cloud_z,
                        st.session_state.ground_truth_x_final, st.session_state.ground_truth_y_final, st.session_state.ground_truth_z_final,
                        max_iterations=int(icp_max_iters),
                        tolerance=float(icp_tolerance),
                        outlier_rejection_ratio=float(outlier_rejection_ratio) if outlier_rejection_ratio is not None else 0.9,
                        init_method=init_method,
                        num_restarts=int(num_restarts) if num_restarts > 0 else 10,
                    )

            # aligned_x, aligned_y, aligned_z, R, t, final_error = icp_alignment(
            #     st.session_state.point_cloud_x_final, st.session_state.point_cloud_y_final, st.session_state.point_cloud_z_final,
            #     st.session_state.ground_truth_x_final, st.session_state.ground_truth_y_final, st.session_state.ground_truth_z_final
            # )

        st.success(f"ICP alignment completed with final error: {final_error:.6f}")
        
        # Perform RMSE calculation between aligned_x, aligned_y, aligned_z and ground truth point clouds
        from sklearn.neighbors import KDTree
        import numpy as np
        
        aligned_points = np.vstack((aligned_x, aligned_y, aligned_z)).T
        gt_points_x = np.vstack((st.session_state.ground_truth_x_final,
                               st.session_state.ground_truth_y_final,
                                 st.session_state.ground_truth_z_final)).T
        
        tree_gt = KDTree(gt_points_x)
        distances, indices = tree_gt.query(aligned_points)
        rmse = np.sqrt(np.mean(distances**2))
        st.info(f"RMSE between aligned generated point cloud and ground truth: {rmse:.6f}")

        # Similarly, compute Chamfer Distance
        tree_aligned = KDTree(aligned_points)
        distances_gt_to_aligned, _ = tree_aligned.query(gt_points_x)
        chamfer_distance = np.mean(distances**2) + np.mean(distances_gt_to_aligned**2)
        st.info(f"Chamfer Distance between aligned generated point cloud and ground truth: {chamfer_distance:.6f}")


        # Display transformation details
        st.write("**Rotation Matrix:**")
        st.write(R)
        st.write("**Translation Vector:**")
        st.write(t)
        
        # Visualize aligned point cloud
        fig_aligned = go.Figure()
        fig_aligned.add_trace(go.Scatter3d(
            x=aligned_x,
            y=aligned_y,
            z=aligned_z,
            mode='markers',
            marker=dict(size=2, color='green', opacity=0.8),
            name='Aligned Generated Point Cloud'
        ))
        fig_aligned.add_trace(go.Scatter3d(
            x=st.session_state.ground_truth_x_final,
            y=st.session_state.ground_truth_y_final,
            z=st.session_state.ground_truth_z_final,
            mode='markers',
            marker=dict(size=2, color='red', opacity=0.8),
            name='Ground Truth Point Cloud'
        ))
        
        

        st.plotly_chart(fig_aligned)