import cv2
import json
import numpy as np
import os
import yaml
import re
import sys

def read_image(file):
    if isinstance(file, str):
        return cv2.imread(file)
    elif hasattr(file, 'read'):
        return cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    else:
        return None

#Create a function to transpose the image size to (width, height)
def transpose_image_size(img_size):
    return (img_size[1], img_size[0])

def is_colab():
  return 'google.colab' in sys.modules

def show_image(image, title="Image"):
    """
    Displays an image using OpenCV. Automatically adapts for Google Colab
    or local environments using the is_colab() helper.

    Args:
        image (numpy.ndarray): The image to display (a NumPy array).
        title (str, optional): The title for the display window (used in local environments).
                               Defaults to "Image".
    """
    if is_colab():
        print(f"Displaying image in Colab output (title '{title}' ignored).")
        # For Colab, import cv2_imshow from google.colab.patches inside the function
        # to ensure it's only attempted when in Colab.
        from google.colab.patches import cv2_imshow
        cv2_imshow(image)
    else:
        print(f"Displaying image in a new window titled '{title}'.")
        print("Press any key on the keyboard to close the window.")
        # Standard OpenCV display for local environments
        cv2.imshow(title, image)
        cv2.waitKey(0) # Wait indefinitely for a key press
        cv2.destroyAllWindows() # Close all OpenCV windows

class YamlCameraCalibration:
    def __init__(self, yaml_file_path):

        self.data = cv2.FileStorage(yaml_file_path, cv2.FILE_STORAGE_READ)
        if not self.data.isOpened():
            raise IOError(f"Could not open YAML file: {yaml_file_path}")

    def load_camera_calibration(self, camera_index, scale_factor=1.0):
        camera_key = f"camera_matrix_{camera_index + 1}"
        dist_key = f"dist_coeff_{camera_index + 1}"

        # Load original intrinsic matrix
        K_original_data = self.data.getNode(camera_key).mat()
        if K_original_data is None:
            raise ValueError(f"Could not find camera matrix for {camera_key} in YAML.")
        K_original = K_original_data.reshape((3, 3))

        # Apply scaling to the intrinsic matrix
        K_scaled = K_original.copy()
        K_scaled[0, 0] *= scale_factor  # fx
        K_scaled[1, 1] *= scale_factor  # fy
        K_scaled[0, 2] *= scale_factor  # cx
        K_scaled[1, 2] *= scale_factor  # cy

        # Load distortion coefficients (remain unchanged by image scaling)
        D_data = self.data.getNode(dist_key).mat()
        if D_data is None:
            raise ValueError(f"Could not find distortion coefficients for {dist_key} in YAML.")
        D = D_data.flatten()

        return K_scaled, D

    def load_stereo_calibration(self):

        R_data = self.data.getNode('Rot_mat').mat()
        if R_data is None:
            raise ValueError("Could not find 'Rot_mat' in YAML.")
        R = np.array(R_data).reshape((3, 3))

        T_data = self.data.getNode('Trans_vect').mat()
        if T_data is None:
            raise ValueError("Could not find 'Trans_vect' in YAML.")
        T = np.array(T_data).reshape((3, 1))

        return R, T

    def get_all_calibration(self, cam1_idx=0, cam2_idx=1, scale_factor=1.0):

        K1, D1 = self.load_camera_calibration(cam1_idx, scale_factor)
        K2, D2 = self.load_camera_calibration(cam2_idx, scale_factor)
        R, T = self.load_stereo_calibration() # R and T are not scaled
        return K1, D1, K2, D2, R, T

    def close(self):
        self.data.release()

class JsonCameraCalibration:
    def __init__(self, json_file_path):
        try:
            if isinstance(json_file_path, str):
                
                with open(json_file_path, 'r') as f:
                    self.calibration_data = json.load(f)
            
            elif hasattr(json_file_path, 'read'):

                self.calibration_data = json.load(json_file_path)
        
        except FileNotFoundError:
            raise FileNotFoundError(f"JSON file not found: {json_file_path}")
        except json.JSONDecodeError:
            raise ValueError(f"Error decoding JSON from file: {json_file_path}")
        except Exception as e:
            raise Exception(f"An unexpected error occurred while loading JSON: {e}")

    def load_camera_calibration(self, camera_index, scale_factor=1.0):

        try:
            camera_data = self.calibration_data["calibration"]["cameras"][camera_index]
            parameters = camera_data["model"]["ptr_wrapper"]["data"]["parameters"]
        except KeyError as e:
            raise KeyError(f"Missing key in JSON structure for camera {camera_index}: {e}")

        # Original intrinsic parameters
        fx_original = parameters["f"]["val"]
        fy_original = fx_original / parameters["ar"]["val"] # Assuming ar is fx/fy
        cx_original = parameters["cx"]["val"]
        cy_original = parameters["cy"]["val"]

        # Apply scaling to intrinsic parameters
        fx_scaled = fx_original * scale_factor
        fy_scaled = fy_original * scale_factor
        cx_scaled = cx_original * scale_factor
        cy_scaled = cy_original * scale_factor

        K_scaled = np.array([[fx_scaled, 0, cx_scaled],
                             [0, fy_scaled, cy_scaled],
                             [0, 0, 1]], dtype=np.float64)

        # Distortion coefficients (remain unchanged by image scaling)
        k1 = parameters["k1"]["val"]
        k2 = parameters["k2"]["val"]
        k3 = parameters["k3"]["val"]
        p1 = parameters["p1"]["val"]
        p2 = parameters["p2"]["val"]
        D = np.array([k1, k2, p1, p2, k3], dtype=np.float64)

        return K_scaled, D

    def load_stereo_calibration(self, camera_index=1):

        try:
            # Assuming 'transform' describes the pose of camera_index relative to camera 0
            transform_data = self.calibration_data["calibration"]["cameras"][camera_index]["transform"]

            # Convert Rodrigues vector to Rotation Matrix
            rx = transform_data["rotation"]["rx"]
            ry = transform_data["rotation"]["ry"]
            rz = transform_data["rotation"]["rz"]
            R_vec = np.array([rx, ry, rz], dtype=np.float64)
            R, _ = cv2.Rodrigues(R_vec) # cv2.Rodrigues returns R and Jacobian, we only need R

            # Extract Translation Vector
            tx = transform_data["translation"]["x"]
            ty = transform_data["translation"]["y"]
            tz = transform_data["translation"]["z"]
            T = np.array([[tx], [ty], [tz]], dtype=np.float64)

        except KeyError as e:
            raise KeyError(f"Missing key in JSON structure for stereo transformation: {e}")
        except Exception as e:
            raise Exception(f"An unexpected error occurred while loading stereo parameters: {e}")

        return R, T

    def get_all_calibration(self, cam1_idx=0, cam2_idx=1, scale_factor=1.0):

        K1, D1 = self.load_camera_calibration(cam1_idx, scale_factor)
        K2, D2 = self.load_camera_calibration(cam2_idx, scale_factor)
        R, T = self.load_stereo_calibration(camera_index=cam2_idx)

        return K1, D1, K2, D2, R, T

    def get_num_cameras(self):
        """
        Returns the number of cameras in the calibration file.
        This assumes the JSON structure has a "calibration" key with a "cameras" list.
        """
        try:
            return len(self.calibration_data["calibration"]["cameras"])
        except KeyError:
            raise KeyError("Missing 'calibration' or 'cameras' key in JSON structure.")

    # get the reference camera index ( refrence camera is the camera whose rotation and translation vector is zero)
    def get_reference_camera_index(self):
        """
        Returns the index of the reference camera (the camera with zero rotation and translation).
        This assumes the JSON structure has a "calibration" key with a "cameras" list.
        """
        try:
            cameras = self.calibration_data["calibration"]["cameras"]
            for idx, camera in enumerate(cameras):
                transform = camera.get("transform", {})
                rotation = transform.get("rotation", {})
                translation = transform.get("translation", {})
                if (rotation.get("rx", 0) == 0 and
                    rotation.get("ry", 0) == 0 and
                    rotation.get("rz", 0) == 0 and
                    translation.get("x", 0) == 0 and
                    translation.get("y", 0) == 0 and
                    translation.get("z", 0) == 0):
                    return idx
            return None # No reference camera found
        except KeyError:
            raise KeyError("Missing 'calibration' or 'cameras' key in JSON structure.")

    def close(self):
        pass # No explicit resource to release for json.load

class CalibrationLoader:
    def __init__(self, file_path):
        if isinstance(file_path, str):
            
            ext = os.path.splitext(file_path)[1].lower()
            self.file_path = file_path

        elif hasattr(file_path, 'read'):
            
            ext = os.path.splitext(file_path.name)[1].lower()
            self.file_path = file_path.name

        if ext in ['.json']:
            self.loader = JsonCameraCalibration(file_path)

        elif ext in ['.yaml', '.yml']:
            self.loader = YamlCameraCalibration(file_path)
            
        else:
            raise ValueError(f"Unsupported file extension: {ext}")

    def get_all_calibration(self, cam1_idx, cam2_idx, scale_factor, dict_format=False):

        if dict_format:
            K1, D1, K2, D2, R, T = self.loader.get_all_calibration(cam1_idx, cam2_idx, scale_factor)
            return {
                "K1": K1,
                "D1": D1,
                "K2": K2,
                "D2": D2,
                "R": R,
                "T": T
            }
        else:
            return self.loader.get_all_calibration(cam1_idx, cam2_idx,scale_factor)

    def is_valid(self):
        """
        Checks if the calibration file is valid by attempting to load it.
        Returns True if successful, False otherwise.
        """
        try:
            self.loader.get_all_calibration()  # Try loading with scale factor 1.0
            return True
        except Exception as e:
            print(f"Error loading calibration data: {e}")
            return False
    
    #get the number of cameras in the calibration file
    def get_num_cameras(self):
        """
        Returns the number of cameras in the calibration file.
        This is a placeholder method and should be implemented in subclasses.
        """
        if hasattr(self.loader, 'get_num_cameras'):
            return self.loader.get_num_cameras()
        else:
            raise NotImplementedError("This method should be implemented in the loader class.")
    
    # get the reference camera index ( refrence camera is the camera whose rotation and translation vector is zero)
    def get_reference_camera_index(self):
        """
        Returns the index of the reference camera (the camera with zero rotation and translation).
        This is a placeholder method and should be implemented in subclasses.
        """
        if hasattr(self.loader, 'get_reference_camera_index'):
            return self.loader.get_reference_camera_index()
        else:
            raise NotImplementedError("This method should be implemented in the loader class.")

    def close(self):
        if hasattr(self.loader, 'close'):
            self.loader.close()
# Get Rectification map
def generate_rectify_data(K1, K2, R, T, d1, d2, size, flag=cv2.CALIB_ZERO_TANGENT_DIST, rectified_img_size=None):

    if rectified_img_size is None:
        rectified_img_size = size

    print(f"Generating rectification maps for images of size: {size}")
    R1, R2, P1, P2, Q, valid_roi1, valid_roi2 = cv2.stereoRectify(K1, d1, K2, d2, size, R, T, alpha=0, flags=flag, newImageSize= rectified_img_size)
    # shape of the rectification matrices
    print(f"R1 shape: {R1.shape}, R2 shape: {R2.shape}, P1 shape: {P1.shape}, P2 shape: {P2.shape}, Q shape: {Q.shape}, valid_roi1: {valid_roi1}, valid_roi2: {valid_roi2}")
    # print the rectification matrices
    print(f"Rectification matrices:\nR1: {R1}\nR2: {R2}\nP1: {P1}\nP2: {P2}\nQ: {Q}")

    # Get the area used for rectification in terms of percentage comapred to the original image size
    valid_roi1_area = (valid_roi1[2] -valid_roi1[0] )* (valid_roi1[3] -valid_roi1[1])
    valid_roi2_area = (valid_roi2[2] -valid_roi2[0] )* (valid_roi2[3] -valid_roi2[1])
    total_area = size[0] * size[1]

    valid_roi1_percentage = (valid_roi1_area / total_area) * 100
    valid_roi2_percentage = (valid_roi2_area / total_area) * 100

    #Get the prinicipal point of the rectified images
    principal_point1 = (P1[0, 2], P1[1, 2])
    principal_point2 = (P2[0, 2], P2[1, 2])


    #create a dictinaory to hold this evaluation data
    evaluation_data = {
        "roi1_percentage": valid_roi1_percentage,
        "roi2_percentage": valid_roi2_percentage,
        "principal_point1": principal_point1,
        "principal_point2": principal_point2
    }

    map1x, map1y = cv2.initUndistortRectifyMap(K1, d1, R1, P1, size, cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(K2, d2, R2, P2, size, cv2.CV_32FC1)

    return map1x, map1y, map2x, map2y, P1, P2, Q, evaluation_data

def rectify(img, map_x, map_y):
    print(f"Rectifying image with shape: {img.shape}")
    return cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)

def show_stereo_images(img_1, img_2):
    # Create an image for the line with the same height as the resized images
    line_img = np.full((img_1.shape[0], 2, 3), (0, 255, 255), dtype=np.uint8)
    show_image(np.concatenate([cv2.cvtColor(img_1, cv2.COLOR_GRAY2BGR),line_img, cv2.cvtColor(img_2, cv2.COLOR_GRAY2BGR)], axis=1))

def resize_image(image, scale_factor):
    """
    Resizes an image based on a given scale factor.

    Args:
        image (np.ndarray): The input image (OpenCV format).
        scale_factor (float): The factor by which to scale the image.

    Returns:
        np.ndarray: The resized image.
        tuple: The new dimensions of the resized image (width, height).
    """
    if scale_factor <= 0:
        raise ValueError("Scale factor must be positive.")

    # Get original dimensions
    original_height, original_width = image.shape[:2]
    print(image.shape[:2])
    
    # Calculate new dimensions
    new_width = int(original_width * scale_factor)
    new_height = int(original_height * scale_factor)
    
    # Resize the image
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    return resized_image, (new_height, new_width)

def save_images(rectified_img1, rectified_img2, output_base_dir="/output"):
    """
    Saves two rectified images to a specified output directory.

    Args:
        rectified_img1 (numpy.ndarray): The first rectified image (e.g., left camera).
        rectified_img2 (numpy.ndarray): The second rectified image (e.g., right camera).
        output_base_dir (str, optional): The base directory where images will be saved.
                                        Defaults to "/content/rectified_images"
                                        (common for Colab).
    Returns:
        tuple: A tuple containing the paths of the two saved images (path1, path2).
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_base_dir, exist_ok=True)
    print(f"Ensured output directory exists: {output_base_dir}")
    
    # Define the filenames for the rectified images
    rectified_img1_path = os.path.join(output_base_dir, "rectified_img1.png")
    rectified_img2_path = os.path.join(output_base_dir, "rectified_img2.png")

    # Save the rectified images
    cv2.imwrite(rectified_img1_path, rectified_img1)
    cv2.imwrite(rectified_img2_path, rectified_img2)

    print(f"Rectified images saved:")
    print(f"  - Left: {rectified_img1_path}")
    print(f"  - Right: {rectified_img2_path}")

    return rectified_img1_path, rectified_img2_path

def save_scaled_calibration_parameters(K1_scaled, D1_scaled, K2_scaled, D2_scaled,
                                       R_scaled, T_scaled, img_size, new_scale_factor,
                                       output_dir="/output"):
    """
    Saves scaled camera calibration parameters to a JSON file.

    Args:
        K1_scaled (numpy.ndarray): Scaled camera matrix for camera 1.
        D1_scaled (numpy.ndarray): Scaled distortion coefficients for camera 1.
        K2_scaled (numpy.ndarray): Scaled camera matrix for camera 2.
        D2_scaled (numpy.ndarray): Scaled distortion coefficients for camera 2.
        R_scaled (numpy.ndarray): Scaled rotation matrix (between cameras).
        T_scaled (numpy.ndarray): Scaled translation vector (between cameras).
        img_size (tuple): The resized image dimensions (width, height) used for scaling.
        new_scale_factor (float): The scale factor applied to the original parameters.
        output_dir (str, optional): The directory to save the JSON file. Defaults to "/content".

    Returns:
        str or None: The path to the saved JSON file if successful, None otherwise.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    print(f"Ensured output directory exists: {output_dir}")

    # Convert NumPy arrays to lists for JSON serialization
    K1_scaled_list = K1_scaled.tolist()
    D1_scaled_list = D1_scaled.tolist()
    K2_scaled_list = K2_scaled.tolist()
    D2_scaled_list = D2_scaled.tolist()
    R_scaled_list = R_scaled.tolist()
    T_scaled_list = T_scaled.tolist()

    # Create a dictionary to hold the scaled calibration parameters
    scaled_calibration_data = {
        "camera_matrix_1": K1_scaled_list,
        "dist_coeff_1": D1_scaled_list,
        "camera_matrix_2": K2_scaled_list,
        "dist_coeff_2": D2_scaled_list,
        "Rot_mat": R_scaled_list,
        "Trans_vect": T_scaled_list,
        "image_size_resized": img_size,
        "image_size_actual": (np.array(img_size)*(1/new_scale_factor)).tolist(),
        "scale_factor_applied": new_scale_factor,
        "baseline_distance": np.linalg.norm(T_scaled)  # Distance between cameras

    }

    # Define the output JSON file path
    scaled_calibration_output_path = os.path.join(output_dir, "scaled_calibration_parameters.json")

    # Save the scaled calibration parameters to a JSON file
    try:
        with open(scaled_calibration_output_path, 'w') as f:
            json.dump(scaled_calibration_data, f, indent=4) # indent=4 makes the JSON readable
        print(f"Scaled calibration parameters saved to: {scaled_calibration_output_path}")
        return scaled_calibration_output_path
    except Exception as e:
        print(f"Error saving scaled calibration parameters to JSON: {e}")
        return None

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

        if isinstance(json_file_path, str):
            
            with open(json_file_path, 'r') as f:
                loaded_data = json.load(f)
        
        elif hasattr(json_file_path, 'read'):
            loaded_data = json.load(json_file_path)

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

        loaded_params = {
            "K1": K1_scaled,
            "D1": D1_scaled,
            "K2": K2_scaled,
            "D2": D2_scaled,
            "R": R_scaled,
            "T": T_scaled,
            "image_size_resized": image_size_resized,
            "scale_factor_applied": scale_factor_applied
        }
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

def create_calibration_data_report(K1_scaled, D1_scaled, K2_scaled, D2_scaled, R_scaled, T_scaled, img_size, new_scale_factor, output_dir="output", html_path="assets/camera_calibration_template.html"):
    """
    Updates the calibrationData object in an HTML file with new scaled camera parameters.

    Parameters:
        html_path (str): Path to the input HTML file.
        output_path (str): Path to save the updated HTML file.
        K1_scaled, K2_scaled (list of list): 3x3 camera matrix.
        D1_scaled, D2_scaled (list): Distortion coefficients.
        R_scaled (list of list): 3x3 rotation matrix.
        T_scaled (list of list): 3x1 translation vector.
        img_size (list): New image size [width, height].
        new_scale_factor (float): New scale factor.
    """
    # Print current working directory for debugging

    print(f"Current working directory: {os.getcwd()}")
    
    #check if file exists or not
    if not os.path.exists(html_path):
        raise FileNotFoundError(f"HTML template file not found: {html_path}")
    

    with open(html_path, "r", encoding="utf-8") as f:
        html = f.read()

    # Extract calibrationData JS object using regex
    pattern = r"(const calibrationData\s*=\s*)(\{.*?\})(\s*;)"
    match = re.search(pattern, html, re.DOTALL)
    if not match:
        raise ValueError("calibrationData block not found in the HTML.")

    prefix, js_object_str, suffix = match.groups()

    # Convert JS object to JSON-compatible string
    json_compatible = re.sub(r'(\w+):', r'"\1":', js_object_str)
    json_compatible = json_compatible.replace(";", "")
    data = json.loads(json_compatible)

    # Convert NumPy arrays to lists for JSON serialization
    K1_scaled_list = K1_scaled.tolist()
    D1_scaled_list = D1_scaled.tolist()
    K2_scaled_list = K2_scaled.tolist()
    D2_scaled_list = D2_scaled.tolist()
    R_scaled_list = R_scaled.tolist()
    T_scaled_list = T_scaled.tolist()

    # Apply updates
    data["camera_matrix_1"] = K1_scaled_list
    data["dist_coeff_1"] = D1_scaled_list
    data["camera_matrix_2"] = K2_scaled_list
    data["dist_coeff_2"] = D2_scaled_list
    data["Rot_mat"] = R_scaled_list
    data["Trans_vect"] = T_scaled_list
    data["image_size_resized"] = img_size
    data["scale_factor_applied"] = new_scale_factor
    data["image_size_original"] = (np.array(img_size) * (1 / new_scale_factor)).tolist()  # Original size
    data["baseline_distance"] = np.linalg.norm(T_scaled_list)  # Distance between cameras L2 Distance

    # Convert back to JS-style object string
    new_js = json.dumps(data, indent=4)
    new_js = re.sub(r'"(\w+)":', r'\1:', new_js)

    updated_html = html[:match.start()] + prefix + new_js + suffix + html[match.end():]

    with open(os.path.join(output_dir, "Stereo_Calibration_Parameters.html"), "w", encoding="utf-8") as f:
        f.write(updated_html)

    print(f"calibrationData updated and saved to '{output_dir}'")

def suggest_next_folder_name(base_path, prefix="batch_"):
    """
    Suggests the next folder name based on existing folders in the directory.

    Args:
        base_path (str): Full path to the directory containing folders.
        prefix (str): Prefix for the folder names. Defaults to "batch_".

    Returns:
        str: Full path of the suggested next folder name.
    """
    existing_folders = [name for name in os.listdir(base_path) if name.startswith(prefix)]
    existing_numbers = [int(name[len(prefix):]) for name in existing_folders if name[len(prefix):].isdigit()]

    next_number = max(existing_numbers, default=0) + 1
    return os.path.join(base_path, f"{prefix}{next_number}")

def save_rectification_artifacts(P1, P2, output_dir):
    """
    Save rectification artifacts (P1 and P2 matrices) to a JSON file.

    Args:
        P1 (np.ndarray): Projection matrix for Camera 1.
        P2 (np.ndarray): Projection matrix for Camera 2.
        output_dir (str): Directory to save the JSON file.

    Returns:
        None
    """

    artifacts = {
        "P1": P1.tolist(),
        "P2": P2.tolist()
    }

    os.makedirs(output_dir, exist_ok=True)
    artifact_path = os.path.join(output_dir, "rectification_artifacts.json")

    with open(artifact_path, "w") as f:
        json.dump(artifacts, f, indent=4)






def process_images_and_calibration(img_path1, calib_path, scale_factor, cam_idx1, cam_idx2, output_dir):

    image_data = read_image(img_path1)
    resized_image, new_size = resize_image(image_data, scale_factor)
    print(f"Resized image to: {new_size}")

    # Save resized image
    resized_image_path = os.path.join(output_dir, "resized_image.png")
    cv2.imwrite(resized_image_path, resized_image)
    print(f"Resized image saved to: {resized_image_path}")

    #resize calibration parameters
    calibration_loader = CalibrationLoader(calib_path)

    K1_scaled, D1_scaled, K2_scaled, D2_scaled, R_scaled, T_scaled = calibration_loader.get_all_calibration(cam_idx1, cam_idx2, scale_factor)

    #Save scaled calibration parameters

    return save_scaled_calibration_parameters(K1_scaled, D1_scaled, K2_scaled, D2_scaled,
                                        R_scaled, T_scaled, new_size, scale_factor,
                                        output_dir=output_dir)