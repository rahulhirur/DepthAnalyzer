from cam_data_generator.main import process_images_and_calibration

from depth_2_pcd.main import depth_to_ply_file

import os


output_dir = "/home/raptor/hirurrahul_prjs/output/DA_2_rel_small/Point_Cloud"



img_path1 = "/home/raptor/hirurrahul_prjs/output/DA_2_rel_small/Input/in_P_0_L_1.png"
calib_path = "/home/raptor/hirurrahul_prjs/output/DA_2_rel_small/Input/cam1.json"

depth_file = "/home/raptor/hirurrahul_prjs/output/DA_2_rel_small/Numpy_File/depth_data_Relative_(Small_-_25M).npy"

tag = os.path.basename(img_path1).split(".")[0]

scale_factor = 1.0
cam_idx1 = 0
cam_idx2 = 1




scaled_calibration_output_path = process_images_and_calibration(img_path1, calib_path, scale_factor, cam_idx1, cam_idx2, output_dir)


depth_to_ply_file(depth_file, scaled_calibration_output_path, output_dir, tag)