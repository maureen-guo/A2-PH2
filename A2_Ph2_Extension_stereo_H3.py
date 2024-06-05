import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import glob
import matplotlib.pyplot as plt
import cv2
from Get3d2dCoordinates import *
import json
import A2_Ph2_utilities as A2_uti

def main():

    with open('./A2_Ph2/settings.json', 'r') as f:
        settings = json.load(f)

    H3_pixel_size_x = settings['H3_pixel_size_x']
    H3_pixel_size_y = settings['H3_pixel_size_y']

    H3_points_left = pd.read_excel('./A2_Ph2/H3/Left/H3_Cube_Left.xlsx')
    H3_points_right = pd.read_excel('./A2_Ph2/H3/Right/H3_Cube_Right.xlsx')
    # start iteration
    for iteration in range(15):
        xu_H3_left = H3_points_left['u\''].values
        xu_H3_right = H3_points_right['u\''].values
        yu_H3_left = H3_points_left['v\''].values
        yu_H3_right = H3_points_right['v\''].values

        X_left = H3_points_left['X'].values
        Y_left = H3_points_left['Y'].values
        Z_left = H3_points_left['Z'].values
        X_right = H3_points_right['X'].values
        Y_right = H3_points_right['Y'].values
        Z_right = H3_points_right['Z'].values

        H3_image_left = cv2.imread('./A2_Ph2/H3/Left/H3_Cube_Left.png')
        H3_image_right = cv2.imread('./A2_Ph2/H3/Right/H3_Cube_Right.png')

        h1, w1 = H3_image_left.shape[:2] 
        cx = w1 / 2
        cy = h1 / 2

        h2, w2 = H3_image_right.shape[:2] 
        cx2 = w2 / 2
        cy2 = h2 / 2

        H3_pixel_size = (H3_pixel_size_x + H3_pixel_size_y) / 2
        dx = dy = H3_pixel_size
 
        xu_left, yu_left = A2_uti.calculate_undistorted_coordinates(xu_H3_left, yu_H3_left, cx, cy, H3_pixel_size_x, H3_pixel_size_y,sx=1)
        xu_right, yu_right = A2_uti.calculate_undistorted_coordinates(xu_H3_right, yu_H3_right, cx2, cy2, H3_pixel_size_x, H3_pixel_size_y,sx=1)

        # Select the farthest reference point for H3
        idx_h = np.argmax(np.sqrt(xu_left**2 + yu_left**2))
        X_ref_h, Y_ref_h, Z_ref_h = X_left[idx_h], Y_left[idx_h], Z_left[idx_h]
        x_ref_h, y_ref_h = xu_left[idx_h], yu_left[idx_h]

        #calculate L
        L_left = A2_uti.calculate_L(X_left, Y_left, Z_left, xu_left, yu_left)
        L_right = A2_uti.calculate_L(X_right, Y_right, Z_right, xu_right, yu_right)

        #left image
        #calculate the extrinsic matrix
        r_values, tx, ty, tz, f_left, xu_left, yu_left= A2_uti.calculate_matrices_and_focal_length(L_left, xu_H3_left, yu_H3_left, X_left, Y_left, Z_left, dx, dy, cx, cy, X_ref_h, Y_ref_h, Z_ref_h, x_ref_h, y_ref_h)
        Rt_left = A2_uti.create_matrices(r_values, tx, ty, tz)

        # Chi-Square coordinates of the origin of the camera's coordinate system
        Oc = np.array([0, 0, 0, 1])  
        OcW_left = np.linalg.pinv(Rt_left).dot(Oc)
        F = np.array([xu_left, yu_left, f_left, 1], dtype=object)
        MwI_left = np.linalg.pinv(Rt_left).dot(F)

        #left plane
        left_plane_points1 = A2_uti.calculate_plane_points(OcW_left, MwI_left, 0)
        left_projected1 = left_plane_points1[:70]

        #right plane
        right_plane_points1 = A2_uti.calculate_plane_points(OcW_left, MwI_left, 1)
        right_projected1 = right_plane_points1[70:]

        projected_points1 = left_projected1 + right_projected1

        # the points given from the file
        original_points1 = list(zip(X_left, Y_left, Z_left))
        original_left1 = [i for i in original_points1 if i[1] == 0]
        original_right1 = [i for i in original_points1 if i[0] == 0]

        left_projected1 = np.array(left_projected1)
        right_projected1 = np.array(right_projected1)
        original_left1 = np.array(original_left1)
        original_right1 = np.array(original_right1)

        A2_uti.plot_projection_results(original_left1, left_projected1, original_right1, right_projected1)

        #calculate the error in cube in left image of H3
        rmse_value = A2_uti.calculate_rmse(original_points1, projected_points1)
        print(f"Iteration {iteration + 1} - RMSE1:", rmse_value)

        errors = np.sqrt(np.sum((np.array(original_points1) - np.array(projected_points1))**2, axis=1))

        max_error_index = np.argmax(errors)

        # Drop the point with the largest error
        H3_points_left = H3_points_left.drop(H3_points_left.index[max_error_index])

        #right image
        # Select the farthest reference point for H3
        idx_h = np.argmax(np.sqrt(xu_right**2 + yu_right**2))
        X_ref_h_right, Y_ref_h_right, Z_ref_h_right = X_right[idx_h], Y_right[idx_h], Z_right[idx_h]
        x_ref_h_right, Y_ref_h_right = xu_right[idx_h], yu_right[idx_h]

        #calculate the extrinsic matrix, focal length, updated undistorted coordinates
        r_values, tx, ty, tz, f_right, xu_right, yu_right = A2_uti.calculate_matrices_and_focal_length(L_right, xu_H3_right, yu_H3_right, X_right, Y_right, Z_right, dx, dy, cx2, cy2, X_ref_h_right, Y_ref_h_right, Z_ref_h_right, x_ref_h_right, Y_ref_h_right)
        Rt_right = A2_uti.create_matrices( r_values, tx, ty, tz)

        # Chi-Square coordinates of the origin of the camera's coordinate system
        Oc = np.array([0, 0, 0, 1])  
        OcW_right = np.linalg.pinv(Rt_right).dot(Oc)
        F = np.array([xu_right, yu_right, f_right, 1], dtype=object)
        MwI_right = np.linalg.pinv(Rt_right).dot(F)

        #left plane
        left_plane_points2 = A2_uti.calculate_plane_points(OcW_right, MwI_right, 0)
        left_projected2 = left_plane_points2[:70]

        #right plane
        right_plane_points2 = A2_uti.calculate_plane_points(OcW_right, MwI_right, 1)
        right_projected2 = right_plane_points2[70:]

        #merge the projected points
        projected_points2 = left_projected2 + right_projected2

        # the points given from the file
        original_points2 = list(zip(X_left, Y_left, Z_left))
        original_left2 = [i for i in original_points2 if i[1] == 0]
        original_right2 = [i for i in original_points2 if i[0] == 0]

        left_projected2 = np.array(left_projected2)
        right_projected2 = np.array(right_projected2)
        original_left2 = np.array(original_left2)
        original_right2 = np.array(original_right2)

        #draw the cube
        A2_uti.plot_projection_results(original_left2, left_projected2, original_right2, right_projected2)

        #calculate the error in cube in the right image of H3
        rmse_value = A2_uti.calculate_rmse(original_points2, projected_points2)
        print(f"Iteration {iteration + 1} - RMSE2:", rmse_value)

        errors = np.sqrt(np.sum((np.array(original_points2) - np.array(projected_points2))**2, axis=1))

        max_error_index = np.argmax(errors)

        # Drop the point with the largest error
        H3_points_right = H3_points_right.drop(H3_points_right.index[max_error_index])

        OcW_left = OcW_left[:3]
        MwI_left = MwI_left[:3]
        OcW_right = OcW_right[:3]
        MwI_right = MwI_right[:3]
        MwI_left = list(zip(*MwI_left))
        MwI_right = list(zip(*MwI_right))

        # calculate the intersection of left image and right image
        intersections = A2_uti.compute_intersections(OcW_left, MwI_left, OcW_right, MwI_right)
        A2_uti.plot_projection_results(original_left2, np.array(intersections[:70]), original_right2, np.array(intersections[70:]))

        original_points = list(zip(X_left, Y_left, Z_left))
        rmse_value = A2_uti.calculate_rmse(original_points, intersections)
        print(f"Iteration {iteration + 1} - RMSE_intersection:", rmse_value)
    
if __name__ == "__main__":
    main()