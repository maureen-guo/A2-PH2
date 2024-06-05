import numpy as np
import pandas as pd
import json
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import A2_Ph2_utilities as uti

def main():
    with open('./A2_Ph2/settings.json', 'r') as f:
        settings = json.load(f)

    H3_pixel_size_x = settings['H3_pixel_size_x']
    H3_pixel_size_y = settings['H3_pixel_size_y']
    W3_pixel_size_x = settings['W3_pixel_size_x']
    W3_pixel_size_y = settings['W3_pixel_size_y']

    H3_points = pd.read_excel('./A2_Ph2/CalibPointsH3image.xlsx')
    W3_points = pd.read_excel('./A2_Ph2/CalibPointsW3image.xlsx')
    H3_image = cv2.imread('./A2_Ph2/H3.JPG')
    W3_image = cv2.imread('./A2_Ph2/W3.png')

    h_h, w_h = H3_image.shape[:2] 
    cx_h = w_h / 2
    cy_h = h_h / 2

    h_w, w_w = W3_image.shape[:2] 
    cx_w = w_w / 2
    cy_w = h_w / 2
    
    initial_points_H3 = H3_points.copy()
    initial_points_W3 = W3_points.copy()
    # start iteration
    for iteration in range(6):
        print(f"Iteration {iteration + 1}")
        
        H3_points = initial_points_H3.copy()
        W3_points = initial_points_W3.copy()

        xd_H3 = H3_points['u\''].values
        yd_H3 = H3_points['v\''].values
        X_H = H3_points['X'].values
        Y_H = H3_points['Y'].values
        Z_H = H3_points['Z'].values
        xd_W3 = W3_points['u\''].values
        yd_W3 = W3_points['v\''].values
        X_W = W3_points['X'].values
        Y_W = W3_points['Y'].values
        Z_W = W3_points['Z'].values

        xu_h, yu_h = uti.calculate_undistorted_coordinates(xd_H3, yd_H3, cx_h, cy_h, H3_pixel_size_x, H3_pixel_size_y, sx=1)
        xu_w, yu_w = uti.calculate_undistorted_coordinates(xd_W3, yd_W3, cx_w, cy_w, W3_pixel_size_x, W3_pixel_size_y, sx=1)
        # Select the farthest reference point for H3
        idx_h = np.argmax(np.sqrt(xu_h**2 + yu_h**2))
        X_ref_h, Y_ref_h, Z_ref_h = X_H[idx_h], Y_H[idx_h], Z_H[idx_h]
        x_ref_h, y_ref_h = xu_h[idx_h], yu_h[idx_h]
        # Select the farthest reference point for W3
        idx_w = np.argmax(np.sqrt(xu_w**2 + yu_w**2))
        X_ref_w, Y_ref_w, Z_ref_w = X_W[idx_w], Y_W[idx_w], Z_W[idx_w]
        x_ref_w, y_ref_w = xu_w[idx_w], yu_w[idx_w]
        # calculte parameters for H3
        L_h = uti.calculate_L(X_H, Y_H, Z_H, xu_h, yu_h)
        ty_h = uti.calculate_ty(L_h)
        sx_h = uti.calculate_sx(L_h, ty_h)
        ty_sign_h = uti.determine_ty_sign(L_h, ty_h, X_ref_h, Y_ref_h, Z_ref_h, x_ref_h, y_ref_h)
        xu_new_h, yu_new_h = uti.calculate_undistorted_coordinates(xd_H3, yd_H3, cx_h, cy_h, H3_pixel_size_x, H3_pixel_size_y, sx_h)
        R_h, tx_h = uti.calculate_rotation_matrix_and_tx(L_h, sx_h, ty_sign_h)
        f_h, tz_h = uti.calculate_f_and_tz(R_h, X_H, Y_H, Z_H, yu_new_h, ty_sign_h)
        projected_u_h, projected_v_h, RT_h = uti.project_3D_to_2D(X_H, Y_H, Z_H, R_h, tx_h, ty_sign_h, tz_h, sx_h, f_h, cx_h, cy_h, H3_pixel_size_x, H3_pixel_size_y)
        uti.plot_results(xd_H3, yd_H3, projected_u_h, projected_v_h)
        # calculte parameters for W3    
        L_w = uti.calculate_L(X_W, Y_W, Z_W, xu_w, yu_w)
        ty_w = uti.calculate_ty(L_w)
        sx_w = uti.calculate_sx(L_w, ty_w)
        ty_sign_w = uti.determine_ty_sign(L_w, ty_w, X_ref_w, Y_ref_w, Z_ref_w, x_ref_w, y_ref_w)
        xu_new_w, yu_new_w = uti.calculate_undistorted_coordinates(xd_W3, yd_W3, cx_w, cy_w, W3_pixel_size_x, W3_pixel_size_y, sx_w)
        R_w, tx_w = uti.calculate_rotation_matrix_and_tx(L_w, sx_w, ty_sign_w)
        f_w, tz_w = uti.calculate_f_and_tz(R_w, X_W, Y_W, Z_W, yu_new_w, ty_sign_w)
        projected_u_w, projected_v_w, RT_w = uti.project_3D_to_2D(X_W, Y_W, Z_W, R_w, tx_w, ty_sign_w, tz_w, sx_w, f_w, cx_w, cy_w, W3_pixel_size_x, W3_pixel_size_y)
        uti.plot_results(xd_W3, yd_W3, projected_u_w, projected_v_w)
        # ray for H3
        Oc = np.array([0, 0, 0, 1])
        OcW_h = np.linalg.pinv(RT_h).dot(Oc)
        F_h = np.array([xu_new_h, yu_new_h, f_h, 1], dtype=object)
        MwI_h = np.linalg.pinv(RT_h).dot(F_h)
        ray_left_h = uti.calculate_plane_points(OcW_h, MwI_h, 0)[:70]
        ray_right_h = uti.calculate_plane_points(OcW_h, MwI_h, 1)[70:]
        projected_points_h = ray_left_h + ray_right_h
        original_points_h = list(zip(X_H, Y_H, Z_H))
        original_left_h = [i for i in original_points_h if i[1] == 0]
        original_right_h = [i for i in original_points_h if i[0] == 0]
        ray_left_h = np.array(ray_left_h)
        ray_right_h = np.array(ray_right_h)
        original_left_h = np.array(original_left_h)
        original_right_h = np.array(original_right_h)
        uti.plot_projection_results(original_left_h, ray_left_h, original_right_h, ray_right_h)

        rmse_value_h = uti.calculate_rmse(original_points_h, projected_points_h)
        print("RMSE for H3:", rmse_value_h)
        # ray for W3
        OcW_w = np.linalg.pinv(RT_w).dot(Oc)
        F_w = np.array([xu_new_w, yu_new_w, f_w, 1], dtype=object)
        MwI_w = np.linalg.pinv(RT_w).dot(F_w)
        ray_left_w = uti.calculate_plane_points(OcW_w, MwI_w, 0)[:70]
        ray_right_w = uti.calculate_plane_points(OcW_w, MwI_w, 1)[70:]
        projected_points_w = ray_left_w + ray_right_w
        original_points_w = list(zip(X_W, Y_W, Z_W))
        original_left_w = [i for i in original_points_w if i[1] == 0]
        original_right_w = [i for i in original_points_w if i[0] == 0]
        ray_left_w = np.array(ray_left_w)
        ray_right_w = np.array(ray_right_w)
        original_left_w = np.array(original_left_w)
        original_right_w = np.array(original_right_w)
        uti.plot_projection_results(original_left_w, ray_left_w, original_right_w, ray_right_w)

        rmse_value_w = uti.calculate_rmse(original_points_w, projected_points_w)
        print("RMSE for W3:", rmse_value_w)

        # find error
        errors_h = np.sqrt(np.sum((np.array(original_points_h) - np.array(projected_points_h))**2, axis=1))
        errors_w = np.sqrt(np.sum((np.array(original_points_w) - np.array(projected_points_w))**2, axis=1))
        # find max error
        max_error_index_h = np.argmax(errors_h)
        max_error_index_w = np.argmax(errors_w)
        # drop the max error
        drop_index_h = H3_points.index[max_error_index_h]
        drop_index_w = W3_points.index[max_error_index_w]

        initial_points_H3 = initial_points_H3.drop(index=drop_index_h)
        initial_points_W3 = initial_points_W3.drop(index=drop_index_w)

if __name__ == "__main__":
    main()