import numpy as np
import pandas as pd
import json
from PIL import Image
import matplotlib.pyplot as plt
import A2_Ph2_utilities as uti
import cv2

# Find the single tsai calibration for the H3 and W3 images and calculate the calibration error.
def main():
    # from setting.json get the basic information of W3 and H3
    with open('./A2_Ph2/settings.json', 'r') as f:
        settings = json.load(f)

    H3_pixel_size_x = settings['H3_pixel_size_x']
    H3_pixel_size_y = settings['H3_pixel_size_y']
    W3_pixel_size_x = settings['W3_pixel_size_x']
    W3_pixel_size_y = settings['W3_pixel_size_y']

    # get points from xlsx
    H3_points = pd.read_excel('./A2_Ph2/CalibPointsH3image.xlsx')
    W3_points = pd.read_excel('./A2_Ph2/CalibPointsW3image.xlsx')
    H3_image = cv2.imread('./A2_Ph2/H3.JPG')
    W3_image = cv2.imread('./A2_Ph2/W3.png')

    # calculate cx,cy
    h_h, w_h = H3_image.shape[:2] 
    cx_h = w_h / 2
    cy_h = h_h / 2

    h_w, w_w = W3_image.shape[:2] 
    cx_w = w_w / 2
    cy_w = h_w / 2
    
    xd_H3 = H3_points['u\''].values
    yd_H3 = H3_points['v\''].values
    X_H= H3_points['X'].values
    Y_H = H3_points['Y'].values
    Z_H = H3_points['Z'].values
    xd_W3 = W3_points['u\''].values
    yd_W3 = W3_points['v\''].values
    X_W = W3_points['X'].values
    Y_W = W3_points['Y'].values
    Z_W = W3_points['Z'].values

    # Compute the (xu, yu) corresponding to each (u, v)
    xu_h, yu_h = uti.calculate_undistorted_coordinates(xd_H3, yd_H3, cx_h, cy_h, H3_pixel_size_x, H3_pixel_size_y,sx=1)
    xu_w, yu_w = uti.calculate_undistorted_coordinates(xd_W3, yd_W3, cx_w, cy_w, W3_pixel_size_x, W3_pixel_size_y,sx=1)

    # Select the farthest reference point for H3
    idx_h = np.argmax(np.sqrt(xu_h**2 + yu_h**2))
    X_ref_h, Y_ref_h, Z_ref_h = X_H[idx_h], Y_H[idx_h], Z_H[idx_h]
    x_ref_h, y_ref_h = xu_h[idx_h], yu_h[idx_h]
    # Select the farthest reference point for W3
    idx_w = np.argmax(np.sqrt(xu_w**2 + yu_w**2))
    X_ref_w, Y_ref_w, Z_ref_w = X_W[idx_w], Y_W[idx_w], Z_W[idx_w]
    x_ref_w, y_ref_w = xu_w[idx_w], yu_w[idx_w]
    # Calculate for H3- L,ty,sx,ty_sign,R,tx,f,tz
    L_h = uti.calculate_L(X_H, Y_H, Z_H, xu_h, yu_h)
    ty_h = uti.calculate_ty(L_h)
    sx_h = uti.calculate_sx(L_h, ty_h)
    ty_sign_h= uti.determine_ty_sign(L_h, ty_h, X_ref_h, Y_ref_h, Z_ref_h, x_ref_h, y_ref_h)
    xu_new_h,yu_new_h = uti.calculate_undistorted_coordinates(xd_H3, yd_H3, cx_h, cy_h, H3_pixel_size_x,H3_pixel_size_y, sx_h)
    R_h, tx_h = uti.calculate_rotation_matrix_and_tx(L_h, sx_h, ty_sign_h)
    f_h, tz_h = uti.calculate_f_and_tz(R_h, X_H, Y_H, Z_H, yu_new_h, ty_sign_h)
    projected_u_h, projected_v_h, RT_h = uti.project_3D_to_2D(X_H, Y_H, Z_H, R_h, tx_h, ty_sign_h, tz_h, sx_h, f_h, cx_h, cy_h, H3_pixel_size_x,H3_pixel_size_y)
    uti.plot_results( xd_H3, yd_H3, projected_u_h, projected_v_h)
    # calculate OcW and MwI
    Oc = np.array([0, 0, 0, 1]) 
    OcW_h = np.linalg.pinv(RT_h).dot(Oc)
    F_h = np.array([xu_new_h, yu_new_h, f_h, 1], dtype=object)
    MwI_h = np.linalg.pinv(RT_h).dot(F_h)
    # drawing the ray
    ray_left_h = uti.calculate_plane_points(OcW_h, MwI_h,0)
    ray_left_h = ray_left_h[:70]
    ray_right_h = uti.calculate_plane_points(OcW_h, MwI_h,1)
    ray_right_h = ray_right_h[70:]
    projected_points_h = ray_left_h + ray_right_h
    original_points_h = list(zip(X_H, Y_H, Z_H))
    original_left_h = [i for i in original_points_h if i[1] == 0]
    original_right_h = [i for i in original_points_h if i[0] == 0]
    ray_left_h = np.array(ray_left_h)
    ray_right_h = np.array(ray_right_h)
    original_left_h = np.array(original_left_h)
    original_right_h = np.array(original_right_h)
    uti.plot_projection_results(original_left_h, ray_left_h, original_right_h, ray_right_h)
    
    #calculate rmse in image
    rmse_image_h = uti.calculate_rmse_image(xd_H3, yd_H3, projected_u_h, projected_v_h)
    print("RMSE in image for H3:", rmse_image_h)
    rmse_value_h = uti.calculate_rmse(original_points_h, projected_points_h)
    print("RMSE in cube for H3:", rmse_value_h)

    # Calculate for W3- L,ty,sx,ty_sign,R,tx,f,tz
    L_w = uti.calculate_L(X_W, Y_W, Z_W, xu_w, yu_w)
    ty_w = uti.calculate_ty(L_w)
    sx_w= uti.calculate_sx(L_w, ty_w)
    ty_sign_w= uti.determine_ty_sign(L_w, ty_w, X_ref_w, Y_ref_w, Z_ref_w, x_ref_w, y_ref_w)
    xu_new_w,yu_new_w = uti.calculate_undistorted_coordinates(xd_W3, yd_W3, cx_w, cy_w, W3_pixel_size_x, W3_pixel_size_y, sx_w)
    R_w, tx_w = uti.calculate_rotation_matrix_and_tx(L_w, sx_w, ty_sign_w)
    f_w, tz_w = uti.calculate_f_and_tz(R_w, X_W, Y_W, Z_W, yu_new_w, ty_sign_w)
    projected_u_w, projected_v_w, RT_w = uti.project_3D_to_2D(X_W, Y_W, Z_W, R_w, tx_w, ty_sign_w, tz_w, sx_w, f_w, cx_w, cy_w, W3_pixel_size_x, W3_pixel_size_y)
    uti.plot_results( xd_W3, yd_W3, projected_u_w, projected_v_w)
    # calculate OcW and MwI
    Oc = np.array([0, 0, 0, 1]) 
    OcW_w = np.linalg.pinv(RT_w).dot(Oc)
    F_w = np.array([xu_new_w, yu_new_w, f_w, 1], dtype=object)
    MwI_W = np.linalg.pinv(RT_w).dot(F_w)
    # drawing the ray
    ray_left_w = uti.calculate_plane_points(OcW_w, MwI_W,0)
    ray_left_w = ray_left_w[:70]
    ray_right_w = uti.calculate_plane_points(OcW_w, MwI_W,1)
    ray_right_w = ray_right_w[70:]
    projected_points_w = ray_left_w + ray_right_w
    original_points_w= list(zip(X_W, Y_W, Z_W))
    original_left_w = [i for i in original_points_w if i[1] == 0]
    original_right_w = [i for i in original_points_w if i[0] == 0]
    ray_left_w = np.array(ray_left_w)
    ray_right_w = np.array(ray_right_w)
    original_left_w = np.array(original_left_w)
    original_right_w = np.array(original_right_w)
    uti.plot_projection_results(original_left_w, ray_left_w, original_right_w, ray_right_w)

    #calculate rmse in image
    rmse_image_w = uti.calculate_rmse_image(xd_W3, yd_W3, projected_u_w, projected_v_w)
    print("RMSE in image for W3:", rmse_image_w)
    rmse_value_w = uti.calculate_rmse(original_points_w, projected_points_w)
    print("RMSE in cube for W3:", rmse_value_w)

if __name__ == "__main__":
    main()
