import numpy as np
import cv2
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def NearestNeighbourInterpolation(img,kappa):
    def undistPixel(xd,yd,kappa):
        cx=img.shape[1]//2
        cy=img.shape[0]//2
        xu = (xd-cx)*(1+kappa*((xd-cx)**2+(yd-cy)**2))+cx
        yu = (yd-cy)*(1+kappa*((xd-cx)**2+(yd-cy)**2))+cy
        return xu,yu

    #Find known points
    known = dict()
    out = -1*np.ones(img.shape)
    for y in range(0, img.shape[0]*2):
        for x in range(0, img.shape[1]*2):
            x_real = x//2
            y_real = y//2
            xu,yu = undistPixel(x/2,y/2,kappa)
            xu_int = int(round(xu))
            yu_int = int(round(yu))

            #If value is within the image bounds
            if 0<=int(round(xu))<img.shape[1] and 0<=int(round(yu))<img.shape[0]:
                #Find closest matching point
                if (xu_int, yu_int) not in known:
                    known[(xu_int, yu_int)] = (xu_int-xu)**2+(yu_int-yu)**2
                    out[yu_int,xu_int] = img[y_real,x_real]
                else:
                    #If a new point is found closer to the original
                    if known[(xu_int, yu_int)] > (xu_int-xu)**2+(yu_int-yu)**2:
                        known[(xu_int, yu_int)] = (xu_int-xu)**2+(yu_int-yu)**2
                        out[yu_int,xu_int] = img[y_real,x_real]

    out = out.astype(np.uint8)
    return out


def show (img, scale_percent = 30, waitKey=-1):
    w = int(img.shape[1] * scale_percent / 100)
    h = int(img.shape[0] * scale_percent / 100)
    img = cv2.resize(img, (w, h), interpolation = cv2.INTER_AREA)
    cv2.imshow('image', img)
    k = cv2.waitKey(waitKey) & 0xFF
    if k == ord('s'):
        cv2.imwrite("image.png", img)
        cv2.destroyAllWindows()     
    if k == ord('q'):
        cv2.destroyAllWindows()  
    cv2.destroyAllWindows()


def group_corners(data, max_rows=None):
    if max_rows is None:
        max_rows = data.shape[0]
    grouped_corners = [] 
    for i in range(20): 
        start = i % 10 
        step_start = (i // 10) * 70 
        indices = list(range(start + step_start, max_rows, 10))
        group = data.loc[indices[:7], ['u', 'v']].apply(tuple, axis=1).tolist()
        grouped_corners.append(group)
    return grouped_corners

def compute_new_coordinates(corners, k1, cx, cy):
    new_corners = []
    for corner in corners:
        xd, yd = corner[0], corner[1]
        xu = (xd - cx) * (1 + k1 * ((xd - cx) ** 2 + (yd - cy) ** 2)) + cx
        yu = (yd - cy) * (1 + k1 * ((xd - cx) ** 2 + (yd - cy) ** 2)) + cy
        new_corners.append([xu, yu])
    return np.array(new_corners)

def ransac_line_fitting(grouped_corners, cx, cy, max_iterations=10000, threshold=1e-11, step_size=1e-11):
    k1 = 0.0
    best_k1 = 0.0
    min_avg_error = float('inf')

    for _ in range(max_iterations):
        avg_error = 0

        for corners in grouped_corners:
            # Compute new coordinates for the current k1
            new_corners = compute_new_coordinates(corners, k1, cx, cy)
            
            # Apply RANSAC to find the best line fitting
            best_inliers = 0
            best_slope = None
            best_intercept = None
            
            for i in range(len(new_corners)):
                for j in range(i + 1, len(new_corners)):
                    x1, y1 = new_corners[i]
                    x2, y2 = new_corners[j]
                    if x2 == x1:  # Avoid division by zero
                        continue
                    slope = (y2 - y1) / (x2 - x1)
                    intercept = y1 - slope * x1
                    num_inliers = 0
                    
                    for x, y in new_corners:
                        if abs(y - (slope * x + intercept)) < threshold:
                            num_inliers += 1
                    
                    if num_inliers > best_inliers:
                        best_inliers = num_inliers
                        best_slope = slope
                        best_intercept = intercept
            
            # Ensure best_slope and best_intercept are valid before calculating error
            if best_slope is not None and best_intercept is not None:
                line_error = sum((y - (best_slope * x + best_intercept))**2 for x, y in new_corners)
                avg_error += line_error / len(new_corners)
            else:
                continue  # Skip this set of corners if no valid line was found

        avg_error /= len(grouped_corners)  # Compute average error across all lines
        
        # Update the best_k1 if the new average error is lower
        if avg_error < min_avg_error:
            min_avg_error = avg_error
            best_k1 = k1
        
        # Stop if the error is below a certain threshold
        if avg_error < threshold:
            break
        
        # Update k1 for the next iteration
        k1 += step_size

    return best_k1

def calculate_undistorted_coordinates(u_coords, v_coords, cx, cy, pixel_size_x, pixel_size_y, sx):
    xu = sx * pixel_size_x * (u_coords - cx)
    yu = pixel_size_y * (cy - v_coords)
    return xu, yu

def calculate_L(X, Y, Z, xu, yu):
    n = len(X)
    A = np.zeros((n, 7))
    B = np.zeros(n)

    for i in range(n):
        A[i] = [yu[i] * X[i], yu[i] * Y[i], yu[i] * Z[i], yu[i], -xu[i] * X[i], -xu[i] * Y[i], -xu[i] * Z[i]]
        B[i] = xu[i]
    # calculate L
    L = np.linalg.pinv(A).dot(B)
    return L

def calculate_ty(L):
    a5 = L[4]
    a6 = L[5]
    a7 = L[6]
    ty = 1 / np.sqrt(a5**2 + a6**2 + a7**2)
    return ty

def calculate_sx(L, ty):
    a1 = L[0]
    a2 = L[1]
    a3 = L[2]
    sx = ty * np.sqrt(a1**2 + a2**2 + a3**2)
    return sx

def determine_ty_sign(L, ty, X, Y, Z, x, y):
    a1, a2, a3, a4, a5, a6, a7 = L

    r11 = a1 * ty
    r12 = a2 * ty
    r13 = a3 * ty
    r21 = a5 * ty
    r22 = a6 * ty
    r23 = a7 * ty
    tx = a4 * ty

    x_new = r11 * X + r12 * Y + r13 * Z + tx
    y_new = r21 * X + r22 * Y + r23 * Z + ty

    if (np.sign(x_new) != np.sign(x)).any() or (np.sign(y_new) != np.sign(y)).any():
        ty = -ty

    return ty

def calculate_rotation_matrix_and_tx(L, sx, ty):
    a1, a2, a3, a4, a5, a6, a7 = L

    r11 = a1 * (ty / sx)
    r12 = a2 * (ty / sx)
    r13 = a3 * (ty / sx)
    r21 = a5 * ty
    r22 = a6 * ty
    r23 = a7 * ty
    tx = a4 * (ty / sx)

    r31 = r12 * r23 - r13 * r22
    r32 = r13 * r21 - r11 * r23
    r33 = r11 * r22 - r12 * r21

    r1 = np.array([r11, r12, r13])
    r2 = np.array([r21, r22, r23])
    r3 = np.array([r31, r32, r33])
    R = np.vstack((r1, r2, r3))

    return R, tx

def calculate_f_and_tz(R, X, Y, Z, yu, ty):
    n = len(X)
    A = np.zeros((n, 2))
    B = np.zeros(n)

    r21, r22, r23 = R[1]
    r31, r32, r33 = R[2]

    for i in range(n):
        A[i] = [r21 * X[i] + r22 * Y[i] + r23 * Z[i] + ty, -yu[i]]
        B[i] = yu[i]* r31 * X[i] + yu[i]*r32 * Y[i] + yu[i]*r33 * Z[i]

    params = np.linalg.pinv(A).dot(B)
    f = params[0]
    tz = params[1]

    return f, tz


def project_3D_to_2D(X, Y, Z, R, tx, ty, tz, sx, f, cx, cy, pixel_size_x, pixel_size_y):
    n = len(X)
    X_homogeneous = np.vstack((X, Y, Z, np.ones(n)))
    extrinsic_matrix = np.vstack((np.hstack((R, np.array([[tx], [ty], [tz]]))), np.array([0, 0, 0, 1])))
    # Define the intrinsic matrix
    intrinsic_matrix = np.array([[sx / pixel_size_x, 0, cx],
                                 [0, -1 / pixel_size_y, cy],
                                 [0, 0, 1]])
    # Define the projection matrix
    projection_matrix = np.array([[f, 0, 0, 0], 
                                  [0, f, 0, 0], 
                                  [0, 0, 1, 0]])
    uvw = intrinsic_matrix @ projection_matrix @ extrinsic_matrix @ X_homogeneous
    u = uvw[0] /uvw[2] 
    v = uvw[1] /uvw[2] 
    return u, v, extrinsic_matrix

def plot_results(original_u, original_v, projected_u, projected_v):
    plt.scatter(original_u, original_v, color='red', label='original pixels')
    plt.scatter(projected_u, projected_v, color='blue', label='projected pixels')
    plt.legend()
    plt.show()

# Drawing actual and projected pixel points
def plot_calibration_results(image, xu, yu, projected_points):
    plt.figure(figsize=(10, 10))
    plt.imshow(image, cmap='gray')
    
    plt.scatter(xu, yu, color='blue', s=10, label='real_pixels')
    plt.scatter(projected_points[0, :], projected_points[1, :], color='red', s=10, label='projected_pixels')
    
    plt.xlabel('x (px)')
    plt.ylabel('y (px)')
    plt.legend()
    plt.title('Calibration results')
    plt.show()

def calculate_rmse(original, projected):
    n = len(original)

    original = np.array(original)
    projected = np.array(projected)

    sum_of_squares = 0

    for i in range(n):
        diff_squared_x = (original[i][0] - projected[i][0]) ** 2
        diff_squared_y = (original[i][1] - projected[i][1]) ** 2
        diff_squared_z = (original[i][2] - projected[i][2]) ** 2

        sum_of_squares += diff_squared_x + diff_squared_y + diff_squared_z

    mean_of_squares = sum_of_squares / n
    rmse = np.sqrt(mean_of_squares)
    
    return rmse

def calculate_rmse_image(original_x, original_y, projected_x, projected_y):
    n = len(original_x)

    # original = np.array(original)
    # projected = np.array(projected)

    sum_of_squares = 0

    for i in range(n):
        diff_squared_x = (original_x[i] - projected_x[i]) ** 2
        diff_squared_y = (original_y[i] - projected_y[i]) ** 2
        #diff_squared_z = (original[i][2] - projected[i][2]) ** 2

        sum_of_squares += diff_squared_x + diff_squared_y

    mean_of_squares = sum_of_squares / n
    rmse = np.sqrt(mean_of_squares)
    
    return rmse

def ray(t, OcW, MwI):
    ray = [[0] * len(t) for _ in range(len(OcW))]
    for i in range(len(t)):
        for j in range (len(OcW)):
                ray [j][i] = OcW[j] + t[i] * (MwI[j][i] - OcW[j])
    return ray

def calculate_plane_points(OcW, MwI, index):
    t = [-OcW[index] / (m - OcW[index]) for m in MwI[index]]  # Use index to determine whether to process the left or right plane

    ray_pre = ray(t, OcW, MwI)
    points = []
    for y in range(len(MwI[index])):
        tuple_point = tuple(ray_pre[i][y] for i in range(len(OcW)))
        points.append(tuple_point)
    points = [(a, b, c) for a, b, c, d in points] 

    return points

def plot_projection_results(original_left, left_projected, original_right, right_projected):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # left
    ax.scatter(original_left[:, 0], original_left[:, 1], original_left[:, 2], c='blue', marker='o', label='Original Left')
    ax.scatter(left_projected[:, 0], left_projected[:, 1], left_projected[:, 2], c='red', marker='^', label='Back-projected Left')
    # right
    ax.scatter(original_right[:, 0], original_right[:, 1], original_right[:, 2], c='green', marker='o', label='Original Right')
    ax.scatter(right_projected[:, 0], right_projected[:, 1], right_projected[:, 2], c='purple', marker='^', label='Back-projected Right')

    ax.legend()
    ax.set_xlabel('X Coordinates')
    ax.set_ylabel('Y Coordinates')
    ax.set_zlabel('Z Coordinates')
    ax.set_title('Backprojection Results on Two Planes')

    plt.show()


#integrate the previous steps to get focal length, updated undistorted coordinates
def calculate_matrices_and_focal_length(L, xu_H3, yu_H3, X, Y, Z, dx, dy, cx, cy, X_ref, Y_ref, Z_ref, x_ref, y_ref):

    ty = calculate_ty(L)
    #calculte sx
    sx = calculate_sx(L, ty)
    xu, yu = calculate_undistorted_coordinates(xu_H3, yu_H3, cx, cy, dx, dy, sx)

    ty = determine_ty_sign(L, ty, X_ref, Y_ref, Z_ref, x_ref, y_ref)
    
    R, tx = calculate_rotation_matrix_and_tx(L, sx, ty)
    f, tz = calculate_f_and_tz(R, X, Y, Z, yu, ty)
    r_values = tuple(R.flatten())

    return  r_values, tx, ty, tz, f, xu, yu


def create_matrices( r_values, tx, ty, tz):
    r11, r12, r13, r21, r22, r23, r31, r32, r33 = r_values

    # calculte extrinsic matrix Rt
    Rt = np.array([
        [r11, r12, r13, tx],
        [r21, r22, r23, ty],
        [r31, r32, r33, tz],
        [0, 0, 0, 1]
    ])

    return Rt

def calculate_plane_points(OcW, MwI, index):
    #use index to decide to handle the right plane or the left plane
    t = [-OcW[index] / (m - OcW[index]) for m in MwI[index]]  

    ray_pre = ray(t, OcW, MwI)
    points = []
    for y in range(len(MwI[index])):
        tuple_point = tuple(ray_pre[i][y] for i in range(len(OcW)))
        points.append(tuple_point)
    points = [(a, b, c) for a, b, c, d in points]  # only obtain the first three numbers

    return points

def find_intersection(p1, p2, p3, p4):
    p1, p2, p3, p4 = map(np.array, [p1, p2, p3, p4])
    
    delta21 = p2 - p1
    delta34 = p3 - p4
    delta13 = p1 - p3
    
    A = np.array([
        [np.dot(delta21.T, delta21), np.dot(delta21.T, delta34)],
        [np.dot(delta21.T, delta34), np.dot(delta34.T, delta34)]
    ])
    
    b = np.array([
        -np.dot(delta13.T, delta21),
        -np.dot(delta13.T, delta34)
    ])
 
    try:
        # Solve systems of linear equations using the least squares method
        t1_t2 = np.linalg.inv(A).dot(b)
        t1, t2 = t1_t2
        # calculate the closest point
        closest_point_1 = p1 + t1 * (p2 - p1)
        closest_point_2 = p3 + t2 * (p4 - p3)
        # clculate and return the center point
        mid_point = (closest_point_1 + closest_point_2) / 2
        return mid_point
    except np.linalg.LinAlgError:
        return None  # returns None if there is no solution or an infinite number of solutions

def compute_intersections(OcW_left, MwI_left, OcW_right, MwI_right):

    intersections = []
    p1, p3 = OcW_left, OcW_right
    for (p2, p4) in zip(MwI_left, MwI_right):
        # calculte the intersection
        intersection = find_intersection(p1, p2, p3, p4)
        if intersection is not None:
            intersections.append(intersection)
    return intersections

