import cv2
import numpy as np
import os

class TsaiBoard:
    #Defines the board that we use for calibration
    #Arguments:
    # - square_size: size of the squares in the checkerboard pattern
    # - rows: number of rows in the checkerboard patterns
    # - cols: number of columns in the checkerboard patterns
    # - left_offset: distance between the left checkerboard pattern and the upright axis
    # - right_offset: distance between the right checkerboard pattern and the upright axis
    def __init__(self, square_size, rows, cols, left_offset, right_offset):
        self.square_size = square_size
        self.rows = rows
        self.cols = cols
        self.left_offset = left_offset
        self.right_offset = right_offset

def SeparateGrids(gray, debug=False):
    #Estimate the position of the two checkerboards in the image
    #Arguments:
    # - gray: numpy array for grayscale image (Works best if normalized and median blur is applied)
    # - debug: bool, display the output grids (Optional)
    #Returns:
    # - Bounding boxes for the two checkerboards
    #Find the two checkerboards on downsampled images
    scale_factor = 16
    mini = cv2.resize(gray, (gray.shape[1]//scale_factor, gray.shape[0]//scale_factor))
    mini = np.clip(255*(mini - np.percentile(mini, 20))/(np.percentile(mini, 80)-np.percentile(mini, 20)), 0, 255)
    mini = cv2.normalize(mini, None, 0, 255, cv2.NORM_MINMAX)
    mini_y = cv2.Sobel(mini,cv2.CV_64F, 0, 1, ksize=5)
    mini_y = np.abs(mini_y)
    mini_y = (mini_y > 4000)
    mini_x = cv2.Sobel(mini,cv2.CV_64F, 1, 0, ksize=5)
    mini_x = np.abs(mini_x)
    mini_x = (mini_x > 4000)
    double = np.logical_or(mini_x, mini_y).astype(np.uint8)
    numLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(double, 8, cv2.CV_32S)

    # cv2.imshow("test", (255/numLabels*labels).astype(np.uint8))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    #Sort by number of pixels
    stats.view(dtype=("int32,int32,int32,int32,int32")).sort(order=['f4'], axis=0)
    #Upsample bounding boxes
    board1 = stats[-2]*scale_factor
    board2 = stats[-3]*scale_factor

    #Pad the image in case grid is cut off
    padding_factor = gray.shape[1]//50
    board1 = [max(board1[0]-padding_factor,0), max(board1[1]-padding_factor,0), min(board1[2]+2*padding_factor,gray.shape[1]), min(board1[3]+2*padding_factor,gray.shape[0])]
    board2 = [max(board2[0]-padding_factor,0), max(board2[1]-padding_factor,0), min(board2[2]+2*padding_factor,gray.shape[1]), min(board2[3]+2*padding_factor,gray.shape[0])]

    if debug:
        #Plot bounding boxes of grids
        gray_box = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        gray_box = cv2.rectangle(gray_box, (board1[0],board1[1]), (board1[0]+board1[2],board1[1]+board1[3]), (255, 0, 0), thickness = 4)
        gray_box = cv2.rectangle(gray_box, (board2[0],board2[1]), (board2[0]+board2[2],board2[1]+board2[3]), (0, 0, 255), thickness = 4)
        imS = cv2.resize(gray_box, (gray_box.shape[1]//3, gray_box.shape[0]//3))
        cv2.imshow("Tsai grids found", imS)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return board1, board2

import matplotlib.pyplot as plt

def GetCorners(board_img, orig_img, board_dims):
    #Apply opencv corner detection method to segmented image with a single checkerboard
    #Arguments:
    # - board_img: image with background of region of interest segmented out
    # - orig_img: image to draw corners on
    # - board_dims: tuple of board dimensions (rows, columns)
    #Returns:
    # - corners: output numpy array of corner locations
    # - drawn_frame: OpenCV output of image with
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # Find the chessboard corners
    ret, corners = cv2.findChessboardCornersSB(board_img, board_dims, flags=cv2.CALIB_CB_EXHAUSTIVE)
    if not ret:
        print("Warning: not all corners were found. Checkerboard may be obscured.")
    if not corners is None:
        corners = cv2.cornerSubPix(board_img, corners, board_dims, (-1, -1), criteria)
        drawn_frame = cv2.drawChessboardCorners(orig_img, board_dims, corners, ret)
        return corners, drawn_frame
    else:
        print("Warning: No corners found. Check board dimensions.")
    return corners, orig_img

def OrderBoard(corner_points, board, origin_estimate, orientation):
    #Order the points in the board such that the first point is closest to the estimated origin, and
    #order by column, then by row moving away from the origin.
    #Arguments:
    # - corner_points: numpy array storing the detected corner points
    # - board: tsai calibration board object
    # - origin_estimate: the coordinate estimated as the orientation
    # - orientation: "Left" or "Right" checkerboard
    #Returns:
    # - corner_points: sorted in order
    #Originally ordered by column. See which direction the front corner is facing.
    pt1 = corner_points[0] - origin_estimate            #First point
    pt2 = corner_points[1] - origin_estimate            #Column neighbour
    pt3 = corner_points[board.rows] - origin_estimate   #Row neighbour
    x_orientation = pt3[0][0] - pt1[0][0]
    y_orientation = pt2[0][1] - pt1[0][1]
    #If points are upside down, flip columns
    if y_orientation > 0:
        corner_points = corner_points.reshape((board.cols, board.rows, 2))
        corner_points = np.flipud(corner_points)
        corner_points = corner_points.reshape((board.rows*board.cols,1, 2))
        corner_points = corner_points[::-1]
    #If points are left to right, flip rows (Left board)
    if orientation == "Left" and x_orientation > 0:
        corner_points = corner_points.reshape((board.cols, board.rows, 2))
        corner_points = np.flipud(corner_points)
        corner_points = corner_points.reshape((board.rows*board.cols,1, 2))
    #If points are right to left, flip rows (Right board)
    if orientation == "Right" and x_orientation < 0:
        corner_points = corner_points.reshape((board.cols, board.rows, 2))
        corner_points = np.flipud(corner_points)
        corner_points = corner_points.reshape((board.rows*board.cols,1, 2))
    return corner_points

def EstimateOrigin(box1, box2, b1_corners, b2_corners):
    #Get a crude estimate of the origin and sort the corner sets left to right
    #Arguments:
    # - box1, box2: the bounding box dimensions
    # - b1_corners, b2_corners: the two detected corner sets
    #Returns:
    # - origin_estimate: point of the approximate world origin
    # - ordered_boards: the corner sets in order
    if box1[0] < box2[0]:
        boxes = [box1, box2]
        ordered_boards = [b1_corners, b2_corners]
    else:
        boxes = [box2, box1]
        ordered_boards = [b2_corners, b1_corners]
    est_x = (boxes[0][0]+boxes[0][2]+boxes[1][0])//2
    est_y = (boxes[0][1]+boxes[0][3]+boxes[1][1]+boxes[1][3])//2
    origin_estimate = np.array([est_x, est_y])
    return origin_estimate, ordered_boards

def AssignWorldCoordinates(corner_set, board, orientation):
    #Allocate world coordinates to the cornersets
    #Arguments:
    # - corner_set: the ordered corner set
    # - board: the corresponding tsai board object with dimensions
    # - orientation: "Left" or "Right", which board view does this correspond to
    #Returns:
    # - coordinates: a (r*c,5) numpy array where each row is the X, Y, Z, u, v coordinates of the points
    if orientation == "Left":
        side_offset = board.left_offset
        side = 0
    elif orientation == "Right":
        side_offset = board.right_offset
        side = 1
    #Coordinates are in the form [X,Y,Z,u,v]
    coordinates = np.zeros(((board.cols, board.rows, 5)))
    coordinates[:,:,side] = 0
    coordinates[:,:,3:] = corner_set.reshape((board.cols, board.rows, 2))
    for column in range(board.cols): #0
        for row in range(board.rows): #1
            #Set the (X or Y) and Z for each point
            coordinates[column,row,1-side] = column*board.square_size + side_offset
            coordinates[column,row,2] = row*board.square_size
    return coordinates.reshape((board.rows*board.cols, 5))

def GetTsaiGrid(img, board, debug=False):
    #Steps through the pipeline to return a (2*r*c,5) numpy array where each row is the X, Y, Z, u, v coordinates of the points.
    #Arguments:
    # - img: an image of a tsai callibration object
    # - board: a tsai calibration object structure containing information of board dimensions
    # - debug: whether to display visual output as processing occurs (Optional)
    #Returns:
    # - coordinates: a (2*r*c,5) numpy array where each row is the X, Y, Z, u, v coordinates of the points of the two checkerboards
    ## termination criteria
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    #gray = cv2.medianBlur(gray, 7)
    board1, board2 = SeparateGrids(gray, debug)

    #Cut out boards and put onto a blank image with a white background
    # - Apply percentile based mapping to improve contrast
    board1_img = np.full(gray.shape, 255, dtype=np.uint8)
    cut1 = gray[board1[1]:board1[1]+board1[3], board1[0]:board1[0]+board1[2]]
    cut1 = np.clip(255*(cut1 - np.percentile(cut1, 20))/(np.percentile(cut1, 80)-np.percentile(cut1, 20)), 0, 255)
    board1_img[board1[1]:board1[1]+board1[3], board1[0]:board1[0]+board1[2]] = cut1

    board2_img = np.full(gray.shape, 255, dtype=np.uint8)
    cut2 = gray[board2[1]:board2[1]+board2[3], board2[0]:board2[0]+board2[2]]
    cut2 = np.clip(255*(cut2 - np.percentile(cut2, 20))/(np.percentile(cut2, 80)-np.percentile(cut2, 20)), 0, 255)
    board2_img[board2[1]:board2[1]+board2[3], board2[0]:board2[0]+board2[2]] = cut2

    #Display cutouts
    if debug:
        imS = np.full(gray.shape, 255, dtype=np.uint8)
        imS[board1[1]:board1[1]+board1[3], board1[0]:board1[0]+board1[2]] = cut1
        imS[board2[1]:board2[1]+board2[3], board2[0]:board2[0]+board2[2]] = cut2
        imS = cv2.resize(imS, (board2_img.shape[1]//3, board2_img.shape[0]//3))
        cv2.imshow("Board2", imS)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    #Get corners for each image
    board_dims = (board.rows, board.cols)
    b1_corners, drawn_frame = GetCorners(board1_img, img, board_dims)
    b2_corners, drawn_frame = GetCorners(board2_img, drawn_frame, board_dims)
    if debug:
    #Draw and display
        imS = cv2.resize(drawn_frame, (drawn_frame.shape[1]//3, drawn_frame.shape[0]//3))
        cv2.imshow("Corners detected", imS)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    #Assign to world coordinates
    # - Order boards based on bounding box positions [left, right] and estimate the origin
    origin_estimate, ordered_boards = EstimateOrigin(board1, board2, b1_corners,  b2_corners)
    ordered_boards[0] = OrderBoard(ordered_boards[0], board, origin_estimate, "Left")
    ordered_boards[1] = OrderBoard(ordered_boards[1], board, origin_estimate, "Right")
    c1 = AssignWorldCoordinates(ordered_boards[0], board, "Left")
    c2 = AssignWorldCoordinates(ordered_boards[1], board, "Right")
    return np.concatenate((c1, c2), axis=0)

def Get3d2dpoints(image, square_size, rows, columns, left_offset,  right_offset, debug=False):
    board = TsaiBoard(square_size, rows, columns, left_offset, right_offset)
    coordinates = GetTsaiGrid(image, board, debug)
    return coordinates

