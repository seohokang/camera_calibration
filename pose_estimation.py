# -*- coding: utf-8 -*-
"""
Camera pose estimation according to one checkerboard
(We can get pose of two cameras according to the same checkerboard
then get the relative pose between two cameras)
Note: make sure that the origins in the two images are the same point in real world

Input: all checkerboard images from one camera
Output:    rvecs: rotation vector
           tvecs: translation vector
           
Reference: 
    https://github.com/njanirudh/Aruco_Tracker
    OpenCV-Python Tutorials » Camera Calibration and 3D Reconstruction
    https://opencv.org/license/
"""

import numpy as np
import cv2
import math


IMAGE_NAME = 'seoho_image/IMG-2492.jpg'
PARA_SAVE_PATH = "parameters/seoho_parameters_4_7.yaml"

# File storage in OpenCV
cv_file = cv2.FileStorage(PARA_SAVE_PATH, cv2.FILE_STORAGE_READ)

# Note : we also have to specify the type
# to retrieve otherwise we only get a 'None'
# FileNode object back instead of a matrix
mtx = cv_file.getNode("camera_matrix").mat()
dist = cv_file.getNode("dist_coeff").mat()

cv_file.release()

def rot_mat_to_euler_angle(R):
    sy = math.sqrt(R[0,0] * R[0,0] + R[1,0]*R[1,0])
    singular = sy < 1e-6
    if not singular:
        x = math.atan2(R[2,1], R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else:
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
    return np.array([x,y,z])

def get_body_heading(rvec):
    R, _ = cv2.Rodrigues(rvec)
    euler = rot_mat_to_euler_angle(R)
    print("euler", euler)
    gamma = euler[2] + np.pi/2
    return gamma

# Function to draw the axis
def draw(img, corners, imgpts):
    corner = tuple(np.asarray(corners[0].ravel(),dtype=np.int32))
    imgpts = np.asarray(imgpts, dtype=np.int32)
    img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255, 0, 0), 5)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0, 255, 0), 5)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0, 0, 255), 5)
    return img

# termination criteria (each small sqare in our checkerboard is 20mm x 20mm)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 32, 0.001)

objp = np.zeros((4*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:4].T.reshape(-1,2)

axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)

img = cv2.imread(IMAGE_NAME)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret, corners = cv2.findChessboardCorners(gray, (7,4),None)

if ret == True:
    corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
    if np.linalg.norm(corners2[0,0,:]) > np.linalg.norm(corners2[4*7-1,0,:]):
        corners2 = corners2[::-1,:,:]
    # Find the rotation and translation vectors.
    _,rvecs, tvecs, inliers = cv2.solvePnPRansac(objp, corners2, mtx, dist)
    # project 3D points to image plane
    imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)

    img = draw(img,corners2,imgpts)
    cv2.imshow('img',img)
    k = cv2.waitKey(1000)

cv2.destroyAllWindows()
print("rvecs")
print(str(rvecs))
print(get_body_heading(rvecs))

print("tvecs")
print(str(tvecs))
