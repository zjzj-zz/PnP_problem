import os
import math
import numpy as np
import cv2

COORD_SET_FOR_CALIB = [([663, 306], [560, 500, 100]), ([871, 313], [500, 560, 100]), ([776, 366], [500, 500, 100]), ([758, 263], [560, 560, 100]), ([762, 469], [500, 500, 50]), ([898, 562], [440, 500, 50]), ]

COORD_SET_FOR_TEST = [([1022, 376], [440, 560, 100]), ([935, 453], [440, 500, 100]), ([748, 354], [560, 560, 50]), ([849, 408], [500, 560, 50]), ([661, 402], [560, 500, 50]), ([978, 481], [440, 560, 50]), ]

FOV = 90
PW = 1280
PH = 720

FX = 1.0 / (2.0 * math.tan(np.radians(FOV) / 2.0)) * PW
FY = FX 
CX = PW / 2.0
CY = PH / 2.0
K = np.asarray([[FX, 0, CX], [0, FY, CY], [0, 0, 1], ])

CAMERA_MATRIX = K
print(K)
DIST_COEFFS = np.zeros((4, 1))

imagePoints = np.array([coord_set[0] for coord_set in COORD_SET_FOR_CALIB], dtype=np.float32)

objectPoints = np.array([coord_set[1] for coord_set in COORD_SET_FOR_CALIB], dtype=np.float32)

ret, rvec, tvec = cv2.solvePnP(objectPoints, imagePoints, CAMERA_MATRIX, DIST_COEFFS)

print("tvec")
print(tvec)
print("rvec")
print(rvec)

imagePointsTest = np.array([coord_set[0] for coord_set in COORD_SET_FOR_TEST], dtype=np.float32)

objectPointsTest = np.array([coord_set[1] for coord_set in COORD_SET_FOR_TEST], dtype=np.float32)

imgpts, jac = cv2.projectPoints(np.array(objectPointsTest, dtype=np.float32), rvec, tvec, CAMERA_MATRIX, DIST_COEFFS)

print("convert")
print("xyz -> img_pt uv")
for img_pt, uv, xyz in zip(imgpts, imagePointsTest, objectPointsTest):
	print(xyz, "->", img_pt[0], uv)

R_mat, _ = cv2.Rodrigues(rvec)
print("R_mat")
print(R_mat)

R_raw = R_mat.T
t_raw = -R_mat.T @ tvec

print("t_raw")
print(t_raw)
print("R_raw")

R = R_raw
t = t_raw

K_inv = np.linalg.inv(K)
print("convert")
print("xyz -> img_pt uv")
for coord_set in COORD_SET_FOR_TEST:
	uv = coord_set[0]
	xyz = coord_set[1]
	xyz_ = np.array(xyz).reshape((3,1))
	uvz_est = K @ R.T @ (xyz_ - t)
	uv_est = uvz_est[0:2] / uvz_est[2]

	print(f'{xyz} --> {uv_est[0][0]}, {uv_est[1][0]} / {uv}')
