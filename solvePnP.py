import os
import math
import numpy as np
import cv2

COORD_SET_FOR_TEST = [([848, 357], [750, 1900, 2900]), ([1100, 493], [1600, 2000, 2950]), ([1376, 627], [3000, 2240, 3000]), ([865, 813], [2000, 4300, 3300]), ([915, 807], [2300, 4400, 3500]), ([981, 824], [2600, 4300, 3450]), ]



FX = 1280.0
FY = FX 
CX = 960.0
CY = 540.0
K = np.asarray([[FX, 0, CX], [0, FY, CY], [0, 0, 1], ])

CAMERA_MATRIX = K
DIST_COEFFS = np.zeros((4, 1))

R_mat = np.array([[0.8660254, -0.36567685, -0.34099918],[-0.5, -0.63337088, -0.59062791], [0.0, 0.68199836 -0.7313537]], dtype=np.float32)

R_raw = R_mat.T
t_raw = np.array([[0.0],[0.0],[0.0]])

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
	uvz_est = K @ R.T @ (xyz_)
	uv_est = uvz_est[0:2] / uvz_est[2]

	print(f'{xyz} --> {uv_est[0][0]}, {uv_est[1][0]} / {uv}')
