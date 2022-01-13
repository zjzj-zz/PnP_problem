import numpy as np
import cv2

def camera_Matrix(f, cx, cy):
	return np.array([[f, 0, cx],[0, f, cy],[0, 0, 1]])

obj_points = np.array([[750, 1900, 2900], [1650, 2000, 2950], [3000, 2240, 3000], [2000, 4300, 3300], [2300, 4400, 3500], [2600, 4300, 3450]]) * 1000.0
world_points = np.array([500, 500, 500]) * 1000.0
obj_points = obj_points - world_points

img_points = np.array([[[848, 357], [1100, 493], [1376, 627], [865, 813], [915, 807], [981, 824]]], dtype=np.float32)

cameraMatrix = camera_Matrix(1280.0, 960.0, 540.0)
distCoeffs = np.array([[0, 0, 0, 0]], dtype=np.float32)
r = np.array([], dtype=np.float32)

(x, r, t) = cv2.solvePnP(obj_points, img_points, cameraMatrix, distCoeffs)

print(r)
print(t)

imgpts, jac = cv2.projectPoints(obj_points, r, t, cameraMatrix, distCoeffs)

for n in range(6):
	print(imgpts[n][0])

print('\n', img_points)

R, jacob = cv2.Rodrigues(r)
print(R)

cameraPosition = -np.matrix(R).T * np.matrix(t)

print(cameraPosition)
