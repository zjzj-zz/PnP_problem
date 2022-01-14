import numpy as np
import cv2

def camera_Matrix(f, cx, cy):
	return np.array([[f, 0, cx],[0, f, cy], [0, 0, 1]])

obj_points = np.array([[560, 500, 100], [500, 560, 100], [500, 500, 100], [560, 560, 100], [500, 500, 50], [440, 500, 50]], dtype=np.float32)
img_points = np.array([[[663, 306], [871, 313], [776, 366],[758, 263], [762, 469], [898, 562]]], dtype=np.float32)

cameraMatrix = camera_Matrix(640.0, 640.0, 360.0)
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
_R = R.T
print(R)
#print(_R)

cameraPosition = -np.matrix(R).T * np.matrix(t)

print(cameraPosition)
