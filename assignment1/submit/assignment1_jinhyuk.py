import cv2
import numpy as np
import matplotlib.pyplot as plt

fig1 = plt.figure(1)
ax1  = fig1.subplots(1, 1)

## =================================================================
## C0 -> LiDAR Extrinsic Parameter
## =================================================================
R = np.array([[ 7.533745e-03, -9.999714e-01, -6.166020e-04],
              [ 1.480249e-02,  7.280733e-04, -9.998902e-01],
              [ 9.998621e-01,  7.523790e-03,  1.480755e-02]])

t = np.array([ -4.069766e-03, -7.631618e-02, -2.717806e-01 ])

C02L = np.hstack((R, t.reshape(3, 1)))
C02L = np.vstack((C02L, [0, 0, 0, 1]))
## =================================================================

## =================================================================
## C0 -> C2 Extrinsic Parameter
## =================================================================
R_02 = np.array([ [ 9.999758e-01, -5.267463e-03, -4.552439e-03],
                  [ 5.251945e-03,  9.999804e-01, -3.413835e-03],
                  [ 4.570332e-03,  3.389843e-03,  9.999838e-01]])

T_02 = np.array([   5.956621e-02,  2.900141e-04,  2.577209e-03 ])
C02C2 = np.hstack((R_02, T_02.reshape(3, 1)))
C02C2 = np.vstack((C02C2, [0, 0, 0, 1]))
## =================================================================

## =================================================================
## C2 -> L Extrinsic Parameter
## =================================================================
C22L = np.linalg.inv(C02C2) @ C02L
## =================================================================

## =================================================================
## Intrinsic Parameter
## =================================================================
K = np.array([[ 9.597910e+02,  0.000000e+00,  6.960217e+02],
              [ 0.000000e+00,  9.569251e+02,  2.241806e+02],
              [ 0.000000e+00,  0.000000e+00,  1.000000e+00]])

P = np.array([[ 7.215377e+02,  0.000000e+00,  6.095593e+02, 4.485728e+01],
              [ 0.000000e+00,  7.215377e+02,  1.728540e+02, 2.163791e-01],
              [ 0.000000e+00,  0.000000e+00,  1.000000e+00, 2.745884e-03]])

D = np.array([-3.691481e-01,
               1.968681e-01,
               1.353473e-03,
               5.677587e-04,
              -6.770705e-02])
## =================================================================

lidar_dtype=[('x', np.float32), 
             ('y', np.float32), 
             ('z', np.float32), 
             ('intensity', np.float32)]

image   = cv2.imread('0000000000.png', 1)
h, w, = 720, 1280

scan    = np.fromfile('0000000000.bin', dtype=lidar_dtype)
points  = np.stack((scan['x'], scan['y'], scan['z']), axis=-1)
ptcloud = np.insert(points, 3, 1, axis=1).T

## Problem (Generate LiDAR Point based on Camera Coordinates)
## =================================================================

## 1. LiDAR의 PointCloud를 Camera 2 좌표계로 변환하기
ptcloud_cam2 = C22L @ ptcloud
# C22L(변환 행렬)를 ptcloud(원본 포인트 클라우드 좌표)에 곱하여 카메라 좌표계로 변환합니다.

## 2. 카메라 좌표계에 맞게 뒤를 보는 부분 제거
valid_points = ptcloud_cam2[2, :] > 0
ptcloud_cam2_valid = ptcloud_cam2[:, valid_points]
# z 좌표가 양수인 포인트만 고려하여 카메라가 보는 부분만 남겨둡니다.
## 3. Normalization
homogeneous_coordinates = ptcloud_cam2_valid / ptcloud_cam2_valid[2, :]
# z 좌표로 나누어 주어 카메라 좌표계에서 동차 좌표계 (u, v, 1) 형태로 표현합니다.
## 4. 이미지 바깥에 있는 Point (Outlier) 제거
uv_coordinates = P @ homogeneous_coordinates
uv_coordinates = uv_coordinates[:2, :] / uv_coordinates[2]
uv_inliers = (uv_coordinates[0, :] >= 0) & (uv_coordinates[0, :] < w) &\
             (uv_coordinates[1, :] >= 0) & (uv_coordinates[1, :] < h)
# 동차 좌표계에서 이미지 평면에 사영하기 위해 프로젝션 행렬 P를 곱하고,
# u, v 좌표를 구한 후 이미지 범위 안에 있는 포인트만 골라냅니다.

u = uv_coordinates[0, uv_inliers]
v = uv_coordinates[1, uv_inliers]
z = ptcloud_cam2_valid[2, uv_inliers]
# 이미지 내부에 위치한 포인트의 u, v, z 좌표를 추출합니다.
## =================================================================

ax1.scatter(u, v, c=z, cmap='rainbow_r', alpha=0.5, s=2)
ax1.imshow(image)
ax1.set_title('Projection Image')
ax1.axis("off")

plt.show()