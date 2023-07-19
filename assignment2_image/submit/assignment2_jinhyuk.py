import numpy as np
import cv2

# Function for erasing the blank area
def trim(frame):
    #crop top
    if not np.sum(frame[0]):
        return trim(frame[1:])
    #crop bottom
    if not np.sum(frame[-1]):
        return trim(frame[:-2])
    #crop left
    if not np.sum(frame[:,0]):
        return trim(frame[:,1:])
    #crop right
    if not np.sum(frame[:,-1]):
        return trim(frame[:,:-2])
    return frame

######################## Left Image ########################
l_img = cv2.imread('1.jpg',-1)
l_img_gray = cv2.cvtColor(l_img , cv2.COLOR_BGR2GRAY)

######################## Right Image ########################
r_img = cv2.imread('2.jpg',-1)
r_img_gray = cv2.cvtColor(r_img, cv2.COLOR_BGR2GRAY)# ORB로 특징점 및 특징 디스크립터 검출 (desc_orb.py)

##### Detect keypoint & extract local descriptor from two input images #####
orb = cv2.ORB_create()#ORB 추출기 생성
kp1, desc1 = orb.detectAndCompute(l_img_gray, None)# 키 포인트 검출과 서술자 계산
kp2, desc2 = orb.detectAndCompute(r_img_gray, None)# 키 포인트 검출과 서술자 계산

##### Match the keypoints between two images #####
matcher = cv2.BFMatcher(cv2.NORM_HAMMING)#거리측정 알고리즘을 전달하는 norm type 사용
#ORB로 디스크립터 검출기의 경우 NORM_HAMMING이 적합
matches = matcher.match(desc1, desc2)# 매칭 계산 

##### Estimate a homography matrix using the RANSAC algorithm #####
left_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
# 좋은 매칭점의 queryIdx로 원본 영상의 좌표 구하기
right_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1 ,2)
#좋은 매칭점의 trainIdx로 대상 영상의 좌표 구하기
r2l_H, _ = cv2.findHomography(right_pts, left_pts, cv2.RANSAC, 5.0)# 원근 변환 행렬 구하기


##### Project right image into the plane of the left image using the homography matrix
i_size = (l_img.shape[1]+r_img.shape[1],l_img.shape[0])
stitched_image = cv2.warpPerspective(r_img, r2l_H, i_size)

##### Fill the rest of the image #####
stitched_image[0:l_img_gray.shape[0],0:l_img_gray.shape[1]] = l_img

cv2.imshow('Stitched Image',trim(stitched_image))
cv2.waitKey()
cv2.destroyAllWindows()