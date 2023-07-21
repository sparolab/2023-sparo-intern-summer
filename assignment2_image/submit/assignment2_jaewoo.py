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
l_img_gray = cv2.cvtColor(l_img, cv2.COLOR_BGR2GRAY)                         # l_img 이미지를 흑백으로 변환

######################## Right Image ########################
r_img = cv2.imread('2.jpg',-1)
r_img_gray = cv2.cvtColor(r_img, cv2.COLOR_BGR2GRAY)                         # r_img 이미지를 흑백으로 변환

##### Detect keypoint & extract local descriptor from two input images #####
orb = cv2.ORB_create()                                                       # ORB(방향 그래디언트의 특징을 기반으로 하는) 키 포인트 탐지기 및 기술자를 생성
kp1, desc1 = orb.detectAndCompute(l_img_gray, None)                          # 첫 번째 이미지에서 키 포인트와 기술자를 계산하고 추출
kp2, desc2 = orb.detectAndCompute(r_img_gray, None)                          # 두 번째 이미지에서 키 포인트와 기술자를 계산하고 추출

##### Match the keypoints between two images #####
matcher = cv2.BFMatcher(cv2.NORM_HAMMING)                                    # 브루트 포스(Brute Force) 매처를 생성 (각 기술자 간의 거리를 계산하여 가장 가까운 것들을 매칭)
matches = matcher.match(desc1, desc2)                                        # 두 이미지 간의 기술자를 매칭

##### Estimate a homography matrix using the RANSAC algorithm #####
left_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
right_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1,1,2)
                                                                             # 매칭된 키포인트에서 좌표를 추출하여 numpy 배열로 변환
r2l_H, _ = cv2.findHomography( right_pts, left_pts, cv2.RANSAC, 5.0)
                                                                             # RANSAC 알고리즘을 이용해 오른쪽 이미지에서 왼쪽 이미지로의 호모그래피 매트릭스 추출
##### Project right image into the plane of the left image using the homography matrix
i_size = (l_img.shape[1]+r_img.shape[1],l_img.shape[0])                      # 스티치된 이미지의 크기를 정의
stitched_image = cv2.warpPerspective(r_img,r2l_H,i_size)                     # 호모그래피 매트릭스를 사용해 오른쪽 이미지를 왼쪽 이미지의 평면에 투영

##### Fill the rest of the image #####
stitched_image[0:l_img_gray.shape[0],0:l_img_gray.shape[1]] = l_img          # 스티치된 이미지의 나머지 부분에 왼쪽 이미지를 채움

cv2.imshow('Stitched Image',trim(stitched_image))
cv2.waitKey()
cv2.destroyAllWindows()
