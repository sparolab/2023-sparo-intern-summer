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

######################## 왼쪽 이미지 ########################
l_img = cv2.imread('1.jpg',-1)
l_img_gray = cv2.cvtColor(l_img, cv2.COLOR_BGR2GRAY)

######################## 오른쪽 이미지 ########################
r_img = cv2.imread('2.jpg',-1)
r_img_gray = cv2.cvtColor(r_img, cv2.COLOR_BGR2GRAY)

##### 두 입력 이미지에서 키포인트를 검출하고 로컬 디스크립터 추출 #####
orb = cv2.ORB_create()
kp1, desc1 = orb.detectAndCompute(l_img_gray, None)
kp2, desc2 = orb.detectAndCompute(r_img_gray, None)

##### 두 이미지 키포인트 매칭 #####
matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
matches = matcher.match(desc1, desc2)

##### RANSAC 알고리즘을 사용하여 호모그래피 행렬 추정 #####
left_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
right_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

r2l_H, _ = cv2.findHomography(right_pts, left_pts , cv2.RANSAC, 5.0)

##### 호모그래피 행렬을 사용하여 오른쪽 이미지를 왼쪽 이미지의 평면에 투영 #####
i_size = (l_img.shape[1]+r_img.shape[1],l_img.shape[0])
stitched_image = cv2.warpPerspective(r_img, r2l_H, i_size)


##### 나머지 이미지 채우기 #####
stitched_image[0:l_img_gray.shape[0],0:l_img_gray.shape[1]] = l_img

cv2.imshow('Stitched Image',trim(stitched_image))
cv2.waitKey()
cv2.destroyAllWindows()
