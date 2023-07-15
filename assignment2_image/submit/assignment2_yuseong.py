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
l_img_gray = cv2.cvtColor(l_img, cv2.COLOR_BGR2GRAY)

######################## Right Image ########################
r_img = cv2.imread('2.jpg',-1)
r_img_gray = cv2.cvtColor(r_img, cv2.COLOR_BGR2GRAY)

##### Detect keypoint & extract local descriptor from two input images #####
orb = cv2.ORB_create()
kp1, desc1 = orb.detectAndCompute(l_img_gray, None)
kp2, desc2 = orb.detectAndCompute(r_img_gray, None)
#키포인트와 디스크럽터 저장

##### Match the keypoints between two images #####
matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
matches = matcher.match(desc1, desc2)
# 두 image의 디스크럽터 매치결과

##### Estimate a homography matrix using the RANSAC algorithm #####

left_pts =  np.float32([kp1[m.queryIdx].pt for m  in matches]).reshape(-1,1,2)

right_pts = np.float32([kp2[m.trainIdx].pt for m  in matches]).reshape(-1,1,2)

r2l_H,mask = cv2.findHomography( right_pts , left_pts , cv2.RANSAC, 5.0)
# cv2.findHomography( 1 , 2 ...) 1이미지(오른쪽) 을 2(왼쪽) 이미지 좌표로 변환

##### Project right image into the plane of the left image using the homography matrix
i_size = (l_img.shape[1]+r_img.shape[1],l_img.shape[0])
stitched_image = cv2.warpPerspective(r_img,r2l_H,i_size)
# 오른쪽 이미지를 왼쪽 이미지 평면에 투영

##### Fill the rest of the image #####
stitched_image[0:l_img_gray.shape[0],0:l_img_gray.shape[1]] = l_img

cv2.imshow('Stitched Image',trim(stitched_image))
cv2.waitKey()
cv2.destroyAllWindows()
