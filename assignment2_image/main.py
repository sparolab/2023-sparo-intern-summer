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
l_img_gray = 

######################## Right Image ########################
r_img = cv2.imread('2.jpg',-1)
r_img_gray = 

##### Detect keypoint & extract local descriptor from two input images #####
orb = cv2.ORB_create()
kp1, desc1 = 
kp2, desc2 = 

##### Match the keypoints between two images #####
matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
matches = 

##### Estimate a homography matrix using the RANSAC algorithm #####
left_pts = 
right_pts = 
r2l_H, _ = cv2.findHomography( , , cv2.RANSAC, 5.0)

##### Project right image into the plane of the left image using the homography matrix
i_size = (l_img.shape[1]+r_img.shape[1],l_img.shape[0])
stitched_image = 

##### Fill the rest of the image #####
stitched_image[0:l_img_gray.shape[0],0:l_img_gray.shape[1]] = 

cv2.imshow('Stitched Image',trim(stitched_image))
cv2.waitKey()
cv2.destroyAllWindows()