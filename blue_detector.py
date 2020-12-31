import cv2
import numpy as np
import imutils
cap = cv2.VideoCapture(0)

while(1):

    # Take each frame
    _, frame = cap.read()
    
    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
   
    # define range of blue color in HSV
    lower_blue = np.array([110,50,50])
    upper_blue = np.array([130,255,255])
    
    # Threshold the HSV image to get only blue colors
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
    
    # Bitwise-AND mask and original image
    blue_res = cv2.bitwise_and(frame,frame, mask= blue_mask)
    
    #For countour and centroid.
    blue_cnt = cv2.findContours(blue_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    blue_cnt = imutils.grab_contours(blue_cnt)
    
    #for drawing the conyours around blue objects
    for c in blue_cnt:
        area  = cv2.contourArea(c)
        if area > 1000:
            cv2.drawContours(frame,[c],-1,(0,255,0),3)
            M = cv2.moments(c)
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            print('cx=',cx,'cy=',cy)
            cv2.circle(frame,(cx,cy),7,(0,255,0),-1)
            cv2.putText(frame,'centre',(cx-10,cy-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)
            cv2.imshow('frame',frame)
            cv2.imshow('mask',blue_mask)
            cv2.imshow('res',blue_res)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break
