import cv2
import numpy as np
import imutils
cap = cv2.VideoCapture(0)

while(1):

    # Take each frame
    _, frame = cap.read()
     # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    #for red object
    lower_red = np.array([136, 87, 111], np.uint8) 
    upper_red = np.array([180, 255, 255], np.uint8) 
    
    #threshold the HSV image and get only red colors
    red_mask = cv2.inRange(hsv, lower_red, upper_red)
    
    # Bitwise-AND mask and original image
    red_res = cv2.bitwise_and(frame, frame, mask= red_mask)
    
    #For countour and centroid.
    red_cnt = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    red_cnt = imutils.grab_contours(red_cnt)
    
    #for drawinfg the contour around red object
    for c in red_cnt:
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
            cv2.imshow('mask',red_mask)
            cv2.imshow('res',red_res)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break
