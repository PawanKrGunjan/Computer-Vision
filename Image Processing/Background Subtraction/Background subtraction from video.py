# Python code for Background subtraction using OpenCV
import numpy as np
import cv2

#cap = cv2.VideoCapture('Tejas.mp4')
cap = cv2.VideoCapture(0) # for using CAM

fgbg = cv2.createBackgroundSubtractorMOG2()

while(1):
    ret, frame = cap.read()

    fgmask = fgbg.apply(frame)

    cv2.imshow('fgmask', fgmask)
    cv2.imshow('frame',frame )

    
    k = cv2.waitKey(30) & 0xFF
    if k == 27:
        break
    

cap.release()
cv2.destroyAllWindows()

