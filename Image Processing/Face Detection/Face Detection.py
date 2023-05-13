import cv2 as cv
import numpy as np

classifier = cv.CascadeClassifier('E:\Image Processing\Face Detection\haarcascade_frontalface_default.xml')

cam = cv.VideoCapture(0)

while True:
    
    _, img = cam.read()
    img = cv.flip(img,1)
    
    
    try:
        faces = classifier.detectMultiScale(img, 1.1, 5)
        for (x,y,w,h) in faces:
            cv.rectangle(img, (x,y),(x+w,y+h) , (0,180,0), 2)
    except:
        pass
    
    
    cv.imshow('Frame'  , img )
    
    if cv.waitKey(1) == 27:
        cam.release()
        break
    