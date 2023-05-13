import numpy as np
import cv2 as cv

lst = []

for i in range(1020):
    temp = []
    for j in range(1280):
        temp.append(i*0.25)
    lst.append(temp)

lst = np.array(lst)

cv.imwrite('Shades.png',lst)
