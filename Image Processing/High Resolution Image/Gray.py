import numpy as np
import cv2 as cv

img = []
for i in range(1020):
    temp = []
    for j in range(1280):
        temp.append(250)
    img.append(temp)
    
img = np.array(img)   

# Save image as png
cv.imwrite('Gray_250.png',img)