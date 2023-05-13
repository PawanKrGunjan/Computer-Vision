import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

arr = np.array([[[ 0 , 0 , 0 ] , [255,255,255] ,[ 0 , 0 , 0 ]],
                [[255,255,255] , [ 0 , 0 , 0 ] ,[255,255,255]],
                [[ 0 , 0 , 0 ] , [255,255,255] ,[0 , 0 , 0 ]]])


L = []
for i in range(255):
    l =[]
    for j in range(255):
        temp = []
        if (i+j)%2 == 0:
            value = 0
        else:
            value = 255
        for k in range(3):
            temp.append(value)
        l.append(temp)
    L.append(l)
    

arr = np.array(L)

plt.imshow(arr)
plt.show()

cv.imwrite('pattern.jpeg',arr)