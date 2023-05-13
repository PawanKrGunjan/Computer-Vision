import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv


# Create a matrix of 1280 X 1280 shape with Zeros
img = np.zeros((1280,1280), dtype = int)


# Chess size = 1280X1280 
# Color box length or width =  1280 / 8 = 160
# Let's create a box of 160 X 160 
# Create a matrix of 160 X 160 shape with 255
white =[]
for i in range(160):
    temp = []
    for j in range(160):
        temp.append(255)
    white.append(temp)

white = np.array(white)
print(white.shape)


# Now replace each 160 X 160 matrix of zeros with white matrix
for i in range(1,9):
    for j in range(1,9):
        if ((i+j)%2)==0:
            img[(i-1)*160:i*160,(j-1)*160:j*160] = white


# check the shape of image
print(img.shape)

#plot the chess with matplotlib          
plt.imshow(img) 
plt.show()

cv.imwrite('Chess In Gray Colour Space.jpg', img)