import cv2 as cv
import matplotlib
import numpy as np
import matplotlib.pyplot as plt


# Sample Matrix
arr = np.array([[[ 0 , 0 , 0 ] , [255,255,255] ,[ 0 , 0 , 0 ]],
                [[255,255,255] , [ 0 , 0 , 0 ] ,[255,255,255]],
                [[ 0 , 0 , 0 ] , [255,255,255] ,[0 , 0 , 0 ]]])


# Create a matrix of 1280 X 1280 X 3 shape with Zeros
Zeros = np.zeros((1280,1280,3), dtype = int)

# Chess size = 1280 X 1280 X 3 
# Color box length or width =  1280 / 8 = 160
# Let's create a box of 160 X 160 X 3
# Create a matrix of 160 X 160 X 3 shape with 255
white =[]
for i in range(160):
    l = []
    for j in range(160):
        temp = []
        for k in range(3):
            temp.append(255)
        l.append(temp)
    white.append(l)
    
white = np.array(white)

print(white.shape)

# Now replace each 160 X 160 X 3 matrix of zeros with white matrix
for i in range(1,9):
    for j in range(1,9):
        if ((i+j)%2)==0:
            Zeros[(i-1)*160:i*160,(j-1)*160:j*160, :] = white

print('Zeros :', Zeros.shape)


# Create Border 
# Select Border colour
l = [ 25 , 100 , 0 ]
border = []
for i in range(60):
    temp = []
    for j in range(1280):
        temp.append(l)
    border.append(temp)

Border = np.array(border)
print('Border :',Border.shape)

# Create Corner 
corner = []
for i in range(60):
    temp = []
    for j in range(60):
        temp.append(l)
    corner.append(temp)

Corner = np.array(corner)
print('Corner :',Corner.shape)


# Add Chess Border
chess = np.concatenate((Zeros,Border), axis = 0)
chess = np.concatenate((Border, chess), axis = 0)

Border = np.concatenate((Border,Corner), axis = 1)
Border = np.concatenate((Border,Corner), axis = 1)

# Reshape Border
Border = Border.reshape(1400, 60, 3)
print('Border :',Border.shape)
print('Chess :',chess.shape)

# Add Chess Border
chess = np.concatenate((chess, Border), axis = 1)
chess = np.concatenate((Border, chess), axis = 1)

#print(Border.shape)
print('Chess :',chess.shape)


#plot the chess with matplotlib
plt.imshow(chess)
plt.show()

# Save the chess image
cv.imwrite('chess in RGB colour space.jpeg',chess)
