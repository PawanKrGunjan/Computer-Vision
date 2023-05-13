# Import the library OpenCV 
import cv2 

  
# Import the image 
file_name = "Tata.jpg"

  
# Read the image 
image = cv2.imread(file_name, 1) 

  
# Convert image to image gray 
tmp = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 

  
# Applying thresholding technique 
_, alpha = cv2.threshold(tmp, 0, 255, cv2.THRESH_BINARY) 

  
# Using cv2.split() to split channels  
# of coloured image 
b, g, r = cv2.split(image) 

# Making list of Red, Green, Blue 
# Channels and alpha 
rgba = [b, g, r, alpha] 

  
# Using cv2.merge() to merge rgba 
# into a coloured/multi-channeled image 
dst = cv2.merge(rgba, 4) 

# the window showing output images
# with the corresponding thresholding
# techniques applied to the input images
cv2.imshow('original', image)
#cv2.imshow('Blue', b)
#cv2.imshow('Green', g)
#cv2.imshow('red', r)
cv2.imshow('Merged image', dst)

  
# Writing and saving to a new image 

cv2.imwrite("Tata_merge.jpg", dst) 

# De-allocate any associated memory usage
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()