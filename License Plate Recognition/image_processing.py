import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

class ImagePreprocessor:
    def __init__(self, img_path):
        self.img_path = img_path
        self.original_img = cv2.imread(img_path)
        self.image = self.zoom_image(self.original_img, zoom_factor=1.5)
        self.gray, self.img_thresh = self.preprocess_image()

    def zoom_image(self, img, zoom_factor):
        if zoom_factor <= 1:
            raise ValueError("Zoom factor should be greater than 1.")
        height, width = img.shape[:2]
        new_width = int(width / zoom_factor)
        new_height = int(height / zoom_factor)
        x1 = (width - new_width) // 2
        y1 = (height - new_height) // 2
        x2 = x1 + new_width
        y2 = y1 + new_height
        cropped_img = img[y1:y2, x1:x2]
        zoomed_img = cv2.resize(cropped_img, (width, height), interpolation=cv2.INTER_LINEAR)
        return zoomed_img

    def preprocess_image(self):
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        structuringElement = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        imgTopHat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, structuringElement)
        imgBlackHat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, structuringElement)
        imgGrayscalePlusTopHat = cv2.add(gray, imgTopHat)
        gray = cv2.subtract(imgGrayscalePlusTopHat, imgBlackHat)
        img_blurred = cv2.GaussianBlur(gray, ksize=(5, 5), sigmaX=0)
        img_thresh = cv2.adaptiveThreshold(
            img_blurred, 
            maxValue=255.0, 
            adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            thresholdType=cv2.THRESH_BINARY_INV, 
            blockSize=19, 
            C=9
        )
        return gray, img_thresh
    
# paths= os.getcwd()+'\\ANPR_IMAGAS1\\'
# img_path= os.path.join(paths, os.listdir(paths)[1])
# img_path

# # Preprocess Image
# preprocessor = ImagePreprocessor(img_path)

# plt.figure(figsize=(10, 7))

# plt.subplot(1, 2, 1)
# plt.imshow(preprocessor.gray, cmap='gray')
# plt.axis('off')
# plt.subplot(1, 2, 2)
# plt.imshow(preprocessor.img_thresh, cmap='gray')
# plt.axis('off')
# plt.show()