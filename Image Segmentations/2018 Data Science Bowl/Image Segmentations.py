import skimage
import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
#from keras.models import save_model, load_model

print('Tensorflow :', tf.__version__)
print('scikit-image :', skimage.__version__)
print('Matplotlib :', matplotlib.__version__)

# Load the model
model_path = 'D:/Computer-Vision/Image Segmentations/2018 Data Science Bowl/Image_Segmentations_app/saved_model/model.h5'

model = tf.keras.models.load_model(model_path)
print(model.summary())

# Define constants
IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3

def load_resize_input_images(image_path):
    #img = skimage.io.imread(image_path)[:,:,:IMG_CHANNELS]
    # Open the image with PIL
    img_pil = Image.open(image_path)
    
    # Convert PIL image to NumPy array
    img = np.array(img_pil)
    img = skimage.transform.resize(img, 
                                   (IMG_HEIGHT, IMG_WIDTH), 
                                   mode='constant', 
                                   preserve_range=True)
    return img

def display(display_list):
    title = ['Input image', 'Predicted mask']
    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i + 1)
        plt.title(title[i])
        plt.imshow(tf.keras.utils.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()

model_path = 'D:/Computer-Vision/Image Segmentations/2018 Data Science Bowl/Image_Segmentations_app/test_image/test1.png'

test_image = load_resize_input_images(test_image_path)
print(test_image.shape)
prediction = model.predict(test_image[tf.newaxis, ...])[0]

generated_mask = tf.cast(prediction > 0.5, tf.uint8) 
    
display([test_image, generated_mask])
