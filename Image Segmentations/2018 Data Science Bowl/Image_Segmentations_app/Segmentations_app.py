import tensorflow as tf
import numpy as np
import skimage.transform
import gradio as gr
import PIL

# Constants
IMG_WIDTH = 128
IMG_HEIGHT = 128

# Load TensorFlow model
model_path = 'D:/Computer-Vision/Image Segmentations/2018 Data Science Bowl/Image_Segmentations_app/saved_model/model.h5'
model = tf.keras.models.load_model(model_path)

def load_resize_input_images(img_pil):
    # Resize image using skimage
    img = skimage.transform.resize(np.array(img_pil), (IMG_HEIGHT, IMG_WIDTH),
                                   mode='constant', preserve_range=True)
    return img.astype(np.uint8)  # Convert to uint8 for compatibility

def generate_mask(img):
    img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])  # Resize using TensorFlow
    prediction = model.predict(tf.expand_dims(img, axis=0))[0]
    generated_mask = np.uint8(prediction > 0.5) * 255
    return generated_mask

def segmentation_function(image):
    input_image = load_resize_input_images(image)
    mask = generate_mask(input_image)

    # Ensure mask has shape (IMG_HEIGHT, IMG_WIDTH, 3) for RGB image
    mask = np.squeeze(mask)  # Remove singleton dimensions if any
    mask = np.stack([mask, mask, mask], axis=-1)  # Convert to RGB if necessary

    return mask  # Return the NumPy array directly


# Create the Gradio app
interface = gr.Interface(
    fn=segmentation_function,
    inputs=[gr.components.Image(interactive=True,label="Upload the cell image")],
    outputs=["image"],
    title="Image Segmentation App",
    description="Upload an image and get the segmented mask using a pre-trained model.",
)

# Launch Gradio interface
print("Starting Gradio interface...")
interface.launch(inbrowser=True,
             server_name="0.0.0.0", 
             server_port=7000,
             debug=True)
