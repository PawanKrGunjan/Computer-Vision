from flask import Flask, request, render_template, send_from_directory
import os
import tensorflow as tf
import skimage.io
import skimage.transform
import matplotlib.pyplot as plt

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['RESULT_FOLDER'] = 'static/results/'

# Ensure the upload and result directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

# Load the model
model_path = 'D:/Computer-Vision/Image Segmentations/saved_model/model.h5'
model = tf.keras.models.load_model(model_path)

# Define constants
IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3

def load_resize_input_images(image_path):
    img = skimage.io.imread(image_path)[:,:,:IMG_CHANNELS]
    img = skimage.transform.resize(img, 
                                   (IMG_HEIGHT, IMG_WIDTH), 
                                   mode='constant', 
                                   preserve_range=True)
    return img

def generate_mask(image_path):
    test_image = load_resize_input_images(image_path)
    prediction = model.predict(test_image[tf.newaxis, ...])[0]
    generated_mask = tf.cast(prediction > 0.5, tf.uint8)
    return generated_mask

@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file part'
        file = request.files['file']
        if file.filename == '':
            return 'No selected file'
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            
            # Generate mask
            mask = generate_mask(filepath)
            mask_filename = 'mask_' + file.filename
            mask_filepath = os.path.join(app.config['RESULT_FOLDER'], mask_filename)
            plt.imsave(mask_filepath, tf.keras.utils.array_to_img(mask))

            return render_template('result.html', input_image=file.filename, mask_image=mask_filename)

    return render_template('upload.html')


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/results/<filename>')
def result_file(filename):
    return send_from_directory(app.config['RESULT_FOLDER'], filename)


if __name__ == '__main__':
    app.run(debug=True)
