from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
import os
from utils import ECG_Signal_Classifications

# Trained_Model_path
model_dir='D:/Computer-Vision/ECG Signal Classification/model/ECG_model.h5'
ECG=ECG_Signal_Classifications(model_dir)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.getcwd()+'/uploads/'

# Ensure the upload folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'dat_file' not in request.files or 'hea_file' not in request.files:
        return "Please upload both .dat and .hea files.", 400
    
    dat_file = request.files['dat_file']
    hea_file = request.files['hea_file']
    
    if dat_file.filename == '' or hea_file.filename == '':
        return "Please upload both .dat and .hea files.", 400
    
    dat_filename = secure_filename(dat_file.filename)
    hea_filename = secure_filename(hea_file.filename)
    
    if not dat_filename.endswith('.dat') or not hea_filename.endswith('.hea'):
        return "Invalid file types. Please upload .dat and .hea files.", 400
    
    dat_file_path = os.path.join(app.config['UPLOAD_FOLDER'], dat_filename)
    hea_file_path = os.path.join(app.config['UPLOAD_FOLDER'], hea_filename)
    
    dat_file.save(dat_file_path)
    hea_file.save(hea_file_path)
    print('*'*25)
    print('Data file path -->>',dat_file_path)
    # Perform prediction
    predicted_labels = ECG.predictions(dat_file_path)
    print(predicted_labels)
    return render_template('result.html', labels=predicted_labels)

if __name__ == '__main__':
    app.run(debug=True)
