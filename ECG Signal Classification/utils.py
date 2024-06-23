import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import wfdb
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from tensorflow import keras



class ECG_Signal_Classifications:
    def __init__(self, model_dir):
        self.model_path= model_dir
        self.Category  = {0: 'Normal', 
                          1: 'Abnormal', 
                          2: 'Artifact', 
                          3: 'Fusion', 
                          4: 'Escape'}
        self.segment_length = 360

    def load_model(self):
        model = keras.models.load_model(self.model_path)
        print("Model Successfully Loaded")
        return model
    def data_preprocessing(self,data_dir,n=7):
        if data_dir.endswith('.hea'):
            record_path = os.path.splitext(data_dir)[0]
        elif data_dir.endswith('.dat'):
            record_path = os.path.splitext(data_dir)[0]
        else:
            record_path=data_dir

        # Load the record
        """
          sampfrom : Sampling starting point [360, Assuming starting sample may be not correct]
          sampto   : Sampling ending Point   [n*360]
        """
        record = wfdb.rdrecord(record_path, sampfrom=360,sampto=360*n)
        
        return record.p_signal
    
    def predictions(self,data_dir):
        Records   = self.data_preprocessing(data_dir)
        ECG_model = self.load_model()
        pred_labels = []
        # Calculate the total number of segments
        total_segments = len(Records) // self.segment_length

        # Ensure the segment indices are within the bounds of the signal
        for i in tqdm(range(total_segments), desc="Processing segments"):
            # Move to the next segment
            start = i * self.segment_length
            end = start + self.segment_length

            # Extract segments from both leads (assuming two leads)
            if Records.shape[1] >= 2:
                segment = Records[start:end, :2]  # Extracting both channels
            else:
                segment = np.tile(Records[start:end, :1], (1, 2))  # Duplicate the single channel
            
            if segment.shape[0] == self.segment_length:
                # Predict on a single sample
                prediction = ECG_model.predict(segment.reshape((1, 360, 2)))
                prediction = prediction.argmax()

                # Reverse the mapping
                pred_label = self.Category[prediction]
                pred_labels.append(pred_label)
        
        return pred_labels

"""
# Trained_Model_path
model_dir='D:/Computer-Vision/ECG Signal Classification/model/ECG_model.h5'
data_dir="D:/Computer-Vision/ECG Signal Classification/mit-bih-arrhythmia-database/mit-bih-arrhythmia-database-1.0.0/100.dat"
ECG=ECG_Signal_Classifications(model_dir)
print('*'*25)
print(ECG.predictions(data_dir))
"""