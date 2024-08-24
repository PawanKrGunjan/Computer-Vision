import os
import sys
import cv2
import torch
from PIL import Image
from ultralytics import YOLO
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import warnings
from concurrent.futures import ThreadPoolExecutor
import logging
logging.getLogger('ultralytics').setLevel(logging.ERROR)  # Adjust logging level


# Filter all warnings
warnings.filterwarnings("ignore")

class LicensePlateRecognizer:
    def __init__(self, use_cuda=True):
        self.img_path = None
        self.image = None
        self.plate = None
        self.yolov8 = None
        self.ocr_model = None
        self.processor = None
        self.device = 'cuda' if use_cuda and torch.cuda.is_available() else 'cpu'
        self.executor = ThreadPoolExecutor(max_workers=2)  # Create a thread pool for parallel processing

    def load_model(self):

        yolov8_model_path = os.path.join(os.getcwd(),'saved_model', 'yolov8.pt')
        self.yolov8 = YOLO(yolov8_model_path)
        print(f'\n***YOLOv8 Model Successfully Loaded and Moved to {self.yolov8.device}***\n')
        
        ocr_model_path = os.path.join(os.getcwd(),'saved_model', 'TrOCR')
        # Load the processor and model
        self.preprocessor = TrOCRProcessor.from_pretrained(ocr_model_path)
        self.ocr_model = VisionEncoderDecoderModel.from_pretrained(ocr_model_path).to(self.device)
        self.ocr_model.config.some_setting = 'default_mode'
        print(f'\n***OCR Model Successfully Loaded and Moved to {self.ocr_model.device}***\n')

    def license_plate_detector(self, image):
        results = self.yolov8(image)
        bounding_box = {
            'confidence': 0,
            'Box': None,
            'license_plate': None
        }
        for result in results:
            if result.boxes:
                bounding_box['confidence'] = result.boxes.conf[0].cpu().numpy()
                x1, y1, x2, y2 = result.boxes.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                bounding_box['Box'] = [x1, y1, x2, y2]
            if bounding_box['confidence'] > 0:
                if bounding_box['Box']:
                    x1, y1, x2, y2 = bounding_box['Box']
                    if x1 is not None and y1 is not None and x2 is not None and y2 is not None:
                        bounding_box['license_plate'] = image[y1:y2, x1:x2]
                        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        return bounding_box

    def number_plate_recognizer(self, cropped_image):
        plate = Image.fromarray(cropped_image).convert("RGB")
        pixel_values = self.preprocessor(images=plate, return_tensors="pt").pixel_values.to(self.device)  # Move inputs to GPU
        generated_ids = self.ocr_model.generate(pixel_values)
        generated_text = self.preprocessor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return generated_text

    def process_frame(self, frame):
        bounding_box = self.license_plate_detector(frame)
        try:
            cropped=bounding_box['license_plate']
            plate_number = self.number_plate_recognizer(cropped)
            plate_number = plate_number.replace('-', ' ')
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.75
            color = (0, 0, 255) 
            thickness = 2
            text_position = (bounding_box['Box'][0], bounding_box['Box'][1] - 10)
            cv2.putText(frame, plate_number, text_position, font, font_scale, color, thickness)
            return plate_number, frame
        except:
            return 
    def process_image(self, image_path):
        # Load the image
        self.image = cv2.imread(image_path)
        # Process the image
        try:
            plate_number, frame= self.process_frame(self.image)
            return plate_number, frame
        except:
            return 

# Your existing code...
def main():
    recognizer=LicensePlateRecognizer()
    recognizer.load_model()
    while True:
        image_path = input("\nEnter the image path (or type 'exit' to quit): ")
        if image_path.lower() == 'exit':
            print("Exiting...")
            return
        if not os.path.exists(image_path):
            print(f"Error: The file {image_path} does not exist.")
            continue
        try:
            plate_number, processed_image = recognizer.process_image(image_path)
            # Ensure the 'output' directory exists in the current working directory
            output_dir = os.path.join(os.getcwd(), "output")
            os.makedirs(output_dir, exist_ok=True)

            # Create the output file path
            output_image_path = os.path.join(output_dir, f"output_{plate_number}.jpg")

            # Save the processed image
            cv2.imwrite(output_image_path, processed_image)

            print(f"Recognized License Plate: {plate_number}")
            print(f"Processed image saved as {output_image_path}")
        except:
            print('No plate found')

if __name__ == "__main__":
    main()