import os
import cv2
import torch
from PIL import Image
from ultralytics import YOLO
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import warnings
import time
import logging

logging.getLogger('ultralytics').setLevel(logging.ERROR)  # Adjust logging level
warnings.filterwarnings("ignore")  # Filter all warnings

class LicensePlateRecognizer:
    def __init__(self, use_cuda=True):
        self.yolov8 = None
        self.ocr_model = None
        self.processor = None
        self.device = 'cuda' if use_cuda and torch.cuda.is_available() else 'cpu'
        self.frames = []
        self.confidences = []
        self.last_processed_time = time.time()

    def load_model(self):
        yolov8_model_path = os.path.join(os.getcwd(), 'saved_model', 'yolov8.pt')
        self.yolov8 = YOLO(yolov8_model_path)
        print(f'\n***YOLOv8 Model Successfully Loaded and Moved to {self.yolov8.device}***\n')

        ocr_model_path = os.path.join(os.getcwd(), 'saved_model', 'TrOCR')
        # Load the processor and model
        self.preprocessor = TrOCRProcessor.from_pretrained(ocr_model_path)
        self.ocr_model = VisionEncoderDecoderModel.from_pretrained(ocr_model_path).to(self.device)
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
        if bounding_box['license_plate'] is not None:
            cropped = bounding_box['license_plate']
            plate_number = self.number_plate_recognizer(cropped)
            plate_number = plate_number.replace('-', ' ')
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.75
            color = (0, 0, 255) 
            thickness = 2
            text_position = (bounding_box['Box'][0], bounding_box['Box'][1] - 10)
            cv2.putText(frame, plate_number, text_position, font, font_scale, color, thickness)
            self.frames.append((frame, bounding_box['confidence'], plate_number))
            self.confidences.append(bounding_box['confidence'])
            
            # Optional: Save individual detected license plates as images
            output_dir = os.path.join(os.getcwd(), "output", "plates")
            os.makedirs(output_dir, exist_ok=True)
            output_image_path = os.path.join(output_dir, f"plate_{plate_number}.jpg")
            cv2.imwrite(output_image_path, cropped)  # Save the frame with the plate number
            
    def process_image(self, image_path):
        # Load the image
        image = cv2.imread(image_path)
        # Process the image
        self.process_frame(image)
        if self.confidences:
            max_conf_idx = self.confidences.index(max(self.confidences))
            best_frame, best_conf, best_plate_number = self.frames[max_conf_idx]
            return best_plate_number

    def process_video(self, video_path=None, frame_skip=250):
        # Open video source (file or camera)
        cap = cv2.VideoCapture(video_path if video_path else 0)
    
        if not cap.isOpened():
            print("Error: Could not open video source.")
            return

        # Retrieve video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        print(f"Frames per second (FPS): {fps}")
    
        # Define the codec and create VideoWriter object
        output_video_path = os.path.join(os.getcwd(), "output", "detected_video.mp4")
        os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can change codec as needed
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
    
        best_frame = None  # Initialize best_frame to None
        frame_count = 0
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            current_time =  time.strftime("%H:%M:%S", time.localtime())
            
            # Skip frames to improve processing speed
            if frame_count % frame_skip != 0:
                continue
            # Get current time in HH:MM:SS format
            current_time = time.strftime("%H:%M:%S", time.localtime())
            
            self.process_frame(frame)
            if self.confidences:
                max_conf_idx = self.confidences.index(max(self.confidences))
                best_frame, best_conf, best_plate_number = self.frames[max_conf_idx]
                print(f"At :{current_time},\n The plate number: {best_plate_number} detected with confidence: {best_conf}")

            # Clear the stored frames and confidences for the next 30 seconds
            self.frames.clear()
            self.confidences.clear()

            # Write the best_frame or current frame to the video
            if best_frame is not None:
                out.write(best_frame)
            else:
                out.write(frame)
                
            # Check for 'q' key press to exit (useful if running in a windowed environment)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        # Release resources
        cap.release()
        out.release()
        print(f"Saved detected video as {output_video_path}")