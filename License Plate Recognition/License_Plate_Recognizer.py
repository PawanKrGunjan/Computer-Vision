import os
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
    def __init__(self, frame_skip=20, use_cuda=True):
        self.img_path = None
        self.image = None
        self.plate = None
        self.yolov8 = None
        self.ocr_model = None
        self.processor = None
        self.original_plate_number=0
        self.frame_skip = frame_skip  # Number of frames to skip for real-time processing
        self.device = 'cuda' if use_cuda and torch.cuda.is_available() else 'cpu'
        self.executor = ThreadPoolExecutor(max_workers=2)  # Create a thread pool for parallel processing

    def load_model(self, saved_model):
        #yolov8_model_path = os.path.join(saved_model, 'yolov8.pt')
        self.yolov8 = YOLO(saved_model)
        print(f'\n***YOLOv8 Model Successfully Loaded and Moved to {self.yolov8.device}***\n')
        
        ocr_model_path ='PawanKrGunjan/license_plate_recognizer' #os.path.join(saved_model, 'ocr')
        
        self.preprocessor = TrOCRProcessor.from_pretrained(ocr_model_path)
        self.ocr_model = VisionEncoderDecoderModel.from_pretrained(ocr_model_path).to(self.device)  # Move OCR model to GPU
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

    def process_frame(self, frame, save=False):
        bounding_box = self.license_plate_detector(frame)
        if bounding_box['license_plate'] is not None:
            cropped=bounding_box['license_plate']
            if save:
                os.makedirs('plates', exist_ok=True)
                plates_dir= os.path.join(os.getcwd(),'plates')
                if isinstance(self.original_plate_number, int):
                    self.original_plate_number+=1
                saved_path=os.path.join(plates_dir,str(self.original_plate_number)+"_cropped.jpg")
                cv2.imwrite(saved_path,cropped)
                #print(f"Test image saved successfully to {saved_path}")
                cropped=cv2.imread(saved_path)
            plate_number = self.number_plate_recognizer(cropped)
            plate_number = plate_number.replace('-', ' ')
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.75
            color = (0, 255, 0) 
            thickness = 2
            text_position = (bounding_box['Box'][0], bounding_box['Box'][1] - 10)

            cv2.putText(frame, plate_number, text_position, font, font_scale, color, thickness)

        return plate_number, frame

    def process_image(self, image_path,save=True, show=False):
        # Load the image
        image = cv2.imread(image_path)
        self.original_plate_number = os.path.splitext(os.path.basename(image_path))[0]

        # Process the image
        plate_number=0
        plate_number, processed_image = self.process_frame(image, save=True)
        if show==True:
            try:
                # If cv2.imshow fails, save the image and use an alternative display method
                cv2.imwrite("output_with_plate_number.jpg", self.image)
                Image.open("output_with_plate_number.jpg").show(title=f"License Plate Number: {plate_number}")
            except cv2.error:
                # Attempt to display the image with bounding box and text
                cv2.imshow(f"License Plate Number: {plate_number}", self.image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

        return plate_number, processed_image

    def live_video_or_webcam(self, video_path=None):
        # If video_path is provided, use it, otherwise use the webcam
        cap = cv2.VideoCapture(video_path if video_path else 0)

        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                print("End of video or failed to grab frame")
                break

            frame_count += 1

            # Skip frames to improve processing speed
            if frame_count % self.frame_skip != 0:
                continue
            # Process frame using thread pool
            plate_number, processed_frame = self.executor.submit(self.process_frame, frame).result()

            # Display the frame with bounding box and text
            cv2.imshow("License Plate Recognition", processed_frame)

            # Exit on pressing 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
