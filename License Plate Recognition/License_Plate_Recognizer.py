import os
import cv2
from PIL import Image
import random
from ultralytics import YOLO
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import warnings

# Filter all warnings
warnings.filterwarnings("ignore")

class LicensePlateRecognizer:
    def __init__(self):
        self.img_path = None
        self.image = None
        self.plate = None
        self.yolov8 = None
        self.preprocessor = None
        self.ocr_model = None

    def load_model(self, saved_model):
        self.yolov8 = YOLO(os.path.join(saved_model, 'yolov8.pt'))
        print('\n***YOLOv8 Model Successfully Loaded***\n')
        ch = os.path.join(saved_model, 'License_Number_Plate_Recognizer')
        self.preprocessor = TrOCRProcessor.from_pretrained(ch)
        self.ocr_model = VisionEncoderDecoderModel.from_pretrained(ch)
        print('\n***OCR Model Successfully Loaded***\n')

    def license_plate_detector(self):
        results = self.yolov8(self.img_path)
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
                        bounding_box['license_plate'] = self.image[y1:y2, x1:x2]
                        cv2.rectangle(self.image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        return bounding_box

    def number_plate_recognizer(self, cropped_image):
        plate = Image.fromarray(cropped_image).convert("RGB")
        pixel_values = self.preprocessor(images=plate, return_tensors="pt").pixel_values
        generated_ids = self.ocr_model.generate(pixel_values)
        generated_text = self.preprocessor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        return generated_text

    def load_image(self, img_path):
        self.img_path = img_path
        self.image = cv2.imread(self.img_path)

    def get_result(self):
        bounding_box = self.license_plate_detector()
        if bounding_box['license_plate'] is not None:
            plate_number = self.number_plate_recognizer(bounding_box['license_plate'])
            plate_number = plate_number.replace('-', ' ')
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.75
            color = (0, 255, 0)  # Green color for text
            thickness = 2
            text_position = (bounding_box['Box'][0], bounding_box['Box'][1] - 10)

            cv2.putText(self.image, plate_number, text_position, font, font_scale, color, thickness)

            try:
                # Attempt to display the image with bounding box and text
                cv2.imshow(f"License Plate Number: {plate_number}", self.image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            except cv2.error:
                # If cv2.imshow fails, save the image and use an alternative display method
                cv2.imwrite("output_with_plate_number.jpg", self.image)
                Image.open("output_with_plate_number.jpg").show(title=f"License Plate Number: {plate_number}")

            return plate_number
    def live_video_or_webcam(self, video_path=None):
        # If video_path is provided, use it, otherwise use the webcam
        if video_path:
            cap = cv2.VideoCapture(video_path)
        else:
            cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()
            if not ret:
                print("End of video or failed to grab frame")
                break

            # Process the current frame
            self.image = frame
            self.img_path = "frame.jpg"  # Temporary path for processing
            cv2.imwrite(self.img_path, frame)
            
            bounding_box = self.license_plate_detector()
            if bounding_box['license_plate'] is not None:
                plate_number = self.number_plate_recognizer(bounding_box['license_plate'])
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.75
                color = (0, 255, 0)  # Green color for text
                thickness = 2
                text_position = (bounding_box['Box'][0], bounding_box['Box'][1] - 10)

                cv2.putText(self.image, plate_number, text_position, font, font_scale, color, thickness)

            # Display the frame with bounding box and text
            cv2.imshow("License Plate Recognition", self.image)

            # Exit on pressing 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

# Initialize and load the model
recognizer = LicensePlateRecognizer()
saved_model = 'saved_model'
recognizer.load_model(saved_model)


# Example usage
license_plate_dir = 'Dataset/video/lp3.mp4'

recognizer.live_video_or_webcam(license_plate_dir)  # Use video file
recognizer.live_video_or_webcam()  # Use webcam if no video path is provided

"""
license_plate_dir= "D:\Computer-Vision\License Plate Recognition\Dataset\ANPR_IMAGAS1"
# List all files in the train directory
image_files = os.listdir(license_plate_dir)

# Randomly select 50 images
random_images = random.sample(image_files, 5)

# Output the selected image paths
for path in random_images:
    # Full path to the image file
    image_path = os.path.join(license_plate_dir, path)
    recognizer.load_image(image_path)
    # Get the result
    recognizer.get_result()
"""