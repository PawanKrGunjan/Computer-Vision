import argparse
from License_Plate_Recognizer import LicensePlateRecognizer

def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Process image or video for license plate recognition.')
    parser.add_argument('--image', type=str, help='Path to the image file.')
    parser.add_argument('--video', type=str, help='Path to the video file.')
    args = parser.parse_args()

    # Process based on provided arguments
    if args.image:
        if is_image(args.image):
            print(f"Processing image: {args.image}")
            plate_number = recognizer.process_image(args.image)
            print(f"Detected plate number: {plate_number}")
        else:
            print("The provided file path does not correspond to an image.")
    elif args.video:
        # Process video if provided
        if is_video(args.video):
            print(f"Processing video: {args.video}")
            recognizer.process_video(args.video, frame_skip=50)
        else:
            print(f"The provided file path {args.video} does not correspond to a video.")
    else:
        # If no video path is provided, process live video from the camera
        # print("Going live, press 'q' to exit")
        recognizer.process_video(None, frame_skip=90)
     

def is_image(file_path):
    """Check if the file path corresponds to an image based on file extension."""
    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif']
    return any(file_path.lower().endswith(ext) for ext in image_extensions)

def is_video(file_path):
    """Check if the file path corresponds to a video based on file extension."""
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
    return any(file_path.lower().endswith(ext) for ext in video_extensions)

if __name__ == "__main__":
    # Initialize the recognizer
    recognizer = LicensePlateRecognizer()
    # Load the models
    recognizer.load_model()
    main()
