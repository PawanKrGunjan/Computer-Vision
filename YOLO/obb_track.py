import supervision as sv
from ultralytics import YOLO
import cv2

# Load model
model = YOLO('atcc_obb_876.pt')  # OBB-enabled YOLO model

# Initialize tracker & annotator
tracker = sv.ByteTrack()
orient_annotator = sv.OrientedBoxAnnotator()

# Video source
frames_generator = sv.get_video_frames_generator(source_path='192.168.7.151.mp4')

for frame in frames_generator:
    # Run inference with oriented bounding boxes
    results = model(frame)[0]
    
    # Convert YOLO results to supervision format
    detections = sv.Detections.from_ultralytics(results)
    
    # Apply tracker
    tracked_detections = tracker.update_with_detections(detections)
    
    # Annotate frame
    annotated_frame = orient_annotator.annotate(
        scene=frame.copy(),
        detections=tracked_detections
    )
    
    # Display
    cv2.imshow("YOLO OBB Tracking", annotated_frame)
    
    # Break loop on 'q' key
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
