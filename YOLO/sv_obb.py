import supervision as sv
from ultralytics import YOLO
from ultralytics import settings

import cv2

model = YOLO('./saved_model/yolo11n_obb.pt')  # OBB-enabled YOLO model
tracker = sv.ByteTrack()
#frames_generator = sv.get_video_frames_generator(source_path='192.168.7.151.mp4')
source_path = "test.mp4"
frames_generator = sv.get_video_frames_generator(source_path=source_path)
orient_annotator = sv.OrientedBoxAnnotator()

for frame in frames_generator:
    result = model(frame)[0]
    # print(result.boxes)
    # print('CONFIDENCE', result.conf)
    # print('CLASS IDS', result.class_ids)
    # break
    detections = sv.Detections.from_ultralytics(result)
    detections = tracker.update_with_detections(detections)
    annotated_frame = orient_annotator.annotate(scene=frame.copy(), detections=detections)

    # Display annotated frame
    cv2.imshow("YOLO OBB Tracking", annotated_frame)
    
    # Press 'q' to exit display window
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()