import os
import cv2
import numpy as np
import pyopencl as cl
from openvino import Core
import supervision as sv

# Check model files
print(os.listdir('./atcc_obb_openvino_model'))

# Check for Intel iGPU availability
def intel_igpu_available():
    try:
        platforms = cl.get_platforms()
        for platform in platforms:
            if "Intel" in platform.name:
                devices = platform.get_devices(device_type=cl.device_type.GPU)
                for device in devices:
                    if "Intel" in device.name:
                        return True
    except Exception as e:
        print("OpenCL detection error:", e)
        return False
    return False

igpu_available = intel_igpu_available()
DEVICE = "GPU" if igpu_available else "CPU"
print("Using device:", DEVICE)

# Load OpenVINO model
core = Core()
model_xml = "./atcc_obb_openvino_model/atcc_obb.xml"
model_bin = "./atcc_obb_openvino_model/atcc_obb.bin"

model = core.read_model(model=model_xml, weights=model_bin)
detection_model = core.compile_model(model, device_name=DEVICE)

input_layer = detection_model.input(0)
output_layer = detection_model.output(0)
batch, channel, height, width = input_layer.shape
print("Input shape:", (batch, channel, height, width))

# Initialize tracker and annotator
tracker = sv.ByteTrack()
frames_generator = sv.get_video_frames_generator(source_path='192.168.7.151.mp4')
orient_annotator = sv.OrientedBoxAnnotator()

# Function to convert OBB to polygon
def obb_to_polygon(cx, cy, w, h, angle):
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    dx = w / 2
    dy = h / 2

    # Define corners
    corners = np.array([
        [-dx, -dy],
        [ dx, -dy],
        [ dx,  dy],
        [-dx,  dy]
    ])
    
    # Rotate and translate
    rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    rotated_corners = corners @ rotation_matrix.T
    abs_corners = rotated_corners + np.array([cx, cy])

    return abs_corners.flatten()  # 8 values

# Main loop
for frame in frames_generator:
    # Preprocess
    img = cv2.resize(frame, (width, height))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.transpose(2, 0, 1)  # HWC to CHW
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    # Inference
    preds = detection_model([img])[output_layer]  # ensure list input
    output = preds  # shape: (num_boxes, 7) [cx, cy, w, h, angle, conf, class_id]

    boxes, confs, class_ids = [], [], []

    for det in output:
        det = det.flatten()
        conf = float(det[5])
        if conf > 0.3:
            cx, cy, w, h, angle = map(float, det[:5])
            class_id = int(det[6])
            boxes.append([cx, cy, w, h, angle])
            confs.append(conf)
            class_ids.append(class_id)

    if len(boxes) > 0:
        polygons = np.array([obb_to_polygon(cx, cy, w, h, angle) for cx, cy, w, h, angle in boxes])
        
        detections = sv.Detections(
            xyxys=polygons,       # for oriented boxes
            confidence=np.array(confs),
            class_id=np.array(class_ids)
        )

        tracked_detections = tracker.update_with_detections(detections)
        annotated_frame = orient_annotator.annotate(scene=frame.copy(), detections=tracked_detections)
        cv2.imshow("YOLO OBB Tracking", annotated_frame)
    else:
        cv2.imshow("YOLO OBB Tracking", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
