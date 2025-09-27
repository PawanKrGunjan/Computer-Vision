import os
import cv2
import time
import numpy as np
import pyopencl as cl
from openvino import Core
import supervision as sv
from utils import process_detections  # Updated function with transform_params

# -------------------------------
# Device selection
# -------------------------------
DEVICE = "GPU" if any("Intel" in d.name for p in cl.get_platforms() for d in p.get_devices(cl.device_type.GPU)) else "CPU"
print("Using device:", DEVICE)

# -------------------------------
# Class names
# -------------------------------
class_names = {
    0: "2 Wheelers",
    1: "3 Wheelers",
    2: "4 Wheelers",
    3: "LCV",
    4: "Bus",
    5: "Truck",
    6: "Tractor",
    7: "HCM"
}

# -------------------------------
# Load OpenVINO model
# -------------------------------
model_path = "./saved_model/best_openvino_model"
model_files = os.listdir(model_path)
model_xml = next((os.path.join(model_path, f) for f in model_files if f.endswith(".xml")), None)
model_bin = next((os.path.join(model_path, f) for f in model_files if f.endswith(".bin")), None)
if not model_xml or not model_bin:
    raise FileNotFoundError("OpenVINO XML or BIN not found")

core = Core()
model = core.read_model(model=model_xml, weights=model_bin)
compiled_model = core.compile_model(model, device_name=DEVICE)
input_layer = compiled_model.inputs[0]
output_layer = compiled_model.outputs[0]
_, _, h_input, w_input = input_layer.shape
print("Model input shape:", (h_input, w_input))

# -------------------------------
# Tracker & annotator
# -------------------------------
tracker = sv.ByteTrack()
box_annotator = sv.BoxAnnotator()

# -------------------------------
# Video source
# -------------------------------
source_path = "test.mp4"
cap = cv2.VideoCapture(source_path)
if not cap.isOpened():
    raise RuntimeError("Cannot open video source")

# -------------------------------
# Preprocessing function
# -------------------------------
def preprocess_frame(frame, target_size=(640, 640)):
    target_w, target_h = target_size

    h_orig, w_orig = frame.shape[:2]
    if w_orig == 0 or h_orig == 0:
        # Invalid frame, return None
        return None, None

    scale = min(target_w / w_orig, target_h / h_orig)
    new_w, new_h = int(w_orig * scale), int(h_orig * scale)
    resized = cv2.resize(frame, (new_w, new_h))

    pad_x = (target_w - new_w) // 2
    pad_y = (target_h - new_h) // 2
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

    img_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    input_tensor = np.expand_dims(img_rgb.transpose(2, 0, 1), 0).astype(np.float32) / 255.0

    transform_params = {"scale": scale, "pad_w": pad_x, "pad_h": pad_y}
    return input_tensor, transform_params


# -------------------------------
# Main loop
# -------------------------------
prev_time = time.time()
while True:
    ret, frame = cap.read()
    if not ret:
        continue

    # Preprocess frame
    input_tensor, transform_params = preprocess_frame(frame, target_size=(w_input, h_input))
    if input_tensor is None:
        continue  # skip this frame

    # Inference
    result = compiled_model([input_tensor])[output_layer]

    # Process detections (uses transform_params for correct scaling)
    detections = process_detections(result, frame, transform_params)

    # Update tracker
    tracked = tracker.update_with_detections(detections)

    # Prepare safe tracked detections
    safe_tracked = sv.Detections(
        xyxy=tracked.xyxy,
        confidence=tracked.confidence,
        class_id=tracked.class_id
    )

    # Annotate frame
    annotated_frame = box_annotator.annotate(
        scene=frame.copy(),
        detections=safe_tracked
    )

    # FPS overlay
    curr_time = time.time()
    fps = 1.0 / (curr_time - prev_time)
    prev_time = curr_time
    cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Object counts overlay
    if len(safe_tracked.class_id) > 0:
        unique, counts = np.unique(safe_tracked.class_id, return_counts=True)
        counts_str = ", ".join([f"{class_names.get(u, str(u))}: {c}" for u, c in zip(unique, counts)])
    else:
        counts_str = "None"
    cv2.putText(annotated_frame, f"Counts: {counts_str}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    # Show frame
    cv2.imshow("Detections", annotated_frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()