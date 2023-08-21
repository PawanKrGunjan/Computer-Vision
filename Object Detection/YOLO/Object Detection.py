import wget
import cv2
import numpy as np

wget.download("https://pjreddie.com/media/files/yolov3.weights")
wget.download("https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg")
wget.download("https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names")

# Load YOLO model
model = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

# Load COCO class names
classes = []
with open("coco.names", "r") as f:
    classes = f.read().strip().split("\n")

# Load image
image = cv2.imread("image.jpg")
height, width = image.shape[:2]

# Preprocess image
blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
model.setInput(blob)

# Perform inference
output_layers_names = model.getUnconnectedOutLayersNames()
outputs = model.forward(output_layers_names)

# Process detection results
conf_threshold = 0.5
nms_threshold = 0.4

boxes = []
confidences = []
class_ids = []

for output in outputs:
    for detection in output:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]

        if confidence > conf_threshold:
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

# Apply Non-Maximum Suppression
indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

# Draw bounding boxes and labels
colors = np.random.uniform(0, 255, size=(len(classes), 3))

if len(indices) > 0:
    for i in indices.flatten():
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        confidence = str(round(confidences[i], 2))
        color = colors[class_ids[i]]
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        cv2.putText(image, f"{label} {confidence}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

# Display the result
cv2.imshow("Object Detection", image)
cv2.imwrite("Object-Detection.jpg", image)
cv2.waitKey(0)
cv2.destroyAllWindows()