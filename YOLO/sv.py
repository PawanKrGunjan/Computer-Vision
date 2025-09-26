from ultralytics import YOLO
import cv2
import os

# ----------------------------
# Paths
# ----------------------------
cwd = os.getcwd()
detect_yaml_path = os.path.join(cwd, "datasets", "detect8.yaml")
source_path = '192.168.7.151.mp4'  # input video

# Load the detect model
detect_model = YOLO("./saved_model/atcc.pt")

# Correct class names
# detect_model.model.names = {i: name for i, name in enumerate([
#     '2 Wheelers', '3 Wheelers', '4 Wheelers', 'LCV',
#     'Bus', 'Truck', 'Tractor', 'HCM'
# ])}
# detect_model.model.nc = 8
print(detect_model.names)
# ----------------------------
# Run tracking in stream mode
# ----------------------------
results_generator = detect_model.track(
    source=source_path,
    tracker="bytetrack.yaml",
    show=False,    # disable built-in display
    stream=True    # generator mode
)

# ----------------------------
# Display frames manually
# ----------------------------
for result in results_generator:
    # Get annotated frame
    frame = result.plot()

    # Show the frame
    cv2.imshow("Tracking", frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
