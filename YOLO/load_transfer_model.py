from ultralytics import YOLO
import os

# ----------------------------
# Paths
# ----------------------------
cwd = os.getcwd()
obb_yaml_path = os.path.join(cwd, "datasets", "dota8", "dota8.yaml")
detect_yaml_path = os.path.join(cwd, "datasets", "detect8.yaml")  # axis-aligned dataset YAML

# ----------------------------
# Load trained OBB model
# ----------------------------
obb_model = YOLO("./saved_model/atcc_obb_876.pt")

# Transfer backbone from OBB to detect model
# This builds a detect model with OBB model weights transferred
detect_model = YOLO("yolo11n.yaml").load("./saved_model/atcc_obb_876.pt")

# Transfer class names and number of classes
detect_model.model.names = obb_model.names
detect_model.model.nc = obb_model.model.nc

# Verify weights for all layers that have matching shapes
for i, (obb_param, detect_param) in enumerate(zip(obb_model.model.parameters(), detect_model.model.parameters())):
    if obb_param.shape == detect_param.shape:
        assert (obb_param == detect_param).all(), f"Layer {i} mismatch!"
    else:
        print(f"Layer {i} skipped due to shape mismatch: {obb_param.shape} vs {detect_param.shape}")

print("All possible layers transferred successfully including classes.")

# Save the detect model with the new name
detect_model.save("./saved_model/atcc.pt")
print("Model saved as ./saved_model/atcc.pt")

# ----------------------------
# Print class names
# ----------------------------
print("OBB Model Classes:", obb_model.names)
print("Transferred Detect Model Classes:", detect_model.names)

# ----------------------------
# Validate OBB model
# ----------------------------
print("\nValidating OBB Model...")
metrics_obb = obb_model.val(data=obb_yaml_path, task="obb")
print("mAP50-95 (OBB):", metrics_obb.box.map)
print("mAP50 (OBB):", metrics_obb.box.map50)
print("mAP75 (OBB):", metrics_obb.box.map75)

# ----------------------------
# Validate transferred detect model
# ----------------------------
print("\nValidating Transferred Detect Model (Detect)...")
metrics_detect = detect_model.val(data=detect_yaml_path, task="detect")
print("mAP50-95 (Detect):", metrics_detect.box.map)
print("mAP50 (Detect):", metrics_detect.box.map50)
print("mAP75 (Detect):", metrics_detect.box.map75)

# ----------------------------
# Per-class APs
# ----------------------------
print("\nPer-class mAPs (OBB):", metrics_obb.box.maps)
print("Per-class mAPs (Detect):", metrics_detect.box.maps)
