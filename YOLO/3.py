from ultralytics import YOLO
import torch

# Load models
obb_model = YOLO("./saved_model/atcc_obb_876.pt")  # OBB-trained model
detect_model = YOLO("./saved_model/atcc_876.pt")   # transferred detect model

# Compare first conv layer weights
conv_obb = next(obb_model.model.parameters())
conv_detect = next(detect_model.model.parameters())

if torch.allclose(conv_obb, conv_detect):
    print("Weights are identical (no transfer happened)")
else:
    print("Weights differ, transfer seems successful")

# Optional: compare norms
print("OBB model first layer norm:", conv_obb.norm().item())
print("Detect model first layer norm:", conv_detect.norm().item())
