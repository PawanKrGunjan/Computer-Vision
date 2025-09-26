from ultralytics import YOLO
import torch

# Create a fresh model
fresh_model = YOLO("yolo11n.yaml")
obb_model = YOLO("./saved_model/atcc_obb_876.py")
# Load trained model
loaded_model = YOLO("./saved_model/atcc_876.pt")

# Compare first conv layer weights
conv_fresh = next(fresh_model.model.parameters())
conv_loaded = next(loaded_model.model.parameters())

if torch.allclose(conv_fresh, conv_loaded):
    print("Weights NOT loaded (same as random initialization)")
else:
    print("Weights loaded successfully (different from random init)")
