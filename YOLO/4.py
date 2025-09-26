from ultralytics import YOLO
import torch

# Load models
obb_model = YOLO("./saved_model/atcc_obb_876.pt")  # OBB-trained model
detect_model = YOLO("./saved_model/atcc_876.pt")   # transferred detect model

# Get all model parameters as lists
obb_params = list(obb_model.model.parameters())
detect_params = list(detect_model.model.parameters())

# Make sure both models have the same number of parameters
num_layers = min(len(obb_params), len(detect_params))
print(f"Comparing first {num_layers} layers...")

# Compare each layer
for i in range(num_layers):
    if torch.allclose(obb_params[i], detect_params[i]):
        print(f"Layer {i}: weights are identical (no transfer)")
    else:
        diff = (obb_params[i] - detect_params[i]).abs().mean().item()
        print(f"Layer {i}: weights differ, mean abs difference = {diff:.6f}")

# Optionally, overall norm comparison
obb_norm = sum(p.norm() for p in obb_params).item()
detect_norm = sum(p.norm() for p in detect_params).item()
print(f"\nTotal norm OBB model: {obb_norm:.4f}")
print(f"Total norm Detect model: {detect_norm:.4f}")
