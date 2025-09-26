from ultralytics import YOLO
import torch

# Load models
obb_model = YOLO("./saved_model/atcc_obb_876.pt")  # OBB-trained model
detect_model = YOLO("./saved_model/atcc_876.pt")   # transferred detect model

# Get all model parameters as lists
obb_params = list(obb_model.model.parameters())
detect_params = list(detect_model.model.parameters())

num_layers = min(len(obb_params), len(detect_params))
print(f"Comparing first {num_layers} layers...")

diff_layers = []

for i in range(num_layers):
    # Skip comparison if shapes don't match (e.g., last classifier layer)
    if obb_params[i].shape != detect_params[i].shape:
        print(f"Layer {i}: shape mismatch {obb_params[i].shape} vs {detect_params[i].shape}, skipping")
        continue

    if not torch.allclose(obb_params[i], detect_params[i]):
        diff = (obb_params[i] - detect_params[i]).abs().mean().item()
        diff_layers.append((i, diff))

if diff_layers:
    print("Layers with differences:")
    for layer_idx, mean_diff in diff_layers:
        print(f"Layer {layer_idx}: mean abs difference = {mean_diff:.6f}")
else:
    print("All comparable layers are identical.")

# Optional: overall norm comparison
obb_norm = sum(p.norm() for p in obb_params).item()
detect_norm = sum(p.norm() for p in detect_params).item()
print(f"\nTotal norm OBB model: {obb_norm:.4f}")
print(f"Total norm Detect model: {detect_norm:.4f}")
