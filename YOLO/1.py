from ultralytics import YOLO

# Load your model
model = YOLO("./saved_model/atcc_876.pt")

# Print number of parameters
total_params = sum(p.numel() for p in model.model.parameters())
trainable_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
print(f"Total params: {total_params}")
print(f"Trainable params: {trainable_params}")
