import yaml
import os

# Current working directory
cwd = os.getcwd()

# Paths
dota8_yaml_path = os.path.join(cwd, "datasets", "dota8", "dota8.yaml")
detect_yaml_path = os.path.join(cwd, "datasets", "detect8.yaml")

# Define OBB dataset YAML content
obb_data = {
    'train': os.path.join("datasets", "dota8", "images", "train"),
    'val': os.path.join("datasets", "dota8", "images", "val"),
    'nc': 8,
    'names': ['2 Wheelers','3 Wheelers','4 Wheelers','LCV','Bus','Truck','Tractor','HCM']
}

# Save OBB YAML
os.makedirs(os.path.dirname(dota8_yaml_path), exist_ok=True)
with open(dota8_yaml_path, "w") as f:
    yaml.dump(obb_data, f)

print(f"Created OBB YAML at: {dota8_yaml_path}")

# Create Detect YAML (for axis-aligned validation)
detect_data = {
    'train': obb_data['train'],
    'val': obb_data['val'],
    'nc': obb_data['nc'],
    'names': obb_data['names']
}

with open(detect_yaml_path, "w") as f:
    yaml.dump(detect_data, f)

print(f"Created Detect YAML at: {detect_yaml_path}")
