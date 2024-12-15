import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image, ImageDraw
from pathlib import Path
import os
import json
from tqdm import tqdm
import segmentation_models_pytorch as smp

# IoU Calculation Function for a Specific Class
def calculate_iou_for_class(predictions, targets, target_class_id):
    predictions = (predictions == target_class_id).cpu().numpy()
    targets = (targets == target_class_id).cpu().numpy()

    intersection = np.sum(predictions * targets)
    union = np.sum(predictions + targets) - intersection

    if union == 0:
        return float('nan')  # No presence of the class in ground truth or prediction
    return intersection / union

# Generate Ground Truth Mask from JSON
def create_class_mask_from_json(json_path, image_size):
    with open(json_path, 'r') as f:
        data = json.load(f)

    img_width, img_height = image_size
    mask = np.zeros((img_height, img_width), dtype=np.uint8)

    for obj in data["objects"]:
        if not obj["deleted"]:
            polygon = [(point[0], point[1]) for point in obj["polygon"]]
            
            # Check if the polygon has at least 2 valid points
            if len(polygon) < 2:
                print(f"Warning: Skipping invalid polygon with {len(polygon)} points in {json_path}")
                continue

            class_id = obj["id"]  # Extract class ID
            img = Image.new("L", (img_width, img_height), 0)
            ImageDraw.Draw(img).polygon(polygon, outline=class_id, fill=class_id)
            mask += np.array(img, dtype=np.uint8)

    return Image.fromarray(mask)

# Predict and Evaluate IoU for Road Class Across All Images
def evaluate_road_class_iou(model, image_root, json_root, target_class_id):
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])

    all_ious = []

    for subdir, _, files in os.walk(json_root):
        for file in tqdm(files, desc="Processing Images"):
            if file.endswith("_gtFine_polygons.json"):
                base_name = file.replace("_gtFine_polygons.json", "")
                json_path = Path(subdir) / file
                image_path = Path(image_root) / Path(subdir).relative_to(json_root) / f"{base_name}_leftImg8bit.jpg"

                if not image_path.exists():
                    print(f"Image not found for {json_path}")
                    continue

                # Load and preprocess the image
                image = Image.open(image_path).convert("RGB")
                input_image = transform(image).unsqueeze(0).to(device)

                # Generate ground truth mask from JSON
                original_size = image.size
                ground_truth_mask = create_class_mask_from_json(json_path, original_size)
                ground_truth_mask = ground_truth_mask.resize((512, 512))
                ground_truth_mask = torch.tensor(np.array(ground_truth_mask), dtype=torch.long)

                # Predict with the model
                model.eval()
                with torch.no_grad():
                    output = model(input_image)
                    predicted_mask = torch.argmax(output, dim=1).squeeze(0)

                # Calculate IoU for the target class (road)
                iou = calculate_iou_for_class(predicted_mask, ground_truth_mask, target_class_id)
                if not np.isnan(iou):
                    all_ious.append(iou)

    # Calculate average IoU for the road class
    if all_ious:
        average_iou = np.mean(all_ious)
        print(f"\nAverage IoU for class '{target_class_id}' (road): {average_iou:.4f}")
    else:
        print("\nNo valid IoU values found for class 'road'.")

# Model Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 34  # Total number of classes

# Define the model
model = smp.PSPNet(
    encoder_name="MobileNetV2",  # Ensure this matches your checkpoint's encoder
    encoder_weights=None,     # Avoid loading default weights
    in_channels=3,            # Match input channels
    classes=num_classes        # Match number of output classes
)

# Load the checkpoint with strict=False
checkpoint_path = "pspnet_segmentation_model_with_iou.pth"
try:
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    print("Model weights loaded successfully with 'strict=False'.")
except Exception as e:
    print(f"Error loading model weights: {e}")

model.to(device)

# Paths for Image and JSON Directories
image_root = r'C:\1 Main stuff\NUST stuff\7th sems\CV\project\idd20kII\leftImg8bit\val'
json_root = r'C:\1 Main stuff\NUST stuff\7th sems\CV\project\idd20kII\gtFine\val'

# Evaluate IoU for Road Class Across All Images
road_class_id = 1  # Road class ID as defined in JSON files
evaluate_road_class_iou(model, image_root, json_root, road_class_id)