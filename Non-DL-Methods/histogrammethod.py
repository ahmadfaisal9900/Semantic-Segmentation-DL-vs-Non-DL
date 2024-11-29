import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Specify the folder containing the images
folder_path = r"C:\\1 Main stuff\\NUST stuff\\7th sems\\CV\\project\\208\\"  # Folder path

# Pre-defined static colors for segmented regions
static_colors = {
    "road": [0, 0, 255],       # Blue for road
    "vegetation": [0, 255, 0], # Green for vegetation
    "background": [255, 255, 0], # Yellow for background
}

# Get a list of image files in the folder
image_files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.png', '.jpeg'))]

# Process up to 4 images in the folder
for i, image_file in enumerate(image_files[:4]):  # Limit to 4 images
    # Load the image
    image_path = os.path.join(folder_path, image_file)
    image = cv2.imread(image_path)

    if image is None:
        print(f"Error: Could not load image {image_file}. Skipping...")
        continue

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB for consistency
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale

    # Compute the intensity histogram
    hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])

    # Define intensity ranges based on histogram analysis
    # Adjust these values based on the histogram peaks in your dataset
    road_range = (50, 100)  # Example intensity range for road
    vegetation_range = (100, 150)  # Example intensity range for vegetation
    background_range = (0, 50)  # Example intensity range for background

    # Create masks for each region
    road_mask = cv2.inRange(gray_image, road_range[0], road_range[1])
    vegetation_mask = cv2.inRange(gray_image, vegetation_range[0], vegetation_range[1])
    background_mask = cv2.inRange(gray_image, background_range[0], background_range[1])

    # Apply morphological operations to clean up the masks
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_CLOSE, kernel)
    vegetation_mask = cv2.morphologyEx(vegetation_mask, cv2.MORPH_CLOSE, kernel)
    background_mask = cv2.morphologyEx(background_mask, cv2.MORPH_CLOSE, kernel)

    # Create the segmented image
    segmented_image = np.zeros_like(image_rgb)
    segmented_image[road_mask > 0] = static_colors["road"]  # Assign road color
    segmented_image[vegetation_mask > 0] = static_colors["vegetation"]  # Assign vegetation color
    segmented_image[background_mask > 0] = static_colors["background"]  # Assign background color

    # Resize the final output for visualization
    resized_segmented_image = cv2.resize(segmented_image, 
                                         (int(segmented_image.shape[1] * 0.5), 
                                          int(segmented_image.shape[0] * 0.5)))

    # Display the final segmented image
    plt.figure(figsize=(10, 5))
    plt.imshow(cv2.cvtColor(resized_segmented_image, cv2.COLOR_BGR2RGB))  # Convert to RGB for correct display
    plt.title(f"Histogram-Based Segmentation: {image_file}")
    plt.axis("off")
    plt.show()

    # Plot the histogram
    plt.figure(figsize=(10, 5))
    plt.plot(hist, color='black')
    plt.title(f"Intensity Histogram: {image_file}")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.show()
