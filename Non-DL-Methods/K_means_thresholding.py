import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import os

# Specify the folder containing the images
folder_path = r"C:\\1 Main stuff\\NUST stuff\\7th sems\\CV\\project\\208\\"  # Folder path

# Pre-defined static colors for up to 10 clusters
static_colors = [
    [255, 0, 0],   # Red
    [0, 255, 0],   # Green
    [0, 0, 255],   # Blue
    [255, 255, 0], # Yellow
    [0, 255, 255], # Cyan
    [255, 0, 255], # Magenta
    [128, 0, 128], # Purple
    [128, 128, 0], # Olive
    [0, 128, 128], # Teal
    [128, 128, 128] # Gray
]

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

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB for consistency

    # Convert the image to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # Define more refined thresholds for road segmentation
    lower_refined_road = np.array([10, 30, 70])  # Slightly stricter HSV bound for road
    upper_refined_road = np.array([24, 70, 110])

    # Segment the road using refined HSV thresholds
    road_mask_image = cv2.inRange(hsv_image, lower_refined_road, upper_refined_road)

    # Apply morphological operations to clean up the road mask
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    road_mask_image = cv2.morphologyEx(road_mask_image, cv2.MORPH_CLOSE, kernel)
    road_mask_image = cv2.morphologyEx(road_mask_image, cv2.MORPH_OPEN, kernel)

    # Assign colors: road = blue, non-road = black
    road_segmented_image = np.zeros_like(image)  # Create a blank image
    road_segmented_image[road_mask_image > 0] = [0, 0, 255]  # Set road areas to blue (BGR format)

    # Apply K-Means clustering to dynamically detect object colors in the image
    # Reshape the image into a 2D array of pixels for clustering
    pixels = image.reshape((-1, 3))

    # Use K-Means clustering to find dominant colors
    kmeans = KMeans(n_clusters=10, random_state=0).fit(pixels)  # Increased clusters to 10
    dominant_colors = kmeans.cluster_centers_.astype(int)

    # Create masks for each dominant color dynamically
    segmented_image_dynamic = np.zeros_like(image)  # Initialize a blank image
    for idx, color in enumerate(dominant_colors):
        lower_bound = np.array([max(0, c - 40) for c in color], dtype=np.uint8)
        upper_bound = np.array([min(255, c + 40) for c in color], dtype=np.uint8)
        
        # Create a mask for the current dominant color
        mask = cv2.inRange(image, lower_bound, upper_bound)
        
        # Assign static color to each detected region
        static_color = static_colors[idx % len(static_colors)]  # Cycle through pre-defined colors
        segmented_image_dynamic[mask > 0] = static_color

    # Combine the refined road mask and dynamically segmented objects
    segmented_combined = road_segmented_image.copy()
    non_road_mask_image = cv2.bitwise_not(road_mask_image)
    segmented_combined[non_road_mask_image > 0] = segmented_image_dynamic[non_road_mask_image > 0]

    # Resize the final output for visualization
    resized_segmented_combined = cv2.resize(segmented_combined, 
                                            (int(segmented_combined.shape[1] * 0.5), 
                                             int(segmented_combined.shape[0] * 0.5)))

    # Display the final segmented image with refined road mask and static object colors
    plt.figure(figsize=(10, 5))
    plt.imshow(cv2.cvtColor(resized_segmented_combined, cv2.COLOR_BGR2RGB))  # Convert to RGB for correct display
    plt.title(f"Refined Road (Blue) and Static Object Colors: {image_file}")
    plt.axis("off")
    plt.show()
