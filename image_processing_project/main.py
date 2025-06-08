# main.py
import cv2  
import yaml
from preprocess import load_image, rescale_frame, grayscale_image, gaussian_blur,create_grid_mask, inpaint_image
from feature_extraction import extract_edges, compute_histogram, detect_calcined_clay_hsv
from utils import save_image


# Load config
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

image_path = config["input_image"]
image_size = tuple(config["image_size"])
output_dir = config["output_dir"]

img = load_image(image_path)
img_resized = rescale_frame(img, scale=1)  # Rescale the image to half its size
img_gray = grayscale_image(img_resized)  # Normalize and convert to grayscale
cv2.imshow("image", img_gray)
img_blurred = gaussian_blur(img_resized, kernel_size=(5, 5), sigma=0)

edges = extract_edges(img_blurred)
save_image(f"{output_dir}/edges.jpg", edges)

# Gridline positions
horizontal = [10, 125, 240, 366]
vertical = [85, 260, 435, 615, 795, 970, 1145, 1325, 1500, 1675, 1848]


# Create grid mask
mask = create_grid_mask(img_blurred, horizontal_positions=horizontal, vertical_positions=vertical)

# Inpaint the image using the mask
inpainted_img = inpaint_image(img_blurred, mask, method='telea')
save_image(f"{output_dir}/inpainted_image.jpg", inpainted_img)

edges = extract_edges(inpainted_img)
save_image(f"{output_dir}/edges_inpaint.jpg", edges)

#show the processed image
cv2.imshow("Processed Image", img_blurred)
cv2.imshow("Inpainted Image", inpainted_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

