# main.py
import cv2  
import yaml
from preprocess import load_image, rescale_frame, grayscale_image, gaussian_blur
from feature_extraction import extract_edges, morphological_skeleton
from utils import save_image


# Load config
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

image_path = config["input_image"]
image_size = tuple(config["image_size"])
output_dir = config["output_dir"]

img = load_image(image_path)
# img_resized = rescale_frame(img, scale=1)  # Rescale the image to half its size
img_gray = grayscale_image(img)  # Normalize and convert to grayscale
# img_blurred = gaussian_blur(img_gray, kernel_size=(3, 3), sigma=0)

# Edge detection and thresholding
edges = extract_edges(img_gray)



# Save results

save_image(f"{output_dir}/frame_870s_gray.jpg", img_gray)
save_image(f"{output_dir}/frame_870s_bin.jpg", edges)

cv2.waitKey(0)
cv2.destroyAllWindows()


