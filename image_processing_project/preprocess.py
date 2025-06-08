# preprocess.py

import cv2
import numpy as np

def load_image(path):
    """Load an image from a file path."""
    return cv2.imread(path, cv2.IMREAD_COLOR)

def resize_image(image, size=(224, 224)):
    """Resize an image to the given size."""
    return cv2.resize(image, size)

def rescale_frame(frame, scale=0.2):
    width = int(frame.shape[1] * scale) ## [1] mean width, [0] mean height
    height = int(frame.shape[0] * scale) ## [1] mean width, [0] mean height
    dimensions = (width, height)

    return cv2.resize(frame, dimensions, interpolation=cv2.INTER_AREA)

def grayscale_image(image):
    """Convert an image to grayscale."""
    if len(image.shape) == 3:  # Check if the image is colored
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image  # Already grayscale

def normalize_image(image):
    """Normalize image pixel values to [0, 1]."""
    return image.astype(np.float32) / 255.0

def gaussian_blur(image, kernel_size=(5, 5), sigma=0):
    """Apply Gaussian blur to an image."""
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    return cv2.GaussianBlur(image, kernel_size, sigma)

#color conversion functions
def bgr_to_hsv(image):
    """Convert BGR image to HSV color space."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

def draw_gridlines(img, horizontal_positions=None, vertical_positions=None,
                   h_color=(255, 0, 0), v_color=(0, 255, 0), thickness=5):
    """
    Draw horizontal and vertical gridlines on the image.
    
    Parameters:
    - img: Image to draw on.
    - horizontal_positions: List of Y coordinates for horizontal lines.
    - vertical_positions: List of X coordinates for vertical lines.
    - h_color: Color of horizontal lines.
    - v_color: Color of vertical lines.
    - thickness: Line thickness.
    """
    if horizontal_positions:
        for y in horizontal_positions:
            cv2.line(img, (0, y), (img.shape[1], y), h_color, thickness, lineType=cv2.LINE_AA)
    
    if vertical_positions:
        for x in vertical_positions:
            cv2.line(img, (x, 0), (x, img.shape[0]), v_color, thickness, lineType=cv2.LINE_AA)
    
    return img

def create_grid_mask(image_shape, horizontal_positions=None, vertical_positions=None, thickness=5):
    """
    Create a binary mask with white gridlines on a black background.
    """
    # Ensure image_shape is a tuple, not an array
    if isinstance(image_shape, np.ndarray):
        image_shape = image_shape.shape

    mask = np.zeros(image_shape[:2], dtype=np.uint8)  # Single channel for mask

    if horizontal_positions:
        for y in horizontal_positions:
            cv2.line(mask, (0, y), (image_shape[1], y), 255, thickness, lineType=cv2.LINE_AA)
    
    if vertical_positions:
        for x in vertical_positions:
            cv2.line(mask, (x, 0), (x, image_shape[0]), 255, thickness, lineType=cv2.LINE_AA)
    
    return mask

def inpaint_image(image, mask, method='telea'):
    """
    Inpaint the image using the mask. 
    `method`: 'telea' or 'ns' (Navier-Stokes).
    """
    inpaint_method = cv2.INPAINT_TELEA if method == 'telea' else cv2.INPAINT_NS

    # Ensure image is uint8 for OpenCV inpainting
    if image.dtype != np.uint8:
        image = (image * 255).clip(0, 255).astype(np.uint8)

    return cv2.inpaint(image, mask, inpaintRadius=3, flags=inpaint_method)
