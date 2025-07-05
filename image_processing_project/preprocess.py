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

def gaussian_blur(image, kernel_size=(3, 3), sigma=0):
    """Apply Gaussian blur to an image."""
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    return cv2.GaussianBlur(image, kernel_size, sigma)



