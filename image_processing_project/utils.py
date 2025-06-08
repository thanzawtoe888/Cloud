# utils.py

import cv2
import numpy as np
import matplotlib.pyplot as plt

def show_image(image, title="Image", cmap=None):
    """Display an image with a title."""
    plt.imshow(image, cmap=cmap)
    plt.title(title)
    plt.axis('off')
    plt.show()

def convert_color_space(image, flag=cv2.COLOR_BGR2RGB):
    """Convert image color space using OpenCV flags."""
    return cv2.cvtColor(image, flag)

def save_image(path, image):
    """Save an image to the specified path."""
    cv2.imwrite(path, image)

def normalize_image(image):
    """Normalize image pixels to range [0, 1]."""
    return image.astype(np.float32) / 255.0

def resize_image(image, size=(224, 224)):
    """Resize image to the given size."""
    return cv2.resize(image, size)

def flatten_image(image):
    """Flatten image to 1D array."""
    return image.flatten()

def threshold_image(image, thresh=128):
    """Apply a binary threshold to the image."""
    _, binary = cv2.threshold(image, thresh, 255, cv2.THRESH_BINARY)
    return binary
