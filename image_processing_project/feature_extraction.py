# feature_extraction.py

import cv2
import numpy as np

def extract_edges(image):
    """Extract edges using the Canny algorithm."""
    # Convert to grayscale if not already
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Ensure image is uint8
    if image.dtype != np.uint8:
        image = (255 * (image - np.min(image)) / (np.ptp(image) + 1e-8)).astype(np.uint8)
    return cv2.Canny(image, 100, 200)



def compute_histogram(image):
    """Compute color histogram."""
    hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8],
                        [0, 256, 0, 256, 0, 256])
    return cv2.normalize(hist, hist).flatten()

def detect_calcined_clay_hsv(image, lower_hsv=[0, 19, 80], upper_hsv=[20, 50, 255]):
    """
    Detect regions in the image corresponding to calcined clay based on HSV color thresholding.

    Parameters:
        image (np.ndarray): Input BGR image (as loaded by OpenCV).
        lower_hsv (list or np.ndarray): Lower HSV threshold [H, S, V].
        upper_hsv (list or np.ndarray): Upper HSV threshold [H, S, V].

    Returns:
        mask (np.ndarray): Binary mask where detected regions are white (255), others are black (0).
        result (np.ndarray): Image where only the detected regions are shown, others are blacked out.
    """
    # Convert BGR image to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Convert threshold lists to NumPy arrays
    lower = np.array(lower_hsv, dtype=np.uint8)
    upper = np.array(upper_hsv, dtype=np.uint8)

    # Create mask
    mask = cv2.inRange(hsv, lower, upper)

    # Optionally, apply morphological operations to clean up the mask
    kernel = np.ones((5, 5), np.uint8)
    mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel)

    # Extract detected regions from the original image
    result = cv2.bitwise_and(image, image, mask=mask_clean)
    
    # crop the result to the bounding box of the mask
    x, y, w, h = cv2.boundingRect(mask_clean)
    if w > 0 and h > 0:
        result = result[y:y+h, x:x+w]
        mask_clean = mask_clean[y:y+h, x:x+w]
        # Ensure the result is in the same format as the input image
        if result.size > 0 and result.dtype != np.uint8:
            result = (255 * (result - np.min(result)) / (np.ptp(result) + 1e-8)).astype(np.uint8)
        if mask_clean.size > 0 and mask_clean.dtype != np.uint8:
            mask_clean = (255 * (mask_clean - np.min(mask_clean)) / (np.ptp(mask_clean) + 1e-8)).astype(np.uint8)
    else:
        # No region detected, return empty arrays of correct shape
        result = np.zeros((0, 0, image.shape[2]), dtype=np.uint8) if len(image.shape) == 3 else np.zeros((0, 0), dtype=np.uint8)
        mask_clean = np.zeros((0, 0), dtype=np.uint8)
    
    return mask_clean, result


