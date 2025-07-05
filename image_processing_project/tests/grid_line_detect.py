import cv2
import numpy as np

# --- 1. Initial Image Analysis and Foundational Preparation ---

# Load the image from the specified path
image_path = 'frame_0000.jpg'
image = cv2.imread(image_path)

# Check if the image was loaded successfully
if image is None:
    print(f"Error: Could not load image at {image_path}")
else:
    # Convert the image to grayscale to work with a single intensity channel
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply a Gaussian blur to suppress high-frequency background texture.
    # The (7, 7) kernel size is chosen to smooth the concrete texture
    # without excessively blurring the grid lines.
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    # --- 2. Adaptive Thresholding: The Core of Line Segmentation ---

    # Apply adaptive thresholding to create a binary image.
    # ADAPTIVE_THRESH_MEAN_C: Threshold is the mean of the neighborhood.
    # THRESH_BINARY_INV: Inverts the output; dark lines become white foreground.
    # blockSize (41): Size of the pixel neighborhood. Must be large enough
    # to be representative of the local background relative to the line.
    # C (8): A constant subtracted from the mean to fine-tune the threshold.
    # This value is empirically tuned to remove noise without breaking lines.
    adaptive_thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV, 41, 8)

    # --- 3. Post-Segmentation Refinement using Morphological Operations ---

    # Define a small rectangular kernel for morphological operations.
    # A 3x3 kernel is effective for removing small "salt" noise.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    # Apply morphological opening (erosion followed by dilation) to remove
    # small noise objects without affecting the larger grid lines.
    opening = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    # --- 4. Precision Cleaning with Contour-Based Filtering ---

    # Find all contours in the morphologically cleaned image.
    # cv2.RETR_EXTERNAL retrieves only the outer contours of objects.
    # cv2.CHAIN_APPROX_SIMPLE compresses horizontal, vertical, and diagonal segments.
    contours, _ = cv2.findContours(
        opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a new black mask to draw the filtered contours onto.
    filtered_mask = np.zeros_like(opening)

    # Define a minimum area threshold to filter out small noise contours.
    # This value is chosen to be larger than typical noise blobs but
    # smaller than the smallest legitimate grid line segment.
    min_contour_area = 200

    # Loop through all found contours and filter them by area.
    for contour in contours:
        if cv2.contourArea(contour) > min_contour_area:
            # Draw the contours that pass the area filter onto the mask.
            # -1 for contourIdx draws all contours in the list (in this case, one by one).
            # 255 for color (white).
            # -1 for thickness fills the contour.
            cv2.drawContours(filtered_mask, [contour], -1, 255, -1)

    # --- 5. Displaying the Results ---

    # Display the original image and the final extracted grid lines.
    cv2.imshow('Original Image', image)
    cv2.imshow('Extracted Grid Lines', filtered_mask)
    cv2.imwrite('extracted_grid_lines.png', filtered_mask)

    # Wait for a key press and then close all windows.
    cv2.waitKey(0)
    cv2.destroyAllWindows()