import cv2
import numpy as np
import matplotlib.pyplot as plt

def apply_harris_corner_detection(image_path, title):
    """
    Applies Harris Corner Detection to an image and visualizes the corners.

    Args:
        image_path (str): The path to the input image.
        title (str): The title for the plot.

    Returns:
        numpy.ndarray: The image with detected corners drawn.
    """
    # Load the image
    img = cv2.imread(image_path)

    # Check if image loading was successful
    if img is None:
        print(f"Error: Could not load image from {image_path}")
        return None

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Convert to float32 for cornerHarris function
    gray = np.float32(gray)

    # Apply Harris Corner Detection
    # Parameters:
    #   src: Input image (grayscale, float32)
    #   blockSize: Size of neighborhood considered for corner detection
    #   ksize: Aperture parameter for the Sobel operator
    #   k: Harris detector free parameter in the equation
    #      R = det(M) - k * (trace(M))^2
    dst = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)

    # Result is dilated to mark the corners, not just a single pixel
    dst = cv2.dilate(dst, None)

    # Threshold for an optimal value, it may vary depending on the image.
    # We are marking corners where the response is greater than 0.01 * max(dst)
    # The detected corners are marked in red on the original image.
    img[dst > 0.01 * dst.max()] = [0, 0, 255] # BGR color for red

    return img

# --- Main execution ---
if __name__ == "__main__":
    image_paths = {
        "frame_0000.jpg": "Original Image (frame_0000.jpg) with Detected Corners",
        "frame_0013.jpg": "Cracked Image (frame_0013.jpg) with Detected Corners"
    }

    # Create a figure to display the results
    plt.figure(figsize=(15, 7))

    for i, (path, title) in enumerate(image_paths.items()):
        processed_img = apply_harris_corner_detection(path, title)

        if processed_img is not None:
            # Matplotlib expects RGB, but OpenCV reads BGR, so convert
            processed_img_rgb = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)

            plt.subplot(1, 2, i + 1)
            plt.imshow(processed_img_rgb)
            plt.title(title)
            plt.axis('off') # Hide axes for cleaner display
        else:
            print(f"Skipping display for {path} due to loading error.")

    plt.tight_layout()
    plt.show()

