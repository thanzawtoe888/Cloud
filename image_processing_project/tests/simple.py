import cv2
import numpy as np
import matplotlib.pyplot as plt
import base64
from io import BytesIO

# Function to encode image to base64 for display in HTML
def encode_image_to_base64(image):
    """Encodes a CV2 image (numpy array) to a base64 string."""
    _, buffer = cv2.imencode('.png', image)
    return base64.b64encode(buffer).decode('utf-8')

# Load the images from the provided file paths
# In a real environment, these would be accessible directly or via a file system.
# For this simulation, we'll assume they are available or passed as base64.
# Since I cannot directly access local files, I'll simulate loading them if they were
# provided as base64. For this example, I'll use placeholders for actual image data.
# In a real execution environment, you would replace these with actual image loading.

# --- IMPORTANT: Replace the following with your actual image loading ---
# If you are running this locally, uncomment the lines below and ensure the image files are in the same directory.
try:
    img1 = cv2.imread('frame_0000.jpg')
    img2 = cv2.imread('frame_0015.jpg')

    if img1 is None or img2 is None:
        raise FileNotFoundError("Could not load one or both images. Make sure 'frame_0000.jpg' and 'frame_0015.jpg' are in the same directory as the script.")

except Exception as e:
    print(f"Error loading images directly: {e}. Attempting to create dummy images for demonstration.")
    # Create dummy images if actual loading fails (for demonstration purposes)
    img1 = np.zeros((400, 800, 3), dtype=np.uint8) + 200 # Light grey background
    cv2.putText(img1, "Image 1 (frame_0000.jpg)", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    img2 = np.zeros((400, 800, 3), dtype=np.uint8) + 200 # Light grey background
    cv2.putText(img2, "Image 2 (frame_0015.jpg)", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    print("Dummy images created. Please replace with your actual image loading code.")
# --- End of important section ---


# Define baseline parameters
# We'll draw a horizontal line near the bottom of the image.
# Adjust 'baseline_y_offset' based on where you want the baseline to be.
# A value of 0.8 means 80% down from the top.
baseline_y_offset = 0.8
line_color = (0, 0, 255) # Red color in BGR
line_thickness = 2

# Get image dimensions
height1, width1, _ = img1.shape
height2, width2, _ = img2.shape

# Calculate baseline Y-coordinate for each image
baseline_y1 = int(height1 * baseline_y_offset)
baseline_y2 = int(height2 * baseline_y_offset)

# --- Process img1 to create a binary mask and draw baseline ---
gray1_original = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
# Apply Otsu's thresholding for automatic threshold calculation
# This will try to separate foreground from background automatically
_, binary_mask1 = cv2.threshold(gray1_original, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Convert the single-channel binary mask to a 3-channel image to draw a colored line
# This is necessary because cv2.line expects a 3-channel image for a BGR color.
img1_with_baseline = cv2.cvtColor(binary_mask1, cv2.COLOR_GRAY2BGR)
cv2.line(img1_with_baseline, (0, baseline_y1), (width1, baseline_y1), line_color, line_thickness)

# --- Draw baseline on img2 (original color image) ---
img2_with_baseline = img2.copy()
cv2.line(img2_with_baseline, (0, baseline_y2), (width2, baseline_y2), line_color, line_thickness)

# --- Feature Detection and Matching for Transformation Matrix Calculation ---

# Convert images to grayscale for feature detection
# Note: gray1 is already the original grayscale for feature detection, not the binary mask
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) # Re-convert from original img1 for consistent feature detection
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# Initialize SIFT detector (or ORB, AKAZE, etc.)
sift = cv2.SIFT_create()

# Find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)

# Ensure descriptors are not None and have enough features
if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
    print("Not enough keypoints found to compute transformation matrix. Please ensure images have distinct features.")
    H = np.eye(3) # Return identity matrix if not enough features
else:
    # Use a Brute-Force Matcher with default parameters
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # Apply ratio test to find good matches (Lowe's ratio test)
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    # Ensure we have enough good matches to estimate a homography (at least 4 points)
    if len(good_matches) > 4:
        # Extract location of good matches
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # Find the Homography matrix (H) using RANSAC
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if H is None:
            print("Could not estimate Homography matrix. Returning identity matrix.")
            H = np.eye(3) # Fallback to identity if homography estimation fails
    else:
        print("Not enough good matches found to compute transformation matrix. Returning identity matrix.")
        H = np.eye(3) # Return identity matrix if not enough good matches

# Display the images with baselines using matplotlib
fig, axes = plt.subplots(1, 2, figsize=(15, 7))

axes[0].imshow(cv2.cvtColor(img1_with_baseline, cv2.COLOR_BGR2RGB))
axes[0].set_title('Binary Mask of Image 1 with Baseline (frame_0000.jpg)')
axes[0].axis('off')

axes[1].imshow(cv2.cvtColor(img2_with_baseline, cv2.COLOR_BGR2RGB))
axes[1].set_title('Image 2 with Baseline (frame_0015.jpg)')
axes[1].axis('off')

plt.tight_layout()
plt.show() # This will display the plot directly

# Output the results
print("--- Transformation Matrix (Homography) ---")
print("This 3x3 matrix describes the geometric transformation from Image 1 to Image 2.")
print("H =")
print(H)

print("\n--- Interpretation of the Homography Matrix ---")
print("The homography matrix H represents a 2D perspective transformation.")
print("It can account for translation, rotation, scaling, and even perspective distortion.")
print("The elements are generally interpreted as follows:")
print("H = [[h11, h12, h13],")
print("     [h21, h22, h23],")
print("     [h31, h32, h33]]")
print("\n- h11, h22: Primarily related to scaling (x and y directions).")
print("- h12, h21: Primarily related to rotation and shear.")
print("- h13, h23: Primarily related to translation (x and y directions).")
print("- h31, h32: Related to perspective distortion (if not close to zero, indicates non-affine transformation).")
print("- h33: Scaling factor (often normalized to 1).")

print("\nTo quantify 'how many transformation matrix change', you can look at the deviation of H from an identity matrix.")
print("An identity matrix [[1, 0, 0], [0, 1, 0], [0, 0, 1]] means no change.")
print("The further the values deviate from these, the larger the transformation.")

# You can also calculate the Euclidean distance from the identity matrix or decompose it
# For simplicity, we'll just show the matrix and its general interpretation.
