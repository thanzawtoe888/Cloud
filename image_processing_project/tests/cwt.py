import cv2
import numpy as np
from skimage.filters import frangi
from skimage.morphology import closing, square
from skimage.measure import label, regionprops

# 1. Loading and converting to grayscale
image = cv2.imread('../images/inputs/frame_0015.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 1a. Apply Gaussian Blur to reduce small fluctuations and noise
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# 2. Frangi Filter to enhance crack-like structures
frangi_img = frangi(blur)

# 3. Canny Edge Detection
edges = cv2.Canny((frangi_img * 255).astype(np.uint8), 50, 150)

# 4. Distance Transform to approximate width
dist = cv2.distanceTransform(255 - edges, cv2.DIST_L2, 5)

# 5. Combine Frangi and Distance to approximate CWT
cwt = dist * frangi_img

# 6. Binarize for connected components
_, cwt_bin = cv2.threshold((cwt / cwt.max() * 255).astype(np.uint8), 50, 255, cv2.THRESH_BINARY)

# 7. Perform closing to connect segments
cwt_bin = closing(cwt_bin, square(3)).astype(np.uint8)

# 8. Label components and filter by aspect ratio
labels = label(cwt_bin)

filtered = np.zeros_like(cwt_bin)

for region in regionprops(labels):
    if region.minor_axis_length != 0 and region.major_axis_length / region.minor_axis_length > 2:
        coords = region.coords
        filtered[coords[:, 0], coords[:, 1]] = 255

# 9. Display final result
cv2.imshow("Cracks Detected", filtered)
cv2.waitKey(0)
cv2.destroyAllWindows()
