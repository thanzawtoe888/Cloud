import numpy as np
import cv2
import cv2.ximgproc


# 1️⃣ Read the image
img = cv2.imread("../images/inputs/crop_pixel.png")
#convert to grayscale
gs = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#size of the image
print("Image shape:", img.shape)

#edge detection
edges = cv2.Canny(gs, 100, 200)
# Display the original and edge-detected images 
# cv2.imshow("Edge Detected Image", edges)

#draw skeleton between the edges
skeleton = cv2.ximgproc.thinning(edges)
# Display the skeletonized image
cv2.imshow("Skeletonized Image", skeleton)
cv2.imwrite("../images/outputs/skeletonized_image.png", skeleton)
cv2.imshow("gs Image", gs)
cv2.waitKey(0)
cv2.destroyAllWindows()

