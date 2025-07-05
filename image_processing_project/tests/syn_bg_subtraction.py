import cv2
import numpy as np
import yaml

# Load config
with open("../config.yaml", "r") as f:
    config = yaml.safe_load(f)
frame1_path = config["frame1_path"]
frame2_path = config["frame2_path"]
output_dir = config["output_dir"]

# Load images as numpy arrays
frame1 = cv2.imread("../images/outputs/frame_0_bin.jpg")
frame2 = cv2.imread("../images/outputs/frame_870s_bin.jpg")


    # Subtract frame1 from frame2 to get only the new content
diff = cv2.subtract(frame2, frame1)

    # Keep only the pixels with non-zero difference (i.e., the new line)
_, mask = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY)

    # Save or show the resulting image with only the new line
cv2.imwrite("img_subtraction.png", mask)
cv2.imshow("sub", mask)
cv2.waitKey(0)
cv2.destroyAllWindows()