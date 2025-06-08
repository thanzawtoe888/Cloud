# Example: YOLOv8 Segmentation on an Image
from model import segment_roi_with_yolov8
import cv2

# Path to your image and (optionally) YOLOv8 segmentation model
image_path = "images/inputs/frame_0013.jpg"
model_path = "yolov8n-seg.pt"  # Download from https://github.com/ultralytics/ultralytics if not present

# Run segmentation
masks, segmented_image, results = segment_roi_with_yolov8(image_path, model_path)

# Show the first mask and the segmented image
if masks:
    cv2.imshow("First ROI Mask", masks[0])
cv2.imshow("Segmented Image", segmented_image)
cv2.imwrite("images/outputs/segmented_image.jpg", segmented_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
