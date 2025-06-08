from ultralytics import YOLO
import cv2
import numpy as np


def segment_roi_with_yolov8(image_path, model_path="yolov8n-seg.pt", conf=0.5):
    """
    Segment ROI in the image using YOLOv8 segmentation model.
    Args:
        image_path (str): Path to the input image.
        model_path (str): Path to the YOLOv8 segmentation model (default: yolov8n-seg.pt).
        conf (float): Confidence threshold for detections.
    Returns:
        masks (list of np.ndarray): List of binary masks for each detected ROI.
        segmented_image (np.ndarray): Image with segmented ROIs.
        results (YOLO results object): Raw YOLO results for further processing.
    """
    # Load model
    model = YOLO(model_path)
    # Read image
    image = cv2.imread(image_path)
    # Run inference
    results = model(image, conf=conf, task="segment")
    masks = []
    segmented_image = image.copy()
    for r in results:
        if hasattr(r, 'masks') and r.masks is not None:
            for mask in r.masks.data.cpu().numpy():
                mask_bin = (mask > 0.5).astype(np.uint8) * 255
                # Resize mask to match image shape
                mask_bin_resized = cv2.resize(mask_bin, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
                masks.append(mask_bin_resized)
                # Overlay mask on image
                color_mask = np.zeros_like(image)
                color_mask[:, :, 1] = mask_bin_resized  # Green channel
                segmented_image = cv2.addWeighted(segmented_image, 1, color_mask, 0.5, 0)
    return masks, segmented_image, results