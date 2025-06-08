import cv2
import matplotlib.pyplot as plt


# Load the image

image = cv2.imread("../images/outputs/edges_inpaint.jpg")


# Create a function to capture two points
points = []

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        print(f"Point {len(points)}: ({x}, {y})")

        if len(points) == 2:
            # Draw a line between the two points
            cv2.line(image, points[0], points[1], (0, 255, 0), 2)
            distance = ((points[0][0] - points[1][0])**2 + (points[0][1] - points[1][1])**2) ** 0.5
            print(f"Distance: {distance:.2f} pixels")
            cv2.imshow("Image", image)

# Show the image and wait for two mouse clicks

cv2.imshow("Image", image)
cv2.setMouseCallback("Image", click_event)
cv2.waitKey(0)
cv2.destroyAllWindows()
