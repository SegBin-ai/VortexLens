import cv2
import numpy as np

# Load the image using OpenCV
image = cv2.imread(image_path)

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Use GaussianBlur to reduce noise and improve circle detection
gray_blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Use HoughCircles to detect circles in the image
detected_circles = cv2.HoughCircles(
    gray_blurred,
    cv2.HOUGH_GRADIENT,
    dp=1.2,
    minDist=20,
    param1=50,
    param2=30,
    minRadius=5,
    maxRadius=20
)

# If some circles are detected, let's mark them
if detected_circles is not None:
    detected_circles = np.uint16(np.around(detected_circles))
    for circle in detected_circles[0, :]:
        x, y, r = circle
        # Draw the circumference of the circle
        cv2.circle(image, (x, y), r, (0, 255, 0), 2)
        # Draw a small circle to show the center
        cv2.circle(image, (x, y), 1, (0, 0, 255), 3)

# Save the annotated image
annotated_image_path = "/mnt/data/annotated_screws_screw_holes_opencv.jpg"
cv2.imwrite(annotated_image_path, image)

# Display the path of the annotated image
annotated_image_path
