import cv2
import numpy as np

# Function to detect screw holes and draw bounding boxes
def detect_screw_holes(image_path, output_path):
    # Load the image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply GaussianBlur to reduce noise and improve circle detection
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    
    # Detect circles in the image using HoughCircles
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30,
                               param1=50, param2=30, minRadius=5, maxRadius=50)
    
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        
        for (x, y, r) in circles:
            # Draw bounding box
            cv2.rectangle(image, (x - r, y - r), (x + r, y + r), (0, 255, 0), 2)
            
    # Save the output image
    cv2.imwrite(output_path, image)

# Example usage
input_image_path = 'Pictures/sample.png'  # Path to your input image
output_image_path = 'Output/output_car_image.jpg'  # Path to save the output image with bounding boxes
detect_screw_holes(input_image_path, output_image_path)
