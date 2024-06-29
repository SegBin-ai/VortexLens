import cv2
# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Use a binary threshold to distinguish screws and holes
_, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

# Use morphological operations to improve detection
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

# Find contours in the image
contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Loop through the contours to identify screws and screw holes
for contour in contours:
    # Calculate the area and the center of the contour
    area = cv2.contourArea(contour)
    if area > 50:  # Filter out small contours
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = 0, 0
        
        # Fit a circle around the contour
        ((x, y), radius) = cv2.minEnclosingCircle(contour)
        
        # Use the area and radius to distinguish between screws and holes
        if radius > 5 and radius < 15:  # Assuming screws and holes have a radius between 5 and 15
            # If the area is large, it's a screw; otherwise, it's an empty hole
            if area / (np.pi * radius * radius) > 0.6:
                # Draw a red circle for screws
                cv2.circle(image, (int(x), int(y)), int(radius), (0, 0, 255), 2)
            else:
                # Draw a blue circle for empty holes
                cv2.circle(image, (int(x), int(y)), int(radius), (255, 0, 0), 2)

# Save the annotated image
annotated_image_path = "/mnt/data/annotated_screws_and_holes.jpg"
cv2.imwrite(annotated_image_path, image)

# Display the path of the annotated image
annotated_image_path
