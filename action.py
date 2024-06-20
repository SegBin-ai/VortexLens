import cv2
import numpy as np
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('/Users/aadityavoruganti/Desktop/Aadi/TerraVortex/VortexLens/Models/best_screw.pt')  # Ensure you have the correct weights file

# Define a function to detect screws in a frame
def detect_screws(frame):
    results = model(frame)
    screws = []
    for result in results:  # Iterate through the results
        for box in result.boxes:  # Iterate through the bounding boxes
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = box.conf[0].cpu().numpy()
            cls = box.cls[0].cpu().numpy()
            if cls == 0 and conf > 0.5:  # Assuming class_id 0 is for screws
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                screws.append((int(x1), int(y1), int(x2 - x1), int(y2 - y1), center_x, center_y))
    return screws, results

# Open the video file
video_path = '/Users/aadityavoruganti/Desktop/Aadi/TerraVortex/Video/Test1.mov'
cap = cv2.VideoCapture(video_path)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

# Initialize a dictionary to keep track of screw orders
screw_orders = {}
current_order = 1

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    screws, results = detect_screws(frame)
    screw_count = len(screws)

    # Display the number of detected screws on the frame in red and larger font
    cv2.putText(frame, f'Screws Detected: {screw_count}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
    
    # Display the YOLO model output log
    for i, result in enumerate(results):
        log_text = f'{i}: {result.orig_shape[0]}x{result.orig_shape[1]} {screw_count} screws, {result.speed["inference"]:.1f}ms'
        cv2.putText(frame, log_text, (10, 100 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    for screw in screws:
        x, y, w, h, cx, cy = screw
        if (cx, cy) not in screw_orders:
            screw_orders[(cx, cy)] = current_order
            current_order += 1
        order = screw_orders[(cx, cy)]
        
        # Draw a circle around the screw and the order number
        #cv2.circle(frame, (cx, cy), 10, (0, 255, 0), 2)
        cv2.putText(frame, str(order), (cx - 10, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    out.write(frame)
    cv2.imshow('Frame', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
