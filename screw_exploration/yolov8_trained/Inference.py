import cv2
from ultralytics import YOLO

# Load the YOLOv8 model from the saved weights
model_path = 'best.pt'
model = YOLO(model_path)
frame = cv2.imread("test6.jpeg")


    # Perform inference
results = model(frame)

    # Display results
for result in results:
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Extract bounding box coordinates
        conf = box.conf[0]
        cls = box.cls[0]
        label = model.names[int(cls)]
            
        if conf > 0.5:  # Confidence threshold
            color = (0, 255, 0) if cls == 0 else (0, 0, 255)
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            text = f"{label} {conf:.2f}"
            cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
cv2.imshow('Result', frame)
cv2.waitKey(0)
cv2.destroyAllWindows()