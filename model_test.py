import cv2
from ultralytics import YOLO
import threading

# Load the model
model = YOLO('/Users/aadityavoruganti/Desktop/Aadi/TerraVortex/VortexLens/best_screw.pt')

# Initialize the video capture
cap = cv2.VideoCapture(0)  # 0 is the default camera
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Initialize a variable to hold the latest frame
frame = None

# Function to capture frames from the camera
def capture_frames():
    global frame
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Adding a short delay to simulate real-time capture
        cv2.waitKey(1)

# Start the frame capturing thread
capture_thread = threading.Thread(target=capture_frames)
capture_thread.start()

while True:
    if frame is None:
        continue

    # Make predictions
    results = model.predict(frame)

    # Process the results
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = box.conf[0].item() if box.conf is not None else 0
            cls = box.cls[0].item() if box.cls is not None else 0
            label = model.names[int(cls)]
            # Draw the bounding box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
            # Add the label
            cv2.putText(frame, f'{label} {conf:.2f}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Display the resulting frame
    cv2.imshow('Frame', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture
cap.release()
cv2.destroyAllWindows()

# import cv2
# from ultralytics import YOLO
# import matplotlib.pyplot as plt

# model = YOLO('/Users/aadityavoruganti/Desktop/Aadi/TerraVortex/VortexLens/best_screw.pt')

# cap = cv2.VideoCapture(0)

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)    

#     results = model.predict(frame_rgb)

#     for result in results:
#         for box in result.boxes:
#             x1, y1, x2, y2 = box.xyxy[0].tolist()
#             conf = box.conf[0].item() if box.conf is not None else 0
#             cls = box.cls[0].item() if box.cls is not None else 0
#             label = model.names[int(cls)]
#             cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
#             cv2.putText(frame, f'{label} {conf:.2f}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

#     cv2.imshow('Frame', frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()


# from flask import Flask, Response
# import cv2
# from ultralytics import YOLO

# app = Flask(__name__)

# # Load the model
# model = YOLO('/Users/aadityavoruganti/Desktop/Aadi/TerraVortex/VortexLens/best.pt')

# def generate_frames():
#     cap = cv2.VideoCapture(0)  # 0 is the default camera
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         # Convert the frame to RGB
#         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#         # Make predictions
#         results = model.predict(frame_rgb)

#         # Process the results
#         for result in results:
#             for box in result.boxes:
#                 x1, y1, x2, y2 = box.xyxy[0].tolist()
#                 conf = box.conf[0].item() if box.conf is not None else 0
#                 cls = box.cls[0].item() if box.cls is not None else 0
#                 label = model.names[int(cls)]
#                 # Draw the bounding box
#                 cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
#                 # Add the label
#                 cv2.putText(frame, f'{label} {conf:.2f}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

#         # Encode the frame in JPEG format
#         ret, buffer = cv2.imencode('.jpg', frame)
#         frame = buffer.tobytes()

#         # Yield the frame in byte format
#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

#     cap.release()

# @app.route('/video_feed')
# def video_feed():
#     return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# @app.route('/')
# def index():
#     return "<h1>Video Streaming</h1><img src='/video_feed'>"

# if __name__ == "__main__":
#     app.run(host='0.0.0.0', port=5001)
