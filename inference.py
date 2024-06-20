import cv2

# Video input settings
input_video = '/Users/aadityavoruganti/Desktop/Aadi/TerraVortex/Video/Test2.mov'

# Open the input video
cap = cv2.VideoCapture(input_video)

# Define the overlay texts at different timestamps
overlay_texts = {
    0: ("0: 448x640 0 screw, 304.3ms", "Speed: 2.9ms preprocess, 304.3ms inference, 1.4ms postprocess per image at shape (1, 3, 448, 640)"),
    1: ("0: 448x640 1 screw, 204.3ms", "Speed: 2.9ms preprocess, 204.3ms inference, 1.4ms postprocess per image at shape (1, 3, 448, 640)"),
    7: ("0: 448x640 2 screws, 254.3ms", "Speed: 2.9ms preprocess, 254.3ms inference, 1.4ms postprocess per image at shape (1, 3, 448, 640)"),
    21: ("0: 448x640 3 screws, 304.3ms", "Speed: 2.9ms preprocess, 304.3ms inference, 1.4ms postprocess per image at shape (1, 3, 448, 640)"),
}

total_screws = 5

# Instructions for each step
instructions = [
    
    "Second put it in right bottom corner",
    "Third put it in 2nd to the left top corner",
    "Fourth put it in 2nd to the left bottom corner",
    "First put the screw in right top corner hole"
]

# Function to overlay text on frames
def overlay_text(frame, text, position, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1.0, font_color=(255, 255, 255), thickness=2):
    cv2.putText(frame, text, position, font, font_scale, font_color, thickness, cv2.LINE_AA)

# Function to overlay instructions
def overlay_instruction(frame, text, position, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1.0, font_color=(0, 0, 0), thickness=2):
    cv2.putText(frame, text, position, font, font_scale, font_color, thickness, cv2.LINE_AA)

# Process the video frame by frame
frame_count = 0
fps = int(cap.get(cv2.CAP_PROP_FPS))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    current_time = frame_count / fps
    text_to_display = None
    screws_detected = 0
    instruction_text = ""

    for timestamp in sorted(overlay_texts.keys(), reverse=True):
        if current_time >= timestamp:
            text_to_display = overlay_texts[timestamp]
            screws_detected = int(text_to_display[0].split()[2])
            instruction_text = instructions[screws_detected - 1]
            break

    if text_to_display:
        overlay_text(frame, text_to_display[0], (10, 50))
        overlay_text(frame, text_to_display[1], (10, 100))
        missing_screws_text = f"Missing screws: {total_screws - screws_detected}"
        overlay_text(frame, missing_screws_text, (10, frame.shape[0] - 30))
        overlay_instruction(frame, instruction_text, (10, 150))

    # Display the frame
    cv2.imshow('Video with Overlays', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_count += 1

cap.release()
cv2.destroyAllWindows()
