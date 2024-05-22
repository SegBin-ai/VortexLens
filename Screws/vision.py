import cv2
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Load pre-trained models (assumed to be locally available or download them)
# For action detection, you may use a model like YOLO or any suitable action detection model
action_model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # Example using YOLOv5

# Load GPT model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Function to generate commentary
def generate_commentary(actions):
    input_text = f"The worker is currently performing the following actions: {', '.join(set(actions))}."
    inputs = tokenizer.encode(input_text, return_tensors='pt')
    outputs = model.generate(inputs, max_length=100, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
    commentary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return commentary

# OpenCV for video capture
cap = cv2.VideoCapture('/Users/aadityavoruganti/Desktop/Aadi/TerraVortex/VortexLens/test.mp4')  # Use 'test.mp4' for video file

if not cap.isOpened():
    print("Error: Could not open video file.")
else:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Perform action detection on the frame
        results = action_model(frame)

        # Extract detected actions
        detected_actions = []
        for det in results.xyxy[0]:
            confidence = det[4].item()
            if confidence > 0.5:
                class_id = int(det[5].item())
                action_name = action_model.names[class_id]
                detected_actions.append(action_name)

        # Generate commentary based on detected actions
        if detected_actions:
            commentary = generate_commentary(detected_actions)
            print(commentary)  # Print or log the commentary

        # Display the frame (optional)
        cv2.imshow('Video Feed', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
