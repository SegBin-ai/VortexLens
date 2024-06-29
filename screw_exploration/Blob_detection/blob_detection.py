import cv2
import numpy as np
import h5py
from keras.models import model_from_json


IMAGE_SIZE = (71, 71)
MIN_CONFIDENCE = 0.18

model_path = 'C:\\Users\\Aaditya Voruganti\\Desktop\\VortexLens\\screw_exploration\\Blob_detection\\screw_head_detector-2.h5'

def model_init():
    
    with h5py.File(model_path, 'r') as f:
        model_config = f.attrs.get('model_config')

    model = model_from_json(model_config)
    model.load_weights(model_path)
    return model


#required pre processing for the model
def preprocess_proposal(region, image_size=IMAGE_SIZE):
    proposal = cv2.resize(region, image_size)
    proposal = proposal.astype('float32') / 255.0
    proposal = np.expand_dims(proposal, axis=0)
    return proposal


def detect_screws_blobs(image_path, model):
    """
    Detect screw heads in the image using blob detection and classify using the model.
    """
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    
    #blob detection parameters
    #These parameters need some tuning to detect the right circles that could be screw heads

    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = 15
    params.maxArea = 3000
    params.filterByCircularity = True
    params.minCircularity = 0.60
    params.filterByConvexity = True
    params.minConvexity = 0.7
    params.filterByInertia = True
    params.minInertiaRatio = 0.01

    detector = cv2.SimpleBlobDetector_create(params)

    keypoints = detector.detect(gray)
    
    screw_locations = []

    for keypoint in keypoints:
        x, y = int(keypoint.pt[0]), int(keypoint.pt[1])
        diameter = int(keypoint.size)
        radius = diameter // 2
        top_left_x = max(0, x - radius - 5)
        top_left_y = max(0, y - radius - 5)
        bottom_right_x = min(img.shape[1], x + radius + 5)
        bottom_right_y = min(img.shape[0], y + radius + 5)
        
        region = img[top_left_y:bottom_right_y, top_left_x:bottom_right_x]

        proposal = preprocess_proposal(region)
        prediction = model.predict(proposal)
        score = prediction[0][0]
        
        
        #The model has been trained so that a score closer to 0 means a screw head is present

        if score < MIN_CONFIDENCE:
            screw_locations.append((x, y))
            color = (0, 255, 0)
            cv2.rectangle(img, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), color, 2)
            cv2.putText(img, f'Score: {score:.2f}', (top_left_x, top_left_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    #saves the image with the detected screws
    cv2.imwrite('C:\\Users\\Aaditya Voruganti\\Desktop\\VortexLens\\screw_exploration\\Blob_detection\\results\\detected_screws_blobs_model_2.jpg', img)
    return screw_locations



def detect_using_locations(web_frame, screw_locations, model):
    for pos in screw_locations:
        x, y = int(pos[0][0]), int(pos[0][1])
        top_left_x = max(0, x - 7)
        top_left_y = max(0, y - 7)
        bottom_right_x = min(web_frame.shape[1], x + 7)
        bottom_right_y = min(web_frame.shape[0], y + 7)
        
        
        region = web_frame[top_left_y:bottom_right_y, top_left_x:bottom_right_x]
        if region.size == 0:
            continue
        
        proposal = preprocess_proposal(region)
        prediction = model.predict(proposal)
        score = prediction[0][0]
        
        if score < MIN_CONFIDENCE:
            color = (0, 255, 0)
            cv2.rectangle(web_frame, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), color, 2)
            cv2.putText(web_frame, f'Score: {score:.2f}', (top_left_x, top_left_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
        else:
            color = (0, 0, 255)
            cv2.rectangle(web_frame, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), color, 2)
            cv2.putText(web_frame, f'Score: {score:.2f}', (top_left_x, top_left_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            
    return web_frame