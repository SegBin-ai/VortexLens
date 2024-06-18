#Initial attemp at object tracking, bad perfromace, very very slow and not accurate

import cv2
import numpy as np
import h5py
from keras.models import model_from_json

IMAGE_SIZE = (71, 71)
MIN_CONFIDENCE = 0.5


model_path = 'screw_head_detector-2.h5'
with h5py.File(model_path, 'r') as f:
    model_config = f.attrs.get('model_config')
model = model_from_json(model_config)
model.load_weights(model_path)


def preprocess_proposal(region, image_size=IMAGE_SIZE):
    proposal = cv2.resize(region, image_size)
    proposal = proposal.astype('float32') / 255.0
    proposal = np.expand_dims(proposal, axis=0)
    return proposal

def detect_screws_blobs(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    
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
    
    detections = []

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
        
        if score < MIN_CONFIDENCE:
            bbox = (top_left_x, top_left_y, bottom_right_x - top_left_x, bottom_right_y - top_left_y)
            detections.append((bbox, score))
    
    return detections

def normalize_bbox(bbox, img_shape):
    x, y, width, height = bbox
    img_height, img_width = img_shape[:2]
    return (x / img_width, y / img_height, width / img_width, height / img_height)


def denormalize_bbox(bbox, frame_shape):
    x, y, width, height = bbox
    frame_height, frame_width = frame_shape[:2]
    return (int(x * frame_width), int(y * frame_height), int(width * frame_width), int(height * frame_height))


def clamp_bbox(bbox, frame_shape):
    x, y, width, height = bbox
    frame_height, frame_width = frame_shape[:2]
    x = max(0, min(x, frame_width - 1))
    y = max(0, min(y, frame_height - 1))
    width = max(1, min(width, frame_width - x))
    height = max(1, min(height, frame_height - y))
    return (x, y, width, height)


def initialize_trackers(detections, img):
    trackers = []
    normalized_bboxes = [normalize_bbox(bbox, img.shape) for bbox, _ in detections]
    for bbox in normalized_bboxes:
        tracker = cv2.TrackerCSRT_create()
        denorm_bbox = denormalize_bbox(bbox, img.shape)
        tracker.init(img, denorm_bbox)
        trackers.append(tracker)
    return trackers, normalized_bboxes

def track_screws_in_webcam(trackers, normalized_bboxes):
    video_capture = cv2.VideoCapture(0)

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        
        for tracker, normalized_bbox in zip(trackers, normalized_bboxes):
            denorm_bbox = denormalize_bbox(normalized_bbox, frame.shape)
            denorm_bbox = clamp_bbox(denorm_bbox, frame.shape)
            tracker.init(frame, denorm_bbox)
            success, bbox = tracker.update(frame)
            if success:
                x, y, w, h = bbox
                if 0 <= x <= frame.shape[1] - w and 0 <= y <= frame.shape[0] - h:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow('Tracked Screws', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    video_capture.release()
    cv2.destroyAllWindows()

reference_image_path = 'test6.jpeg'
reference_image = cv2.imread(reference_image_path)
screw_heads_ref = detect_screws_blobs(reference_image)
trackers, normalized_bboxes = initialize_trackers(screw_heads_ref, reference_image)
track_screws_in_webcam(trackers, normalized_bboxes)