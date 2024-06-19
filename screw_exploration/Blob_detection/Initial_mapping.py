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

def detect_screws_blobs(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
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

    screw_positions = []

    for keypoint in keypoints:
        x, y = int(keypoint.pt[0]), int(keypoint.pt[1])
        diameter = int(keypoint.size)
        radius = diameter // 2
        top_left_x = max(0, x - radius - 5)
        top_left_y = max(0, y - radius - 5)
        bottom_right_x = min(image.shape[1], x + radius + 5)
        bottom_right_y = min(image.shape[0], y + radius + 5)
        
        region = image[top_left_y:bottom_right_y, top_left_x:bottom_right_x]
        proposal = preprocess_proposal(region)
        prediction = model.predict(proposal)
        score = prediction[0][0]
        
        if score < MIN_CONFIDENCE:
            screw_positions.append((x, y))
            color = (0, 255, 0)
            cv2.rectangle(image, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), color, 2)
            cv2.putText(image, f'Score: {score:.2f}', (top_left_x, top_left_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return screw_positions, image

def detect_and_match_features(reference_image, webcam_frame):
    gray_ref = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.cvtColor(webcam_frame, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    kp_ref, des_ref = sift.detectAndCompute(gray_ref, None)
    kp_frame, des_frame = sift.detectAndCompute(gray_frame, None)

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    matches = flann.knnMatch(des_ref, des_frame, k=2)
    good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]

    return kp_ref, kp_frame, good_matches

def find_homography_and_transform(reference_image, webcam_frame, kp_ref, kp_frame, good_matches):
    ref_pts = np.float32([kp_ref[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    frame_pts = np.float32([kp_frame[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(ref_pts, frame_pts, cv2.RANSAC, 5.0)
    return H, mask

def map_screw_locations(H, screw_positions):
    screw_positions = np.float32(screw_positions).reshape(-1, 1, 2)
    transformed_positions = cv2.perspectiveTransform(screw_positions, H)
    return transformed_positions


reference_image = cv2.imread('test6.jpeg')
webcam_frame = cv2.imread('webcam2.jpeg')


screw_positions, detected_image = detect_screws_blobs(reference_image)

kp_ref, kp_frame, good_matches = detect_and_match_features(reference_image, webcam_frame)

H, mask = find_homography_and_transform(reference_image, webcam_frame, kp_ref, kp_frame, good_matches)

mapped_screw_positions = map_screw_locations(H, screw_positions)

for pos in mapped_screw_positions:
    x, y = pos[0]
    cv2.circle(webcam_frame, (int(x), int(y)), 10, (0, 255, 0), -1)


cv2.imshow("Mapped Screw Positions", webcam_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()