import cv2
import numpy as np
from blob_detection import detect_screws_blobs

MIN_MATCH_COUNT = 10

def detect_and_match_features(ref_img, webcam_img):
    sift = cv2.SIFT_create(nfeatures=5000, contrastThreshold=0.04, edgeThreshold=10, sigma=1.6)
    kp_ref, des_ref = sift.detectAndCompute(ref_img, None)
    kp_frame, des_frame = sift.detectAndCompute(webcam_img, None)

    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=100)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des_ref, des_frame, k=2)
    
    good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]
    print(len(good_matches))

    return kp_ref, kp_frame, good_matches

def find_homography_and_transform(ref_img, webcam_img, kp_ref, kp_frame, good_matches, screw_locations):
    if len(good_matches) <= MIN_MATCH_COUNT:
        print("Not enough matches found")
        return False, None

    ref_pts = np.float32([kp_ref[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    frame_pts = np.float32([kp_frame[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(ref_pts, frame_pts, cv2.RANSAC, 3.0)
    if H is None or not (0.01 < np.linalg.det(H) < 100):
        print("Invalid homography")
        return False, None
    
    matches_mask = mask.ravel().tolist()
    screw_positions = np.float32(screw_locations).reshape(-1, 1, 2)
    transformed_positions = cv2.perspectiveTransform(screw_positions, H)
    draw_params = dict(matchColor=(0, 255, 0), singlePointColor=None, matchesMask=matches_mask, flags=2)
    img_matches = cv2.drawMatches(ref_img, kp_ref, webcam_img, kp_frame, good_matches, None, **draw_params)
    cv2.imshow("Matches", img_matches)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return True, transformed_positions


def draw_matches(webcam_frame, screw_positions):
    for pos in screw_positions:
        x, y = pos[0]
        cv2.circle(webcam_frame, (int(x), int(y)), 10, (0, 255, 0), -1)

    cv2.imshow("Mapped Screw Positions", webcam_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    reference_image = 'test6.jpeg'
    webcam_frame = 'webcam.jpeg'
    
    ref_img = cv2.imread(reference_image, cv2.IMREAD_GRAYSCALE)
    webcam_img = cv2.imread(webcam_frame, cv2.IMREAD_GRAYSCALE)
    screw_positions = detect_screws_blobs(reference_image)
    kp_ref, kp_frame, good_matches = detect_and_match_features(ref_img, webcam_img)
    matched, new_screw_positions = find_homography_and_transform(ref_img, webcam_img, kp_ref, kp_frame, good_matches, screw_positions)
    
    if matched:
        draw_matches(webcam_img, new_screw_positions)
    else:
        print("Perpectives are not similar enough to match.")