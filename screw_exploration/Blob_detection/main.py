import cv2
from blob_detection import detect_screws_blobs, detect_using_locations, model_init
from Perpective_transform import detect_and_match_features_sift, find_homography_and_transform, draw_matches, detect_and_match_features_surf, detect_and_match_features_orb
import time

    
def test_with_image(reference_image, webcam_frame):
    model = model_init()
    index_params = dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=2)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    detect_img = cv2.imread(webcam_frame)
    ref_img = cv2.imread(reference_image, cv2.IMREAD_GRAYSCALE)
    webcam_img = cv2.imread(webcam_frame, cv2.IMREAD_GRAYSCALE)
    screw_positions = detect_screws_blobs(reference_image, model)
    orb = cv2.ORB_create(nfeatures=5000, scaleFactor=1.2, nlevels=8, edgeThreshold=31, firstLevel=0, WTA_K=2, scoreType=cv2.ORB_HARRIS_SCORE, patchSize=31, fastThreshold=20)
    kp_ref, des_ref = orb.detectAndCompute(ref_img, None)
    kp_ref, kp_frame, good_matches = detect_and_match_features_orb(webcam_img, flann, kp_ref, des_ref, orb)
    # kp_ref, kp_frame, good_matches = detect_and_match_features_sift(ref_img, webcam_img)
    matched, new_screw_positions = find_homography_and_transform(ref_img, webcam_img, kp_ref, kp_frame, good_matches, screw_positions)
    
    if matched:
        draw_matches(webcam_img, new_screw_positions)
        res_img = detect_using_locations(detect_img, new_screw_positions, model)
        cv2.imshow("Detected screws", res_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Perpectives are not similar enough to match.")
        
    
    
def main(reference_image_path, video_path=0, fps_limit=10):
    model = model_init()
    index_params = dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=2)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    ref_img = cv2.imread(reference_image_path, cv2.IMREAD_GRAYSCALE)
    screw_positions = detect_screws_blobs(reference_image_path, model)
    orb = cv2.ORB_create(nfeatures=5000, scaleFactor=1.2, nlevels=8, edgeThreshold=31, firstLevel=0, WTA_K=2, scoreType=cv2.ORB_HARRIS_SCORE, patchSize=31, fastThreshold=20)
    kp_ref, des_ref = orb.detectAndCompute(ref_img, None)

    cap = cv2.VideoCapture(video_path)
    prev_time = 0
    frame_count = 1
    display_end_time = None
    frame_count_interval = 30
    display_duration = 4

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        current_time = time.time()
        if display_end_time is not None and current_time < display_end_time:
            continue
        
        if frame_count % frame_count_interval == 0:
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            kp_ref, kp_frame, good_matches = detect_and_match_features_orb(frame_gray, flann, kp_ref, des_ref, orb)
            matched, transformed_positions = find_homography_and_transform(ref_img, frame_gray, kp_ref, kp_frame, good_matches, screw_positions)
            if matched:
                cv2.imwrite("test_images/webcam_test.jpeg", frame)
                frame_with_detections = detect_using_locations(frame, transformed_positions, model)
                cv2.imshow("Screw Verification", frame_with_detections)
                display_end_time = current_time + display_duration
            else:
                cv2.imshow("Screw Verification", frame)
                
        else:
            cv2.imshow("Screw Verification", frame)
            
        frame_count += 1
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    #main("test_images/calc.jpeg")
    test_with_image('C:\\Users\\Aaditya Voruganti\\Desktop\\VortexLens\\screw_exploration\\Blob_detection\\test_images\\Screw_1.jpg', 'C:\\Users\\Aaditya Voruganti\\Desktop\\VortexLens\\screw_exploration\\Blob_detection\\test_images\\Screw_1_test.jpg')
    # test_with_image("test_images/test6.jpeg", "test_images/webcam4.jpeg")
    # test_with_image("test_images/screw_reference.png", "test_images/screw_test4.png")