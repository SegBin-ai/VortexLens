import cv2
import numpy as np
from blob_detection import detect_screws_blobs, detect_using_locations, model_init
from Perpective_transform import detect_and_match_features_sift, find_homography_and_transform, draw_matches, detect_and_match_features_surf, detect_and_match_features_orb
import time
from multiprocessing.pool import ThreadPool
#from asift import affine_detect, init_feature, filter_matches

def preprocess_proposal(image):
    blur = cv2.medianBlur(image, 5)
    # blur = image
    # thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    return blur

def resize_with_aspect_ratio(image, max_size=1000):
    h, w = image.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / float(max(h, w))
        new_w, new_h = int(w * scale), int(h * scale)
        resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return resized_image
    
    return image

    
def test_with_image(reference_image, webcam_frame):
    model = model_init()
    index_params = dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=2)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    detect_img = cv2.imread(webcam_frame)
    
    ref_img = cv2.imread(reference_image, cv2.IMREAD_GRAYSCALE)
    webcam_img = cv2.imread(webcam_frame, cv2.IMREAD_GRAYSCALE)
    
    # Resize images while maintaining aspect ratio
    # ref_img = resize_with_aspect_ratio(ref_img)
    # webcam_img = resize_with_aspect_ratio(webcam_img)
    # detect_img = resize_with_aspect_ratio(detect_img)
    
    
    screw_positions = detect_screws_blobs(reference_image, model)
    orb = cv2.ORB_create(nfeatures=5000, WTA_K=3, scoreType=cv2.ORB_HARRIS_SCORE, patchSize=31, fastThreshold=20)
    #beb = cv2.xfeatures2d.BEBLID_create(0.75)
    kp_ref, des_ref = orb.detectAndCompute(preprocess_proposal(ref_img), None)
    #_, des_ref = beb.compute(preprocess_proposal(ref_img), kp_ref)
    
    kp_ref, kp_frame, good_matches = detect_and_match_features_orb(webcam_img, flann, kp_ref, des_ref, orb)
    matched, new_screw_positions = find_homography_and_transform(ref_img, webcam_img, kp_ref, kp_frame, good_matches, screw_positions)
    
    if matched:
        draw_matches(webcam_img, new_screw_positions)
        res_img = detect_using_locations(detect_img, new_screw_positions, model)
        cv2.imshow("Detected screws", res_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Perspectives are not similar enough to match.")
        
def main(reference_image_path, video_path=0, fps_limit=10, output_file= "C:\\Users\\Aaditya Voruganti\\Desktop\\VortexLens\\screw_exploration\\Blob_detection\\matches_count.txt"):
    with open(output_file, "w") as f:
        f.write("")
    model = model_init()
    index_params = dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=2)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    ref_img = cv2.imread(reference_image_path, cv2.IMREAD_GRAYSCALE)
    
    # ref_img = resize_with_aspect_ratio(ref_img)
    
    screw_positions = detect_screws_blobs(reference_image_path, model)
    orb = cv2.ORB_create(nfeatures=5000, WTA_K=3, scoreType=cv2.ORB_HARRIS_SCORE, patchSize=31, fastThreshold=20)

    kp_ref, des_ref = orb.detectAndCompute(preprocess_proposal(ref_img), None)

    cap = cv2.VideoCapture(video_path)
    prev_time = 0
    frame_count = 1
    display_end_time = None
    frame_count_interval = 30
    display_duration = 4
    match_count = 0

    with open(output_file, "w") as f:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            current_time = time.time()
            if display_end_time is not None and current_time < display_end_time:
                continue
            
            if frame_count % frame_count_interval == 0:
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # frame_gray = resize_with_aspect_ratio(frame_gray)
                # frame = resize_with_aspect_ratio(frame)
                
                kp_ref, kp_frame, good_matches = detect_and_match_features_orb(frame_gray, flann, kp_ref, des_ref, orb)
                f.write(f"{len(good_matches)}\n")
                f.flush()
                
                matched, transformed_positions = find_homography_and_transform(ref_img, frame_gray, kp_ref, kp_frame, good_matches, screw_positions)
                if matched:
                    cv2.imwrite("C:\\Users\\Aaditya Voruganti\\Desktop\\VortexLens\\screw_exploration\\Blob_detection\\test_images\\match" + str(match_count) +".jpg", frame)
                    match_count = match_count + 1
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
    #main("C:\\Users\\Aaditya Voruganti\\Desktop\\VortexLens\\screw_exploration\\Blob_detection\\test_images\\test1.jpg")
    
    test_with_image("C:\\Users\\Aaditya Voruganti\\Desktop\\VortexLens\\screw_exploration\\Blob_detection\\test_images\\test1.jpg", "C:\\Users\\Aaditya Voruganti\\Desktop\\VortexLens\\screw_exploration\\Blob_detection\\test_images\\match.jpg")
    # test_with_image("test_images/mvp3.jpg", "test_images/mvp2.jpg")
    # test_with_image("test_images/screw_reference.png", "test_images/screw_test4.png")

