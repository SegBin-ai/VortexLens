import cv2
from blob_detection import detect_screws_blobs, detect_using_locations, model_init
from Perpective_transform import detect_and_match_features, find_homography_and_transform, draw_matches
    
    
def test_with_image(reference_image, webcam_frame):
    model = model_init()
    detect_img = cv2.imread(webcam_frame)
    ref_img = cv2.imread(reference_image, cv2.IMREAD_GRAYSCALE)
    webcam_img = cv2.imread(webcam_frame, cv2.IMREAD_GRAYSCALE)
    screw_positions = detect_screws_blobs(reference_image, model)
    kp_ref, kp_frame, good_matches = detect_and_match_features(ref_img, webcam_img)
    matched, new_screw_positions = find_homography_and_transform(ref_img, webcam_img, kp_ref, kp_frame, good_matches, screw_positions)
    
    if matched:
        draw_matches(webcam_img, new_screw_positions)
        res_img = detect_using_locations(detect_img, new_screw_positions, model)
        cv2.imshow("Detected screws", res_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Perpectives are not similar enough to match.")
        
    
    
def main(reference_image_path, video_path=0):
    model = model_init()
    ref_img = cv2.imread(reference_image_path, cv2.IMREAD_GRAYSCALE)
    screw_positions = detect_screws_blobs(reference_image_path, model)

    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp_ref, kp_frame, good_matches = detect_and_match_features(ref_img, frame_gray)
        matched, transformed_positions = find_homography_and_transform(ref_img, frame_gray, kp_ref, kp_frame, good_matches, screw_positions)
        
        if matched:
            frame_with_detections = detect_using_locations(frame, transformed_positions, model)
            cv2.imshow("Screw Verification", frame_with_detections)
        else:
            cv2.imshow("Screw Verification", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # main("test6.jpeg", "webcam5.mp4")
    test_with_image("test_images/test6.jpeg", "test_images/webcam4.jpeg")