import cv2
import numpy as np
from blob_detection import detect_screws_blobs, detect_using_locations, model_init, yolo_model_init
import time
import kornia as K
import kornia.feature as KF
import numpy as np
import torch

rectangles = []
drawing = False
ix, iy = -1, -1

def draw_rectangle(event, x, y, flags, param):
    global ix, iy, drawing, rectangles

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            img_copy = img.copy()
            cv2.rectangle(img_copy, (ix, iy), (x, y), (0, 255, 0), 2)
            cv2.imshow("Mark Screws", img_copy)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        rectangles.append((ix, iy, x, y))
        cv2.rectangle(img, (ix, iy), (x, y), (0, 255, 0), 2)
        cv2.imshow("Mark Screws", img)


def resize_image(image, max_size=800):
    height, width = image.shape[:2]
    
    if height > width:
        scaling_factor = max_size / height
    else:
        scaling_factor = max_size / width

    new_width = int(width * scaling_factor)
    new_height = int(height * scaling_factor)
    
    return new_height, new_width


def load_image_from_array(image_array):
    image_tensor = K.image_to_tensor(image_array, keepdim=False).float() / 255.0
    if len(image_tensor.shape) == 3:
        image_tensor = image_tensor.unsqueeze(0)
    
    return image_tensor

    
def test_with_image(reference_image, webcam_frame):
    model_yolo = yolo_model_init()
    model = model_init()
    ref_img = cv2.imread(reference_image)
    # new_height1, new_width1 = resize_image(ref_img, 900)
    # ref_img = cv2.resize(ref_img, (new_width1, new_height1))
    webcam_img = cv2.imread(webcam_frame)
    screw_positions = detect_screws_blobs(ref_img, model_yolo)
    img1 = load_image_from_array(ref_img)
    img2 = load_image_from_array(webcam_img)
    
    new_height1, new_width1 = resize_image(K.tensor_to_image(img1[0]), 640)
    new_height, new_width = resize_image(K.tensor_to_image(img2[0]), 640)
    img1 = K.geometry.resize(img1, (new_height1, new_width1), antialias=True)
    img2 = K.geometry.resize(img2, (new_height, new_width), antialias=True)

    matcher = KF.LoFTR(pretrained="outdoor")

    input_dict = {
        "image0": K.color.rgb_to_grayscale(img1),
        "image1": K.color.rgb_to_grayscale(img2),
    }

    with torch.inference_mode():
        correspondences = matcher(input_dict)

    print(f"Number of correspondences: {len(correspondences['keypoints0'])}")

    mkpts0 = correspondences["keypoints0"].cpu().numpy()
    mkpts1 = correspondences["keypoints1"].cpu().numpy()

    Fm, inliers = cv2.findFundamentalMat(mkpts0, mkpts1, cv2.USAC_MAGSAC, 0.5, 0.999, 100000)
    inliers = inliers.ravel().astype(bool)

    N = 60
    good_matches = inliers.nonzero()[0]
    if len(good_matches) > N:
        np.random.shuffle(good_matches)
        good_matches = good_matches[:N]

    filtered_mkpts0 = mkpts0[good_matches]
    filtered_mkpts1 = mkpts1[good_matches]

    print(f"Number of filtered matches: {len(filtered_mkpts0)}")

    H, mask = cv2.findHomography(filtered_mkpts0, filtered_mkpts1, cv2.RANSAC, 5.0)


    screw_positions = np.float32(screw_positions).reshape(-1, 1, 2)
    transformed_screw_positions = cv2.perspectiveTransform(screw_positions, H)

    detect_img = cv2.imread(webcam_frame)
    new_height, new_width = resize_image(detect_img, 640)
    detect_img = cv2.resize(detect_img, (new_width, new_height))
    res_img = detect_using_locations(detect_img, transformed_screw_positions, model_yolo)
    cv2.imshow("Detected screws", res_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def draw_status(frame, screw_states):
    font_scale = 0.5
    line_spacing = 20
    
    for i, state in enumerate(screw_states):
        color = (0, 255, 0) if state else (0, 0, 255)
        status_text = f"Screw {i+1}: {'Yes' if state else 'No'}"
        y_position = 20 + (i * line_spacing)
        cv2.putText(frame, status_text, (10, y_position), cv2.FONT_HERSHEY_SIMPLEX, 
                    font_scale, color, 1, cv2.LINE_AA)

def mark_screws(ref_img):
    global img
    img = ref_img.copy()
    
    # Create a window and set the mouse callback to draw rectangles
    cv2.namedWindow("Mark Screws")
    cv2.setMouseCallback("Mark Screws", draw_rectangle)

    while True:
        cv2.imshow("Mark Screws", img)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('d'):  # Press 'd' to finish marking
            break
    
    # Calculate screw positions as centers of rectangles
    screw_positions = []
    for (ix, iy, x, y) in rectangles:
        center_x = (ix + x) // 2
        center_y = (iy + y) // 2
        screw_positions.append([center_x, center_y])
    
    return screw_positions


def main(reference_image_path, video_path=0, fps_limit=30, output_file="matches_count.txt", output_video_file="output_video.mp4"):
    with open(output_file, "w") as f:
        f.write("")
        
    model_yolo = yolo_model_init()
    model = model_init()
    ref_img = cv2.imread(reference_image_path)
    # Mark screws automatically
    screw_positions = detect_screws_blobs(ref_img, model)

    # Mark screws manually (testing purposes)
    #screw_positions = mark_screws(ref_img)
    
    screw_positions = np.float32(screw_positions).reshape(-1, 1, 2)
    # np.save("screw_positions.npy", screw_positions)
    # screw_positions = np.load("screw_positions.npy")
    screw_states = [False] * len(screw_positions)
    
    img_1 = load_image_from_array(ref_img)
    new_height1, new_width1 = resize_image(K.tensor_to_image(img_1[0]), 640)
    img1 = K.geometry.resize(img_1, (new_height1, new_width1), antialias=True)
    matcher = KF.LoFTR(pretrained="outdoor")

    cap = cv2.VideoCapture(video_path)
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_file, fourcc, fps_limit, (640, 426))

    frame_count = 0
    display_end_time = None
    frame_count_interval = 200
    display_duration = 4
    # frame_count_intervals = [330, 630, 870, 1320, 1650]
    frame_count_intervals = [180, 570, 840]
    
    #commented code was for demo video purposes
    screws_placed = [0, 2, 3]
    index = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        new_height, new_width = resize_image(frame, 640)
        frame = cv2.resize(frame, (new_width, new_height))
        
        current_time = time.time()
        if display_end_time is not None and current_time < display_end_time:
            for _ in range(int(fps_limit * display_duration)):
                out.write(frame)
            continue
        
        # if frame_count % frame_count_interval == 0:
        if frame_count in frame_count_intervals:
            #The commented code was for demo video purposes
            # screw_states[screws_placed[index]] = True
            # index += 1
            frame_k = load_image_from_array(frame)
            new_height2, new_width2 = resize_image(K.tensor_to_image(frame_k[0]), 640)
            frame_k = K.geometry.resize(frame_k, (new_height2, new_width2), antialias=True)
            input_dict = {
                "image0": K.color.rgb_to_grayscale(img1),
                "image1": K.color.rgb_to_grayscale(frame_k),
            }

            with torch.inference_mode():
                correspondences = matcher(input_dict)

            print(f"Number of correspondences: {len(correspondences['keypoints0'])}")

            mkpts0 = correspondences["keypoints0"].cpu().numpy()
            mkpts1 = correspondences["keypoints1"].cpu().numpy()

            Fm, inliers = cv2.findFundamentalMat(mkpts0, mkpts1, cv2.USAC_MAGSAC, 0.5, 0.999, 100000)
            inliers = inliers.ravel().astype(bool)

            N = 60 
            good_matches = inliers.nonzero()[0]
            if len(good_matches) > N:
                np.random.shuffle(good_matches)
                good_matches = good_matches[:N]

            filtered_mkpts0 = mkpts0[good_matches]
            filtered_mkpts1 = mkpts1[good_matches]

            print(f"Number of filtered matches: {len(filtered_mkpts0)}")

            H, mask = cv2.findHomography(filtered_mkpts0, filtered_mkpts1, cv2.RANSAC, 5.0)

            transformed_screw_positions = cv2.perspectiveTransform(screw_positions, H)
            
            #res_img = detect_using_locations_demo(frame, transformed_screw_positions, index, screws_placed)
            res_img, temp_screw_states = detect_using_locations(frame, transformed_screw_positions, model_yolo, screw_states)
            screw_states = temp_screw_states
            draw_status(res_img, screw_states)
            cv2.imshow("Screw Verification", res_img)
            display_end_time = current_time + display_duration
            for _ in range(int(fps_limit * display_duration)):
                out.write(res_img)
        else:
            draw_status(frame, screw_states)
            cv2.imshow("Screw Verification", frame)
            out.write(frame)
            
        frame_count += 1
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    

if __name__ == '__main__':
    main("Test_new/match2.jpg", video_path="test_video.mp4")
    # test_num = 27
    # for i in range(0, test_num):
    #     test_with_image("Test_new/match2.jpg", "Test_new1/match" + str(i) + ".jpg")
    # test_with_image("Test_new1/match25.jpg", "Test_new1/match26.jpg")
    # test_with_image("test_images/mvp3.jpg", "test_images/mvp2.jpg")
    # test_with_image("test_images/screw_reference.png", "test_images/screw_test4.png")