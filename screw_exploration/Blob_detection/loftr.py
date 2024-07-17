import cv2
import kornia as K
import kornia.feature as KF
import numpy as np
import torch
from kornia_moons.viz import draw_LAF_matches
from blob_detection import detect_screws_blobs, detect_using_locations, yolo_model_init, model_init
import matplotlib.pyplot as plt

def resize_image(image, max_size=800):
    height, width = image.shape[:2]
    
    # Determine the scaling factor based on the maximum dimension
    if height > width:
        scaling_factor = max_size / height
    else:
        scaling_factor = max_size / width
    
    # Calculate new dimensions
    new_width = int(width * scaling_factor)
    new_height = int(height * scaling_factor)
    
    # Resize the image
    return new_height, new_width

# Load images
fname1 = "Test_new/match2.jpg"
fname2 = "Test_new1/match10.jpg"

model = model_init()
model_yolo = yolo_model_init()
screw_positions = detect_screws_blobs(fname1, model)

img1 = K.io.load_image(fname1, K.io.ImageLoadType.RGB32)[None, ...]
img2 = K.io.load_image(fname2, K.io.ImageLoadType.RGB32)[None, ...]


new_height, new_width = resize_image(K.tensor_to_image(img2[0]), 800)
img2 = K.geometry.resize(img2, (new_height, new_width), antialias=True)

# Check if images are loaded correctly
print(f"Image 1 shape: {img1.shape}")
print(f"Image 2 shape: {img2.shape}")

# Initialize LoFTR matcher
matcher = KF.LoFTR(pretrained="outdoor")

# Prepare input dict
input_dict = {
    "image0": K.color.rgb_to_grayscale(img1),  # LoFTR works on grayscale images only
    "image1": K.color.rgb_to_grayscale(img2),
}

# Perform matching
with torch.inference_mode():
    correspondences = matcher(input_dict)

# Check if correspondences are found
print(f"Number of correspondences: {len(correspondences['keypoints0'])}")

# Extract matched keypoints
mkpts0 = correspondences["keypoints0"].cpu().numpy()
mkpts1 = correspondences["keypoints1"].cpu().numpy()

# Find fundamental matrix to get inliers
Fm, inliers = cv2.findFundamentalMat(mkpts0, mkpts1, cv2.USAC_MAGSAC, 0.5, 0.999, 100000)
inliers = inliers.ravel().astype(bool)

# Filter matches to get only the top N inliers
N = 60  # Number of top matches to keep
good_matches = inliers.nonzero()[0]
if len(good_matches) > N:
    np.random.shuffle(good_matches)
    good_matches = good_matches[:N]

filtered_mkpts0 = mkpts0[good_matches]
filtered_mkpts1 = mkpts1[good_matches]

# Check the number of filtered matches
print(f"Number of filtered matches: {len(filtered_mkpts0)}")

# Compute homography matrix using only the filtered keypoints
H, mask = cv2.findHomography(filtered_mkpts0, filtered_mkpts1, cv2.RANSAC, 5.0)

# Convert images to numpy for OpenCV
img1_np = K.tensor_to_image(img1)
img2_np = K.tensor_to_image(img2)

# Transform screw positions using the homography matrix
screw_positions = np.float32(screw_positions).reshape(-1, 1, 2)
transformed_screw_positions = cv2.perspectiveTransform(screw_positions, H)

# Draw transformed screw positions on the second image
# for pos in transformed_screw_positions:
#     x, y = pos[0]
#     cv2.circle(img2_np, (int(x), int(y)), 5, (0, 0, 255), -1)
detect_img = cv2.imread(fname2)
new_height, new_width = resize_image(detect_img, 800)
detect_img = cv2.resize(detect_img, (new_width, new_height))
res_img = detect_using_locations(detect_img, transformed_screw_positions, model_yolo)
cv2.imshow("Detected screws", res_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

fig, ax = plt.subplots(1, 1, figsize=(15, 10))
draw_LAF_matches(
    KF.laf_from_center_scale_ori(
        torch.from_numpy(filtered_mkpts0).view(1, -1, 2),
        torch.ones(filtered_mkpts0.shape[0]).view(1, -1, 1, 1),
        torch.ones(filtered_mkpts0.shape[0]).view(1, -1, 1),
    ),
    KF.laf_from_center_scale_ori(
        torch.from_numpy(filtered_mkpts1).view(1, -1, 2),
        torch.ones(filtered_mkpts1.shape[0]).view(1, -1, 1, 1),
        torch.ones(filtered_mkpts1.shape[0]).view(1, -1, 1),
    ),
    torch.arange(filtered_mkpts0.shape[0]).view(-1, 1).repeat(1, 2),
    K.tensor_to_image(img1),
    K.tensor_to_image(img2),
    np.ones(filtered_mkpts0.shape[0], dtype=bool),  # All filtered matches are inliers
    draw_dict={"inlier_color": (0.2, 1, 0.2), "tentative_color": None, "feature_color": (0.2, 0.5, 1), "vertical": False},
    ax=ax
)
plt.show()