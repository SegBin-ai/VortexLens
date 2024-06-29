
import cv2
import numpy as np
import torch
from scipy.spatial import distance_matrix

MIN_MATCH_COUNT = 20

def get_fginn_indexes(dm, km, th=0.8):
    vals, idxs_in_2 = torch.min(dm, dim=1)
    mask1 = km <= 10.0
    mask2 = mask1[idxs_in_2, :]
    dm[mask2] = 100000
    vals_2nd, idxs_in_2_2nd = torch.min(dm, dim=1)
    ratio = vals / vals_2nd
    mask = ratio <= th
    idxs_in_1 = torch.arange(0, idxs_in_2.size(0), device=dm.device)[mask]
    idxs_in_2 = idxs_in_2[mask]
    ml = torch.cat([idxs_in_1.view(-1, 1).cpu(), idxs_in_2.cpu().view(-1, 1)], dim=1)
    return ml

def match_fginn(desc1, desc2, kps1, kps2, device='cpu'):
    xy1 = np.concatenate([np.array(p.pt).reshape(1, 2) for p in kps1], axis=0)
    xy2 = np.concatenate([np.array(p.pt).reshape(1, 2) for p in kps2], axis=0)

    desc1_tensor = torch.from_numpy(desc1.astype(np.float32)).to(device)
    desc2_tensor = torch.from_numpy(desc2.astype(np.float32)).to(device)
    xy1_tensor = torch.from_numpy(xy1.astype(np.float32)).to(device)
    xy2_tensor = torch.from_numpy(xy2.astype(np.float32)).to(device)

    dm = distance_matrix(desc1_tensor.cpu().numpy(), desc2_tensor.cpu().numpy())
    km = distance_matrix(xy2_tensor.cpu().numpy(), xy2_tensor.cpu().numpy())

    dm_tensor = torch.from_numpy(dm).to(device)
    km_tensor = torch.from_numpy(km).to(device)

    return get_fginn_indexes(dm_tensor, km_tensor, 0.8)

def match_features(desc_ref, desc_frame, kp_ref, kp_frame, is_sift=False, use_fginn=False, device='cpu'):
    if use_fginn:
        matches = match_fginn(desc_ref, desc_frame, kp_ref, kp_frame, device=device)
        good_matches = [cv2.DMatch(_queryIdx=int(m[0]), _trainIdx=int(m[1]), _imgIdx=0, _distance=0) for m in matches]
    else:
        if is_sift:
            bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        else:
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        
        matches = bf.knnMatch(desc_ref, desc_frame, k=2)
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)
    
    print(f"Number of good matches: {len(good_matches)}")
    return kp_ref, kp_frame, good_matches


def detect_and_match_features_orb(webcam_img, flann, kp_ref, des_ref, orb):
    
    # kp_frame = orb.detect(webcam_img, None)
    # _, des_frame = beb.compute(webcam_img, kp_frame)
    kp_frame, des_frame = orb.detectAndCompute(webcam_img, None)
    
    if des_ref is None or des_frame is None or len(des_ref) < 2 or len(des_frame) < 2:
        print("Not enough descriptors found.")
        return kp_ref, kp_frame, []
    
    return match_features(des_ref, des_frame, kp_ref, kp_frame, use_fginn=True)


def detect_and_match_features_surf(ref_img, webcam_img):
    surf = cv2.xfeatures2d.SURF_create(hessianThreshold=400, nOctaves=4, nOctaveLayers=3, extended=False, upright=False)

    kp_ref, des_ref = surf.detectAndCompute(ref_img, None)
    kp_frame, des_frame = surf.detectAndCompute(webcam_img, None)

    return match_features(des_ref, des_frame, kp_ref, kp_frame, is_sift=True)


def detect_and_match_features_sift(ref_img, webcam_img):
    sift = cv2.SIFT_create(nfeatures=5000, contrastThreshold=0.04, edgeThreshold=10, sigma=1.6)
    kp_ref, des_ref = sift.detectAndCompute(ref_img, None)
    kp_frame, des_frame = sift.detectAndCompute(webcam_img, None)
    return match_features(des_ref, des_frame, kp_ref, kp_frame, is_sift=True)




def find_homography_and_transform(ref_img, webcam_img, kp_ref, kp_frame, good_matches, screw_locations):
    if len(good_matches) <= MIN_MATCH_COUNT:
        # print("Not enough matches found")
        return False, None

    ref_pts = np.float32([kp_ref[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    frame_pts = np.float32([kp_frame[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(ref_pts, frame_pts, cv2.RANSAC, 5.0)
    # cv2.imwrite("results/calc_test.jpeg", webcam_img)
    if H is None or not (0.01 < np.linalg.det(H) < 100):
        print("Invalid homography")
        return False, None
    
    
    screw_positions = np.float32(screw_locations).reshape(-1, 1, 2)
    transformed_positions = cv2.perspectiveTransform(screw_positions, H)
    # print(transformed_positions)
    #affine_matrix, _ = cv2.estimateAffinePartial2D(screw_positions, transformed_positions, method=cv2.RANSAC)
    
    #transformed_positions = cv2.transform(screw_positions, affine_matrix)
    # print(transformed_positions)
    
    # This code is for testing purposes
    # It is for viewing the matches between two images
    # matches_mask = mask.ravel().tolist()
    # draw_params = dict(matchColor=(0, 255, 0), singlePointColor=None, matchesMask=matches_mask, flags=2)
    # img_matches = cv2.drawMatches(ref_img, kp_ref, webcam_img, kp_frame, good_matches, None, **draw_params)
    # cv2.imshow("Matches", img_matches)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return True, transformed_positions


def draw_matches(webcam_frame, screw_positions):
    for pos in screw_positions:
        x, y = pos[0]
        cv2.circle(webcam_frame, (int(x), int(y)), 10, (0, 255, 0), -1)

    cv2.imshow("Mapped Screw Positions", webcam_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
