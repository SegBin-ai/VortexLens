import numpy as np
import cv2

def apply_gaussian_smoothing(image):
    # Apply Gaussian blur with kernel size (3, 3) and sigma 10
    blurred_img = cv2.GaussianBlur(image, (3, 3), 50)
    return blurred_img

def apply_clahe(image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image)

def apply_prewitt_edge_detection(image):
    kernelx = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=int)
    kernely = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=int)
    img_prewittx = cv2.filter2D(image, -1, kernelx)
    img_prewitty = cv2.filter2D(image, -1, kernely)
    return cv2.bitwise_or(img_prewittx, img_prewitty)

def preprocess_proposal(image):
    # blur = cv2.medianBlur(image, 5)
    blur = image
    # # # thresh = cv2.adaptiveThreshold(unsharp, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    # clahe_frame = apply_clahe(image)

    # smoothed_frame = apply_gaussian_smoothing(clahe_frame)

    # edges = apply_prewitt_edge_detection(smoothed_frame)
    # cv2.imshow("Edges", edges)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply median blur
    # Edge detection
    # edges = cv2.Canny(blur, 100, 200)
    # # Contrast adjustment
    # equ = cv2.equalizeHist(edges)
    # cv2.imshow("Edges", edges)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return blur