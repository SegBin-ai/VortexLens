import cv2
import numpy as np
import h5py
from keras.models import model_from_json
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torchvision import transforms as T

IMAGENET_MEAN = [0.485, 0.456, 0.406]  # RGB mean
IMAGENET_STD = [0.229, 0.224, 0.225]   # RGB standard deviation


IMAGE_SIZE = (71, 71)
MIN_CONFIDENCE = 0.2

model_path = 'C:\\Users\\Aaditya Voruganti\\Desktop\\VortexLens\\screw_exploration\\Blob_detection\\screw_head_detector-2.h5'

def model_init():
    
    with h5py.File(model_path, 'r') as f:
        model_config = f.attrs.get('model_config')

    model = model_from_json(model_config)
    model.load_weights(model_path)
    return model

#required pre processing for the model
def preprocess_proposal(region, image_size=IMAGE_SIZE):
    proposal = cv2.resize(region, image_size)
    proposal = proposal.astype('float32') / 255.0
    proposal = np.expand_dims(proposal, axis=0)
    return proposal


def yolo_model_init():
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='C:\\Users\\Aaditya Voruganti\\Desktop\VortexLens\\screw_exploration\\Blob_detection\\best_new.pt')
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    return model


def classify_transforms(size=224):
    """Applies a series of transformations including center crop, ToTensor, and normalization for classification."""
    assert isinstance(size, int), f"ERROR: classify_transforms size {size} must be integer, not (list, tuple)"
    # T.Compose([T.ToTensor(), T.Resize(size), T.CenterCrop(size), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    return T.Compose([CenterCrop(size), ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])


class CenterCrop:
    # YOLOv5 CenterCrop class for image preprocessing, i.e. T.Compose([CenterCrop(size), ToTensor()])
    def __init__(self, size=640):
        """Initializes CenterCrop for image preprocessing, accepting single int or tuple for size, defaults to 640."""
        super().__init__()
        self.h, self.w = (size, size) if isinstance(size, int) else size

    def __call__(self, im):
        """
        Applies center crop to the input image and resizes it to a specified size, maintaining aspect ratio.

        im = np.array HWC
        """
        imh, imw = im.shape[:2]
        m = min(imh, imw)  # min dimension
        top, left = (imh - m) // 2, (imw - m) // 2
        return cv2.resize(im[top : top + m, left : left + m], (self.w, self.h), interpolation=cv2.INTER_LINEAR)


class ToTensor:
    # YOLOv5 ToTensor class for image preprocessing, i.e. T.Compose([LetterBox(size), ToTensor()])
    def __init__(self, half=False):
        """Initializes ToTensor for YOLOv5 image preprocessing, with optional half precision (half=True for FP16)."""
        super().__init__()
        self.half = half

    def __call__(self, im):
        """
        Converts BGR np.array image from HWC to RGB CHW format, and normalizes to [0, 1], with support for FP16 if
        `half=True`.

        im = np.array HWC in BGR order
        """
        im = np.ascontiguousarray(im.transpose((2, 0, 1))[::-1])  # HWC to CHW -> BGR to RGB -> contiguous
        im = torch.from_numpy(im)  # to torch
        im = im.half() if self.half else im.float()  # uint8 to fp16/32
        im /= 255.0  # 0-255 to 0.0-1.0
        return im

def preprocess(img):
  transform_custom = classify_transforms(size=64)
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  res = transform_custom(img)
  im = torch.Tensor(res).to(device)
  if len(im.shape) == 3:
    im = im[None]  # expand for batch dim

  return im


def detect_screws_yolo(img, model):
    img_tensor = preprocess(img)
    with torch.no_grad():
        results = model(img_tensor)
    
    probabilities = F.softmax(results, dim=1)
    predicted_class = torch.argmax(probabilities, dim=1).item()
    if predicted_class == 1:
        return True



def detect_screws_blobs(img, model):
    """
    Detect screw heads in the image using blob detection and classify using the model.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    
    #blob detection parameters
    #These parameters need some tuning to detect the right circles that could be screw heads

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
    
    screw_locations = []

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
        
        # result = detect_screws_yolo(region, model)
        
        
        #The model has been trained so that a score closer to 0 means a screw head is present

        if score < MIN_CONFIDENCE:
        # if result == True:
            screw_locations.append((x, y))
            color = (0, 255, 0)
            cv2.rectangle(img, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), color, 2)
            
        # else:
        #     color = (0, 0, 255)
        #     cv2.rectangle(img, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), color, 2)

    #saves the image with the detected screws
    cv2.imwrite("results/detected_screws_blobs_model_2.jpg", img)
    return screw_locations


def detect_using_locations_demo(web_frame, screw_locations, index, screws_placed):
    for i, pos in enumerate(screw_locations):
        x, y = int(pos[0][0]), int(pos[0][1])
        top_left_x = max(0, x - 9)
        top_left_y = max(0, y - 9)
        bottom_right_x = min(web_frame.shape[1], x + 9)
        bottom_right_y = min(web_frame.shape[0], y + 9)
        
        if i in screws_placed[:index]:
            color = (0, 255, 0)
            cv2.rectangle(web_frame, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), color, 2)
            
        else:
            color = (0, 0, 255)
            cv2.rectangle(web_frame, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), color, 2)
                 
    return web_frame


def detect_using_locations(web_frame, screw_locations, model, screw_states):
    for idx, pos in enumerate(screw_locations):
        x, y = int(pos[0][0]), int(pos[0][1])
        top_left_x = max(0, x - 9)
        top_left_y = max(0, y - 9)
        bottom_right_x = min(web_frame.shape[1], x + 9)
        bottom_right_y = min(web_frame.shape[0], y + 9)
        
        
        region = web_frame[top_left_y:bottom_right_y, top_left_x:bottom_right_x]
        if region.size == 0:
            continue
        
        # proposal = preprocess_proposal(region)
        # prediction = model.predict(proposal)
        # score = prediction[0][0]
        score = detect_screws_yolo(region, model)
        
        if score == True:
            color = (0, 255, 0)
            cv2.rectangle(web_frame, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), color, 2)
            screw_states[idx] = True
            # cv2.putText(web_frame, f'Score: {score:.2f}', (top_left_x, top_left_y - 10),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
        else:
            color = (0, 0, 255)
            cv2.rectangle(web_frame, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), color, 2)
            screw_states[idx] = False
            # cv2.putText(web_frame, f'Score: {score:.2f}', (top_left_x, top_left_y - 10),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            
    return web_frame, screw_states