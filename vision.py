
from google.cloud import vision
import os
import threading
import cv2
import time

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "secret.json"
materials = ['plastic', 'paper', 'metal', 'glass', 'cardboard']
client = vision.ImageAnnotatorClient()

def detect_labels(path, results, lock):
    local_result = {material: 0 for material in materials}

    with open(path, "rb") as image_file:
        content = image_file.read()

    image = vision.Image(content=content)
    response = client.label_detection(image=image)
    labels = response.label_annotations

    for label in labels:
        description = label.description.lower()
        for material in materials:
            if material in description:
                local_result[material] = max(local_result[material], label.score)

    if response.error.message:
        raise Exception(
            "{}\nFor more info on error messages, check: "
            "https://cloud.google.com/apis/design/errors".format(response.error.message)
        )

    with lock:
        for material in materials:
            results[material] += local_result[material]

def capture_image(file_path):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("unable to access camera")
        exit()

    ret, frame = cap.read()
    cv2.imwrite(file_path, frame)
    cap.release()

if __name__ == "__main__":
    image_paths = ["test1.jpg", "test2.jpg"]
    for path in image_paths:
        capture_image(path)
        time.sleep(1)
    combined_results = {material: 0 for material in materials}
    threads = []
    lock = threading.Lock()

    for path in image_paths:
        thread = threading.Thread(target=detect_labels, args=(path, combined_results, lock))
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

    max_score = 0
    max_material = "Other"
    for material, score in combined_results.items():
        if score > max_score:
            max_score = score
            max_material = material
            
    print(max_material)
