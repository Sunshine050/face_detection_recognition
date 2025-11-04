import cv2
import numpy as np
import os
import sys

# --- การจัดการพาธโมเดล ---
CORE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CORE_DIR)

PROTOTXT_PATH = os.path.join(BASE_DIR, "models", "deploy.prototxt.txt")
MODEL_PATH = os.path.join(BASE_DIR, "models", "res10_300x300_ssd_iter_140000.caffemodel")

if not os.path.exists(PROTOTXT_PATH) or not os.path.exists(MODEL_PATH):
    print("-" * 50, file=sys.stderr)
    print("[ERROR] No expected DNN/SSD model files found:", file=sys.stderr)
    print("  > Protottxt:", PROTOTXT_PATH, file=sys.stderr)
    print("  > Model:", MODEL_PATH, file=sys.stderr)
    print("Please check if the files are correctly located in the 'models' folder", file=sys.stderr)
    print("-" * 50, file=sys.stderr)
    net = None
else:
    print("[INFO] Loading DNN/SSD model...")
    net = cv2.dnn.readNetFromCaffe(PROTOTXT_PATH, MODEL_PATH)
    print("[INFO] DNN/SSD model loaded successfully")

MIN_CONFIDENCE = 0.5

def detect_faces(image):
    """ตรวจจับใบหน้าด้วย DNN/SSD"""
    if net is None:
        return []

    if image is None or len(image.shape) < 2:
        return []

    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
        (300, 300), (104.0, 177.0, 123.0))

    net.setInput(blob)
    detections = net.forward()
    detected_faces = []

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > MIN_CONFIDENCE:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            x, y = startX, startY
            box_w, box_h = endX - startX, endY - startY
            detected_faces.append(((x, y, box_w, box_h), confidence))

    return detected_faces
