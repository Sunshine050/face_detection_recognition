import cv2
import numpy as np
import os
import sys

# --- (START) แก้ไขการจัดการพาธ ---
# เราจะใช้ Current Working Directory (โฟลเดอร์ที่คุณรัน python) เป็นตัวตั้ง
try:
    BASE_DIR = os.getcwd() 
except Exception as e:
    print(f"[ERROR] Cannot get Current Working Directory: {e}", file=sys.stderr)
    sys.exit(1)
# --- (END) แก้ไขการจัดการพาธ ---

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
    if net is None:
        print("[ERROR] DNN net is not loaded. Cannot detect faces.", file=sys.stderr)
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

            x = startX
            y = startY
            box_w = endX - startX
            box_h = endY - startY

            detected_faces.append(((x, y, box_w, box_h), confidence))

    return detected_faces

if __name__ == "__main__":
    # (ส่วนเทสนี้จะยังคงใช้ os.getcwd() ซึ่งถูกต้องแล้ว)
    BASE_DIR = os.getcwd() 
    test_image_path = os.path.join(BASE_DIR, "test_image.jpg")
    test1_image_path = os.path.join(BASE_DIR, "test1_image.jpg")
    image_files = [test_image_path, test1_image_path]

    if net is None:
        print("[ERROR] Cannot run test, DNN net is not loaded.")
    else:
        for image_path in image_files:
            image = cv2.imread(image_path)
            if image is None:
                print(f"File not found {os.path.basename(image_path)}")
                continue

            faces_found = detect_faces(image)
            print(f"{os.path.basename(image_path)} → Detected {len(faces_found)} faces")

            for (box, confidence) in faces_found:
                (x, y, w, h) = box
                cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                text = f"{confidence * 100:.2f}%"
                cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

            cv2.imshow(f"Face Detection (PRO) - {os.path.basename(image_path)}", image)
            cv2.waitKey(0)
        cv2.destroyAllWindows()

