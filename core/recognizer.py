import cv2
import pickle
import os
import sys
import numpy as np

# --- การจัดการพาธโมเดล ---
CORE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CORE_DIR)
MODEL_PATH = os.path.join(BASE_DIR, "models", "lbph_model.yml")
NAMES_PATH = os.path.join(BASE_DIR, "models", "lbph_names.pickle")

if not os.path.exists(MODEL_PATH) or not os.path.exists(NAMES_PATH):
    print("-" * 50, file=sys.stderr)
    print(f"[ERROR] ไม่พบโมเดล LBPH ที่คาดหวัง:", file=sys.stderr)
    print(f"  > Model: {MODEL_PATH}", file=sys.stderr)
    print(f"  > Names: {NAMES_PATH}", file=sys.stderr)
    print("กรุณารัน train_model.py ก่อน!", file=sys.stderr)
    print("-" * 50, file=sys.stderr)
    recognizer = None
    id_to_name_map = {}
else:
    print("[INFO] กำลังโหลดโมเดล LBPH...")
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(MODEL_PATH)

    with open(NAMES_PATH, 'rb') as f:
        id_to_name_map = pickle.load(f)

    print("[INFO] โหลดโมเดล LBPH สำเร็จ")
    print(f"[INFO] Loaded {len(id_to_name_map)} labels")

# ------------------------------
# ✅ ค่า threshold — ปรับได้ตามโมเดลของคุณ
# ------------------------------
# หมายเหตุ:
# - ค่าต่ำ = แม่นยำ แต่จะ Unknown ง่าย
# - ค่าสูง = ยอมให้คล้ายๆ ก็ถือว่าแมทช์
DEFAULT_CONFIDENCE_THRESHOLD = 90


def recognize_faces_lbph(frame, face_boxes, confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD):
    """
    จดจำใบหน้าโดยใช้ LBPH
    - frame: BGR image (OpenCV)
    - face_boxes: list of (x,y,w,h)
    - confidence_threshold: ค่า threshold เพื่อพิจารณาว่าแมทหรือไม่
    คืนค่า: (names_list, confidences_list)
    """
    names = []
    confidences = []

    if recognizer is None:
        return ["Unknown"] * len(face_boxes), [0.0] * len(face_boxes)

    if frame is None or len(frame.shape) < 2:
        return ["Error"] * len(face_boxes), [0.0] * len(face_boxes)

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    for (x, y, w, h) in face_boxes:
        # ตรวจสอบ boundary
        x1 = max(x, 0)
        y1 = max(y, 0)
        x2 = min(x + w, gray_frame.shape[1])
        y2 = min(y + h, gray_frame.shape[0])

        face_roi = gray_frame[y1:y2, x1:x2]

        if face_roi.size == 0:
            names.append("Error_ROI")
            confidences.append(999.0)
            continue

        try:
            # Resize และ Normalize histogram
            face_resized = cv2.resize(face_roi, (200, 200))
            face_norm = cv2.equalizeHist(face_resized)

            label_id, confidence = recognizer.predict(face_norm)

            # Debug log
            reason = "Matched" if confidence < confidence_threshold else "Unknown"
            print(f"[DEBUG][LBPH] ID={label_id}, Conf={confidence:.2f}, Result={reason}")

            # ตัดสินว่าเป็นใคร
            if confidence < confidence_threshold:
                name = id_to_name_map.get(label_id, "Unknown")
            else:
                name = "Unknown"

            names.append(name)
            confidences.append(confidence)

        except cv2.error as e:
            print(f"[ERROR] OpenCV predict error: {e}", file=sys.stderr)
            names.append("Error_Predict")
            confidences.append(999.0)
        except Exception as e:
            print(f"[ERROR] Unexpected error: {e}", file=sys.stderr)
            names.append("Error")
            confidences.append(999.0)

    return names, confidences
