import cv2

import pickle

import os

import sys



# --- การจัดการพาธโมเดล ---

# หาพาธของโฟลเดอร์ 'core'

CORE_DIR = os.path.dirname(os.path.abspath(__file__))

# หาพาธของโฟลเดอร์หลักของโปรเจกต์ (parent directory ของ 'core')

BASE_DIR = os.path.dirname(CORE_DIR)



# สร้างพาธที่แน่นอนไปยังไฟล์โมเดล

MODEL_PATH = os.path.join(BASE_DIR, "models", "lbph_model.yml")

NAMES_PATH = os.path.join(BASE_DIR, "models", "lbph_names.pickle")



# ตรวจสอบว่ามีไฟล์โมเดลที่เทรนไว้หรือไม่

if not os.path.exists(MODEL_PATH) or not os.path.exists(NAMES_PATH):

    # ใช้ sys.stderr เพื่อแสดงข้อความแจ้งเตือนที่ชัดเจน

    print("-" * 50, file=sys.stderr)

    print(f"[ERROR] ไม่พบโมเดล LBPH ที่คาดหวัง:", file=sys.stderr)

    print(f"  > Model: {MODEL_PATH}", file=sys.stderr)

    print(f"  > Names: {NAMES_PATH}", file=sys.stderr)

    print("กรุณารัน train_model.py ก่อน!", file=sys.stderr)

    print("-" * 50, file=sys.stderr)

    recognizer = None

    id_to_name_map = {}

else:

    # โหลดโมเดล LBPH

    print("[INFO] กำลังโหลดโมเดล LBPH...")

    recognizer = cv2.face.LBPHFaceRecognizer_create()

    recognizer.read(MODEL_PATH)

   

    with open(NAMES_PATH, 'rb') as f:

        id_to_name_map = pickle.load(f)

    print("[INFO] โหลดโมเดล LBPH สำเร็จ")





def recognize_faces_lbph(frame, face_boxes):

    names = []

    confidences = []



    # ถ้าไม่มีโมเดล (เทรนไม่ผ่าน) ให้คืน Unknown

    if recognizer is None:

        return ["Unknown"] * len(face_boxes), [0.0] * len(face_boxes)



    # ตรวจสอบว่าเฟรมมีข้อมูลหรือไม่

    if frame is None or len(frame.shape) < 2:

        return ["Error"] * len(face_boxes), [0.0] * len(face_boxes)



    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)



   

    for (x, y, w, h) in face_boxes:

       

        face_roi = gray_frame[y:y+h, x:x+w]

       

        if face_roi.size == 0:

            names.append("Error_ROI")

            confidences.append(0)

            continue



        try:

            # LBPH ต้องการภาพขนาดเท่าเดิมที่ใช้ตอนเทรน (แต่ปกติจะปรับเอง)

            label_id, confidence = recognizer.predict(face_roi)

           

            # ตั้งค่าเกณฑ์ความมั่นใจ (Confidence Threshold)

            # ค่า confidence ยิ่งต่ำยิ่งดีสำหรับ LBPH

            CONFIDENCE_THRESHOLD = 140



            if confidence < CONFIDENCE_THRESHOLD:

                name = id_to_name_map.get(label_id, "Unknown")

            else:

                name = "Unknown"

               

            names.append(name)

            confidences.append(confidence)



        except cv2.error as e:

            # print(f"CV2 Error in prediction: {e}") # สำหรับดีบั๊ก

            names.append("Error")

            confidences.append(0)

           

    return names, confidences

