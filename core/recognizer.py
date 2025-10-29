import cv2
import pickle
import os

MODEL_PATH = "models/lbph_model.yml"
NAMES_PATH = "models/lbph_names.pickle"

# ตรวจสอบว่ามีไฟล์โมเดลที่เทรนไว้หรือไม่
if not os.path.exists(MODEL_PATH) or not os.path.exists(NAMES_PATH):
    print("[ERROR] ไม่พบโมเดล LBPH (lbph_model.yml) หรือ (lbph_names.pickle)")
    print("กรุณารัน train_model.py ก่อน!")
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

    #ถ้าไม่มีโมเดล (เทรนไม่ผ่าน) ให้คืน Unknown
    if recognizer is None:
        return ["Unknown"] * len(face_boxes), [0.0] * len(face_boxes)


    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    
    for (x, y, w, h) in face_boxes:
        
        
        face_roi = gray_frame[y:y+h, x:x+w]
        
        try:
            label_id, confidence = recognizer.predict(face_roi)
           
            if confidence < 140: 
                name = id_to_name_map.get(label_id, "Unknown")
            else:
                name = "Unknown"
                
            names.append(name)
            confidences.append(confidence) 

        except cv2.error:
            names.append("Error")
            confidences.append(0)
            
    return names, confidences