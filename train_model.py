import cv2
import os
import numpy as np
import pickle
import sys

PROTOTXT_PATH = "models/deploy.prototxt.txt"
MODEL_PATH = "models/res10_300x300_ssd_iter_140000.caffemodel"

# ตรวจสอบและโหลดโมเดล DNN/SSD สำหรับใช้ในการตรวจจับใบหน้าในชุดข้อมูล
try:
    net = cv2.dnn.readNetFromCaffe(PROTOTXT_PATH, MODEL_PATH)
    MIN_CONFIDENCE = 0.5
except Exception as e:
    print(f"[CRITICAL ERROR] ไม่สามารถโหลดโมเดล DNN/SSD ได้: {e}", file=sys.stderr)
    net = None
# --------------------------------

# --- การจัดการพาธสำหรับการเทรนและการบันทึก ---
DATASET_PATH = os.path.join(os.getcwd(), "dataset")
MODEL_SAVE_PATH = os.path.join(os.getcwd(), "models", "lbph_model.yml") 
NAMES_SAVE_PATH = os.path.join(os.getcwd(), "models", "lbph_names.pickle") 

recognizer = cv2.face.LBPHFaceRecognizer_create()

def get_images_and_labels():
    if net is None:
        print("[ERROR] ไม่สามารถตรวจจับใบหน้าได้เนื่องจากโมเดล DNN/SSD มีปัญหา", file=sys.stderr)
        return [], [], {}

    image_paths = []
    # ใช้ os.walk บน DATASET_PATH ที่กำหนดไว้
    for root, dirs, files in os.walk(DATASET_PATH): 
        for file in files:
            # ตรวจสอบเฉพาะไฟล์รูปภาพ
            if file.lower().endswith((".png", ".jpg", ".jpeg")): 
                image_paths.append(os.path.join(root, file))

    face_samples = [] 
    labels = []       
    name_map = {}     
    current_id = 0
    
    print(f"[INFO] กำลังประมวลผลรูปภาพทั้งหมด {len(image_paths)} รูป...")

    for image_path in image_paths:
        # ชื่อบุคคลคือชื่อโฟลเดอร์ที่รูปภาพอยู่ (เช่น Thinakorn)
        person_name = os.path.basename(os.path.dirname(image_path)) 

        if person_name not in name_map:
            name_map[person_name] = current_id
            current_id += 1
        
        label_id = name_map[person_name]

        image = cv2.imread(image_path)
        if image is None:
            print(f"[Warning] ข้ามไฟล์ (อ่านไม่ได้): {os.path.basename(image_path)}")
            continue

        # ตรวจจับใบหน้าในภาพ
        (h, w) = image.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
            (300, 300), (104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()

        # ค้นหาใบหน้าที่มั่นใจที่สุด
        best_face_box = None
        best_confidence = 0.0
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > MIN_CONFIDENCE and confidence > best_confidence:
                best_confidence = confidence
                box = detections[0, 0, i, 3:7] * [w, h, w, h]
                best_face_box = box.astype("int")

        if best_face_box is not None:
            (startX, startY, endX, endY) = best_face_box
            
            # Crop ใบหน้าและแปลงเป็น Grayscale
            face_roi = image[startY:endY, startX:endX]
            gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            
            # ต้องตรวจสอบขนาดของ face_roi ก่อนเพิ่ม
            if gray_face.size > 0:
                face_samples.append(gray_face)
                labels.append(label_id)
                print(f"    [Success] เพิ่มใบหน้า {person_name} (ID: {label_id}) จาก {os.path.basename(image_path)}")
            else:
                print(f"[Warning] ขนาด ROI เป็นศูนย์ใน: {os.path.basename(image_path)}")
        else:
            print(f"[Warning] ไม่พบใบหน้าใน: {os.path.basename(image_path)}")

    return face_samples, labels, name_map

# --- Main Logic สำหรับการเทรน ---
print("[INFO] กำลังเตรียมข้อมูล...")
faces, ids, name_map = get_images_and_labels()

if len(faces) == 0:
    print("[ERROR] ไม่พบใบหน้าดีๆ ใน dataset เลย! กรุณาเพิ่มรูป .jpg ที่ชัดเจนในโฟลเดอร์ dataset")
else:
    print(f"[INFO] พบ {len(faces)} ใบหน้าสำหรับเทรน")
    print("[INFO] กำลังเทรนโมเดล LBPH...")
    # ตรวจสอบให้แน่ใจว่าทั้ง faces และ ids ไม่ว่าง
    try:
        recognizer.train(faces, np.array(ids))

        # บันทึกโมเดล
        recognizer.save(MODEL_SAVE_PATH)
        print(f"[INFO] บันทึกโมเดล LBPH ไปที่ {MODEL_SAVE_PATH}")
        
        # สร้างและบันทึก Name Map
        id_to_name_map = {v: k for k, v in name_map.items()}
        with open(NAMES_SAVE_PATH, 'wb') as f:
            f.write(pickle.dumps(id_to_name_map))
        print(f"[INFO] บันทึกชื่อไปที่ {NAMES_SAVE_PATH}")
        print("[INFO] เสร็จสิ้นการเทรน!")
    except Exception as e:
        print(f"[CRITICAL ERROR] การเทรนโมเดลล้มเหลว: {e}", file=sys.stderr)