import face_recognition
import pickle
import os
import cv2

DATASET_PATH = "dataset"
ENCODINGS_PATH = "models/encodings.pickle"

print("[INFO] กำลังเริ่มประมวลผลใบหน้าจาก dataset...")
knownEncodings = []
knownNames = []

# วนลูปเข้าไปในแต่ละโฟลเดอร์ย่อย (ชื่อคน) ใน dataset
for person_name in os.listdir(DATASET_PATH):
    person_folder_path = os.path.join(DATASET_PATH, person_name)

    # ตรวจสอบว่าเป็นโฟลเดอร์หรือไม่
    if not os.path.isdir(person_folder_path):
        continue

    print(f"[INFO] กำลังประมวลผล {person_name}...")

    # วนลูปในไฟล์รูปภาพของคนๆ นั้น
    for image_name in os.listdir(person_folder_path):
        image_path = os.path.join(person_folder_path, image_name)
        
        # --- START FIX (เพิ่ม try...except) ---
        # เราจะพยายามอ่านและประมวลผล
        # ถ้าไฟล์ไหนมีปัญหา (เช่น เป็น Grayscale, RGBA, Thumbs.db) มันจะโดดไปที่ except
        try:
            # อ่านรูปภาพด้วย OpenCV
            image = cv2.imread(image_path)
            
            # เช็คว่าอ่านไฟล์ได้หรือไม่ (กันไฟล์ขยะ Thumbs.db)
            if image is None:
                print(f"[Warning] ข้ามไฟล์ {image_path} (อ่านไม่ได้ หรือไม่ใช่รูปภาพ)")
                continue

            # แปลงจาก BGR (OpenCV) เป็น RGB (face_recognition)
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # 1. ค้นหาใบหน้าในรูป
            boxes = face_recognition.face_locations(rgb, model="hog")

            # 2. สร้าง encodings (ลายนิ้วมือ) จากใบหน้าที่เจอ
            encodings = face_recognition.face_encodings(rgb, boxes)

            # 3. เก็บ encodings พร้อมชื่อ
            for encoding in encodings:
                knownEncodings.append(encoding)
                knownNames.append(person_name)
                
        except Exception as e:
            # ถ้าเกิด Error ใดๆ กับไฟล์นี้ ให้พิมพ์เตือนและข้ามไปไฟล์ถัดไป
            print(f"[Warning] ข้ามไฟล์ {image_path} (เกิดปัญหา: {e})")
        # --- END FIX ---

# 4. บันทึก encodings และ names ทั้งหมดลงไฟล์ pickle
print("[INFO] กำลังบันทึก encodings ลงไฟล์...")
data = {"encodings": knownEncodings, "names": knownNames}
with open(ENCODINGS_PATH, "wb") as f:
    f.write(pickle.dumps(data))

print("[INFO] เสร็จสิ้นการเทรนโมเดล")