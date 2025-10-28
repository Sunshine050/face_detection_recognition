import face_recognition
import pickle

# Path ของไฟล์ที่เราเทรนไว้
ENCODINGS_PATH = "models/encodings.pickle"

# 1. โหลดฐานข้อมูล encodings (ทำครั้งเดียวตอนเริ่ม)
print("[INFO] กำลังโหลด encodings...")
try:
    with open(ENCODINGS_PATH, 'rb') as f:
        data = pickle.load(f)
except FileNotFoundError:
    print(f"[ERROR] ไม่พบไฟล์ {ENCODINGS_PATH}")
    print("กรุณารัน train_model.py ก่อน")
    data = {"encodings": [], "names": []}

def recognize_faces(rgb_face_image, face_locations):
    """
    รับภาพ RGB และพิกัดใบหน้าที่ 'ตรวจพบ' เข้ามา
    คืนค่า list ของ 'ชื่อ' ที่ระบุตัวตนได้
    """
    if not data["encodings"]:
        return ["Unknown"] * len(face_locations) # ถ้าไม่มีฐานข้อมูล ก็คืน Unknown

    # 1. สร้าง encodings (ลายนิ้วมือ) จากใบหน้าที่เพิ่งเจอสดๆ
    # สังเกตว่าเราใช้ face_locations (พิกัด) ที่ได้จาก 'detector' ของคุณ
    # แต่เราต้องคำนวณ encoding จาก 'recognizer' เพื่อให้ได้ค่าที่แม่นยำ
    current_encodings = face_recognition.face_encodings(rgb_face_image, face_locations)
    
    names = []

    # 2. วนลูปใน encodings ที่เพิ่งเจอ
    for encoding in current_encodings:
        # 3. เปรียบเทียบกับฐานข้อมูล (knownEncodings)
        # tolerance คือค่ายิ่งน้อยยิ่งเข้มงวด (0.6 คือค่ามาตรฐาน)
        matches = face_recognition.compare_faces(data["encodings"], encoding, tolerance=0.6)
        name = "Unknown" # ตั้งค่าเริ่มต้น

        # 4. ตรวจสอบว่ามีใบหน้าที่ 'ตรงกัน' หรือไม่
        if True in matches:
            # หากตรงกัน ให้หาว่าตรงกับใครมากที่สุด
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}

            # นับว่าตรงกับชื่อไหนกี่ครั้ง
            for i in matchedIdxs:
                name = data["names"][i]
                counts[name] = counts.get(name, 0) + 1

            # เลือกชื่อที่ถูกโหวต (match) มากที่สุด
            name = max(counts, key=counts.get)
        
        names.append(name)
        
    return names