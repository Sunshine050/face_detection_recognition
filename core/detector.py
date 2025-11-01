import cv2
import numpy as np
import os # เพิ่ม import os เพื่อใช้จัดการพาธ
import sys

# --- การจัดการพาธโมเดล ---
# หาพาธของโฟลเดอร์ 'core'
CORE_DIR = os.path.dirname(os.path.abspath(__file__)) 
# หาพาธของโฟลเดอร์หลักของโปรเจกต์
BASE_DIR = os.path.dirname(CORE_DIR)

# สร้างพาธที่แน่นอนไปยังไฟล์โมเดล DNN/SSD
PROTOTXT_PATH = os.path.join(BASE_DIR, "models", "deploy.prototxt.txt")
MODEL_PATH = os.path.join(BASE_DIR, "models", "res10_300x300_ssd_iter_140000.caffemodel")

# ตรวจสอบว่าไฟล์โมเดลอยู่หรือไม่
if not os.path.exists(PROTOTXT_PATH) or not os.path.exists(MODEL_PATH):
    print("-" * 50, file=sys.stderr)
    print("[ERROR] ไม่พบไฟล์โมเดล DNN/SSD ที่คาดหวัง:", file=sys.stderr)
    print("  > Protottxt:", PROTOTXT_PATH, file=sys.stderr)
    print("  > Model:", MODEL_PATH, file=sys.stderr)
    print("โปรดตรวจสอบว่าไฟล์อยู่ในโฟลเดอร์ 'models' ถูกต้อง", file=sys.stderr)
    print("-" * 50, file=sys.stderr)
    # กำหนด net เป็น None เพื่อป้องกันโปรแกรมพัง
    net = None 
else:
    # โหลดโมเดล "Pro" ด้วย OpenCV DNN (Deep Neural Network)
    print("[INFO] กำลังโหลดโมเดล DNN/SSD...")
    net = cv2.dnn.readNetFromCaffe(PROTOTXT_PATH, MODEL_PATH)
    print("[INFO] โหลดโมเดล DNN/SSD สำเร็จ")

# ถ้าโมเดลไม่มั่นใจถึง 50% (0.5) เราจะไม่นับว่านั่นคือใบหน้า
MIN_CONFIDENCE = 0.5

def detect_faces(image):
    # ถ้าโมเดลโหลดไม่สำเร็จ
    if net is None:
        return []

    # ตรวจสอบว่าภาพไม่ใช่ None
    if image is None or len(image.shape) < 2:
        return []

    (h, w) = image.shape[:2]
    
    # Preprocessing: ปรับขนาดภาพและสร้าง Blob
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
        (300, 300), (104.0, 177.0, 123.0)) # ค่าเฉลี่ย RGB ที่แนะนำสำหรับโมเดลนี้

    net.setInput(blob)
    detections = net.forward()
    
    detected_faces = [] 
    
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        
        if confidence > MIN_CONFIDENCE:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            
            # การปรับ Box เพื่อให้เป็นรูปแบบ (x, y, width, height)
            x = startX
            y = startY
            box_w = endX - startX
            box_h = endY - startY
            
            detected_faces.append(((x, y, box_w, box_h), confidence))
            
    return detected_faces

# --- ส่วนนี้ไว้สำหรับทดสอบโมดูล (อ้างอิงไฟล์ในโฟลเดอร์หลัก) ---
if __name__ == "__main__":
    # ใช้ os.path.join เพื่อสร้างพาธที่ยืดหยุ่นสำหรับไฟล์ทดสอบในโฟลเดอร์หลัก
    test_image_path = os.path.join(BASE_DIR, "test_image.jpg")
    test1_image_path = os.path.join(BASE_DIR, "test1_image.jpg")
    image_files = [test_image_path, test1_image_path]
    
    for image_path in image_files:
        image = cv2.imread(image_path)
        if image is None:
            print(f"ไม่พบไฟล์ {os.path.basename(image_path)}")
            continue
            
        # ... โค้ดส่วนที่เหลือของการทดสอบเหมือนเดิม
        faces_found = detect_faces(image)
        print(f"{os.path.basename(image_path)} → ตรวจพบ {len(faces_found)} ใบหน้า")
        
        for (box, confidence) in faces_found:
            (x, y, w, h) = box
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            text = f"{confidence * 100:.2f}%"
            cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

        cv2.imshow(f"Face Detection (PRO) - {os.path.basename(image_path)}", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()