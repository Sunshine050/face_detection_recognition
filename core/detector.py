import cv2
import numpy as np

# Path ไปยังโมเดล Deep Learning (SSD)
PROTOTXT_PATH = "models/deploy.prototxt.txt"
MODEL_PATH = "models/res10_300x300_ssd_iter_140000.caffemodel"

# โหลดโมเดล "Pro" ด้วย OpenCV DNN (Deep Neural Network)
net = cv2.dnn.readNetFromCaffe(PROTOTXT_PATH, MODEL_PATH)

# ตั้งค่าความมั่นใจ (Confidence) ขั้นต่ำ
# ถ้าโมเดลไม่มั่นใจถึง 50% (0.5) เราจะไม่นับว่านั่นคือใบหน้า
MIN_CONFIDENCE = 0.5

def detect_faces(image):
    """
    รับภาพ (image) เข้ามา แล้วคืนค่า list ของพิกัด (x, y, w, h)
    และ 'ความมั่นใจ' (confidence) ของใบหน้าที่เจอ
    """
    
    # 1. ดึงขนาดของภาพ
    (h, w) = image.shape[:2]
    
    # 2. แปลงภาพเป็น "blob" (รูปแบบที่โมเดล DNN ต้องการ)
    # ทำการ Resize เป็น 300x300 และปรับค่เฉลี่ยสี
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
        (300, 300), (104.0, 177.0, 123.0))

    # 3. ส่ง blob เข้าโมเดล
    net.setInput(blob)
    detections = net.forward()
    
    # 4. เตรียม list ผลลัพธ์
    detected_faces = [] # [(x, y, w, h), confidence]
    
    # 5. วนลูปผลลัพธ์ที่โมเดลตรวจเจอ
    for i in range(0, detections.shape[2]):
        # 6. ดึงค่าความมั่นใจ (confidence)
        confidence = detections[0, 0, i, 2]
        
        # 7. กรองเฉพาะที่มั่นใจเกิน MIN_CONFIDENCE (0.5)
        if confidence > MIN_CONFIDENCE:
            # 8. คำนวณพิกัด (x, y) คืนเป็นขนาดจริงของภาพ
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            
            # 9. แปลงกลับเป็น (x, y, w, h)
            x = startX
            y = startY
            box_w = endX - startX
            box_h = endY - startY
            
            # 10. เก็บผลลัพธ์
            detected_faces.append(((x, y, box_w, box_h), confidence))
            
    # คืนค่า list ของ tuple ที่มี (พิกัด, ความมั่นใจ)
    return detected_faces

# --- ส่วนนี้ไว้สำหรับทดสอบโมดูล (เหมือนเดิม) ---
if __name__ == "__main__":
    image_files = ["test_image.jpg", "test1_image.jpg"]
    
    for image_path in image_files:
        image = cv2.imread(image_path)
        if image is None:
            print(f"ไม่พบไฟล์ {image_path}")
            continue
            
        # เรียกใช้ฟังก์ชัน "Pro" ของคุณ
        faces_found = detect_faces(image)
        print(f"{image_path} → ตรวจพบ {len(faces_found)} ใบหน้า")
        
        for (box, confidence) in faces_found:
            (x, y, w, h) = box
            # วาดกรอบ
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            # แสดง % ความมั่นใจ
            text = f"{confidence * 100:.2f}%"
            cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

        cv2.imshow(f"Face Detection (PRO) - {image_path}", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()