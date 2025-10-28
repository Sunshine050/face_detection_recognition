# import "โมดูล Pro" ของคุณ
from core.detector import detect_faces 

# import "โมดูล" ของเพื่อน (เราจะสมมติว่าเขาสร้างเสร็จแล้ว)
from core.recognizer import recognize_faces

import cv2
import time

print("[INFO] กำลังโหลดโมเดลและเปิดกล้อง...")

# เปิดกล้อง (0 คือกล้องตัวหลัก)
cap = cv2.VideoCapture(0)

# เช็คว่าเปิดกล้องสำเร็จไหม
if not cap.isOpened():
    print("[ERROR] ไม่สามารถเปิดกล้องได้")
    exit()

# สำหรับวัด FPS (Frames Per Second)
prev_frame_time = 0
new_frame_time = 0

print("[INFO] เริ่มการทำงาน... กด 'q' เพื่อออกจากโปรแกรม")

# --- วงจรหลัก (Loop) ของแอปพลิเคชัน ---
while True:
    # 1. อ่านภาพทีละเฟรมจากกล้อง
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] ไม่สามารถรับเฟรมภาพได้")
        break
        
    # พลิกภาพ (กล้องเว็บแคมมักจะกลับด้าน)
    frame = cv2.flip(frame, 1)
    
    # 2. 🚀 (ส่วนของคุณ) เรียกใช้ Detector
    #    เราส่งภาพ frame เข้าไป
    #    ได้ผลลัพธ์เป็น list ของ ((x,y,w,h), confidence)
    detected_results = detect_faces(frame)
    
    # เตรียมพิกัด (boxes) และ ความมั่นใจ (confidences)
    boxes = []
    confidences = []
    for (box, conf) in detected_results:
        boxes.append(box)
        confidences.append(conf)

    # 3. 🚀 (ส่วนของเพื่อน) เรียกใช้ Recognizer
    #    เราต้องแปลง BGR -> RGB เพราะ recognizer ชอบ RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    #    ส่งภาพ RGB และ "พิกัด" ที่คุณหาเจอเข้าไป
    #    ได้ผลลัพธ์เป็น list ของ "ชื่อ"
    names = recognize_faces(rgb_frame, boxes)
    
    # 4. วาดผลลัพธ์ทั้งหมดลงบนจอ
    
    # วนลูปตาม "พิกัด" ที่คุณหาเจอ
    for i, (x, y, w, h) in enumerate(boxes):
        # ดึง "ชื่อ" ที่เพื่อนหาเจอ
        name = names[i]
        
        # ดึง "ความมั่นใจ" ที่คุณหาเจอ
        confidence = confidences[i]
        
        # สร้าง Text ที่จะแสดง
        text_name = f"Name: {name}"
        text_conf = f"Conf: {confidence * 100:.2f}%"
        
        # --- วาดกรอบ ---
        # (สีเขียวถ้าจำได้, สีแดงถ้า Unknown)
        color = (0, 0, 255) if name == "Unknown" else (0, 255, 0)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        
        # --- วาดชื่อและ % ---
        # สร้างพื้นหลังสีทึบ
        y_text_bg = y - 40 if y - 40 > 0 else y + 10 # ตำแหน่ง
        cv2.rectangle(frame, (x, y_text_bg), (x + w, y), color, cv2.FILLED)
        
        # วาดชื่อ
        cv2.putText(frame, text_name, (x + 6, y - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        # วาด %
        cv2.putText(frame, text_conf, (x + 6, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # 5. คำนวณและแสดง FPS
    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # 6. แสดงผลลัพธ์
    cv2.imshow("Face Detection and Recognition (PRO)", frame)
    
    # 7. รอรับการกดปุ่ม 'q' เพื่อออก
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- จบ Loop ---
print("[INFO] ปิดโปรแกรม...")
cap.release()
cv2.destroyAllWindows()