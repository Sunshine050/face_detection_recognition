from core.detector import detect_faces 

from core.recognizer import recognize_faces_lbph
import cv2
import time

print("[INFO] กำลังโหลดโมเดลและเปิดกล้อง...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("[ERROR] ไม่สามารถเปิดกล้องได้")
    exit()

prev_frame_time = 0
new_frame_time = 0

print("[INFO] เริ่มการทำงาน... กด 'q' เพื่อออกจากโปรแกรม")

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] ไม่สามารถรับเฟรมภาพได้")
        break
        
    frame = cv2.flip(frame, 1)
    
    detected_results = detect_faces(frame)
    
    boxes = []
    detector_confidences = [] 
    for (box, conf) in detected_results:
        boxes.append(box)
        detector_confidences.append(conf)
    names, lbph_confidences = recognize_faces_lbph(frame, boxes)
    
    for i, (x, y, w, h) in enumerate(boxes):
        name = names[i]
        
        text_name = f"Name: {name}"
        text_lbph_conf = f"Match: {lbph_confidences[i]:.2f}" 
        
        color = (0, 0, 255) if name == "Unknown" else (0, 255, 0)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        
        y_text_bg = y - 40 if y - 40 > 0 else y + 10
        cv2.rectangle(frame, (x, y_text_bg), (x + w, y), color, cv2.FILLED)
        
        cv2.putText(frame, text_name, (x + 6, y - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, text_lbph_conf, (x + 6, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Face Detection (DNN) + Recognition (LBPH)", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- จบ Loop ---
print("[INFO] ปิดโปรแกรม...")
cap.release()
cv2.destroyAllWindows()