import face_recognition
import pickle
import os
import cv2

# --- Detector "Pro"  ---
PROTOTXT_PATH = "models/deploy.prototxt.txt"
MODEL_PATH = "models/res10_300x300_ssd_iter_140000.caffemodel"
net = cv2.dnn.readNetFromCaffe(PROTOTXT_PATH, MODEL_PATH)
MIN_CONFIDENCE = 0.5
# ------------------------------------------

ENCODINGS_PATH = "models/encodings.pickle"

#  ถามชื่อ 
name = input("ป้อนชื่อของคุณ (ภาษาอังกฤษ): ")
if not name:
    print("คุณไม่ได้ป้อนชื่อ, ยกเลิกโปรแกรม")
    exit()

#  โหลดฐานข้อมูลเดิม
knownEncodings = []
knownNames = []
if os.path.exists(ENCODINGS_PATH):
    print("[INFO] กำลังโหลดฐานข้อมูลเดิม...")
    try:
        with open(ENCODINGS_PATH, 'rb') as f:
            data = pickle.load(f)
            knownEncodings = data["encodings"]
            knownNames = data["names"]
    except Exception as e:
        print(f"[WARNING] ไม่สามารถโหลดฐานข้อมูลเดิมได้: {e}")

#  เปิดกล้อง 
cap = cv2.VideoCapture(0)
print("[INFO] เปิดกล้อง... กรุณามองกล้อง")
print("--- Press 's' to save your face (multiple times) ---")
print("--- Press 'q' to quit ---")

while True:
    ret, frame = cap.read()
    if not ret:
        print("[Error] ไม่สามารถอ่านภาพจากกล้องได้")
        break
    frame = cv2.flip(frame, 1)
    
    #  Detector หาใบหน้า 
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
        (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    best_face_box = None
    best_confidence = 0.0
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > MIN_CONFIDENCE and confidence > best_confidence:
            best_confidence = confidence
            box = detections[0, 0, i, 3:7] * [w, h, w, h]
            best_face_box = box.astype("int")

    #  วาดกรอบ 
    if best_face_box is not None:
        (startX, startY, endX, endY) = best_face_box
        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
        cv2.putText(frame, "Press 's' to save", (startX, startY - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Add New Face - เพิ่มใบหน้า", frame)
    key = cv2.waitKey(1) & 0xFF

    #  ถ้ากด 's' (Save)
    if key == ord('s'):
        if best_face_box is None:
            print("[Warning] ไม่พบใบหน้า, ไม่สามารถบันทึกได้")
            continue
            
        try:
            
            if frame.shape[2] == 4:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            else:
                frame_bgr = frame
            rgb_maybe_dirty = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            rgb = rgb_maybe_dirty.copy().astype('uint8')
            
            (startX, startY, endX, endY) = best_face_box
            box_for_encoding = [(startY, endX, endY, startX)]
            
            encoding = face_recognition.face_encodings(rgb, box_for_encoding)[0]
            
            knownEncodings.append(encoding)
            knownNames.append(name)
            
            print(f"[Success] บันทึกใบหน้า {name} สำเร็จ! (ตอนนี้มี {len(knownEncodings)} encodings)")
            
        except Exception as e:
            print(f"[Error] ไม่สามารถสร้าง encoding ได้: {e}")

    #  ถ้ากด 'q' (Quit)
    elif key == ord('q'):
        break

print("[INFO] กำลังบันทึกฐานข้อมูล...")
data_to_save = {"encodings": knownEncodings, "names": knownNames}
with open(ENCODINGS_PATH, "wb") as f:
    f.write(pickle.dumps(data_to_save))

print("[INFO] เสร็จสิ้น! ปิดโปรแกรม")
cap.release()
cv2.destroyAllWindows()