import cv2
import os
import sys

# รับชื่อจาก argument เช่น: python add_face_cv.py Thinnakorn
if len(sys.argv) < 2:
    print("Usage: python add_face_cv.py <name>")
    sys.exit(1)

name = sys.argv[1]
dataset_dir = os.path.join("dataset", name)

if not os.path.exists(dataset_dir):
    os.makedirs(dataset_dir)

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

count = 0
print(f"[INFO] เริ่มจับภาพสำหรับ {name}... กด 'q' เพื่อออก")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face_roi = gray[y:y+h, x:x+w]
        face_roi = cv2.resize(face_roi, (200, 200))
        count += 1
        file_path = os.path.join(dataset_dir, f"{count}.png")
        cv2.imwrite(file_path, face_roi)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f"{count}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        print(f"[INFO] Saved {file_path}")

    cv2.imshow("Collecting Faces", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    elif count >= 20:  # เก็บ 20 ภาพ
        break

cap.release()
cv2.destroyAllWindows()
print(f"[INFO] เก็บภาพ {count} ภาพในโฟลเดอร์ {dataset_dir}")
