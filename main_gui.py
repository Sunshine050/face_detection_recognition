import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import time
import numpy as np
import os 
import sys

# --- (START) การจัดการพาธให้เข้าถึง core modules ---
# เพิ่ม sys.path เพื่อให้แน่ใจว่ามันหา 'core' เจอ (เช่น detector.py)
try:
    BASE_DIR = os.getcwd()
except Exception as e:
    print(f"[ERROR] Cannot get Current Working Directory: {e}", file=sys.stderr)
    sys.exit(1)

CORE_PATH = os.path.join(BASE_DIR, "core")
if CORE_PATH not in sys.path:
    sys.path.append(CORE_PATH)
# --- (END) การจัดการพาธ ---

# นำเข้าฟังก์ชันจากไฟล์อื่น ๆ ในโปรเจกต์
try:
    # ต้อง import จาก 'detector' และ 'recognizer' ที่อยู่ในโฟลเดอร์ core/
    from detector import detect_faces 
    from recognizer import recognize_faces_lbph # ใช้ฟังก์ชัน recognize_faces_lbph
except ImportError:
    # พิมพ์คำเตือนหากหาไฟล์ไม่เจอ
    print("[ERROR] ไม่สามารถ import 'core' modules ได้", file=sys.stderr)
    print(f"  > กำลังค้นหาที่: {CORE_PATH}", file=sys.stderr)
    print("  > กรุณาตรวจสอบว่ามีไฟล์ detector.py และ recognizer.py ในโฟลเดอร์ core/", file=sys.stderr)
    sys.exit(1)

# --- ค่าคงที่/การตั้งค่า ---
APP_TITLE = "Face Recognition Application (GUI)"
CAMERA_ID = 0 
UPDATE_DELAY_MS = 15 

class FaceRecognitionApp:
    def __init__(self, master):
        self.master = master
        master.title(APP_TITLE)
        
        # 1. การตั้งค่าตัวแปรสถานะ
        self.cap = None 
        self.is_running_camera = False
        self.prev_frame_time = time.time()
        self._after_id = None 

        # 2. สร้าง Main Frame
        main_frame = tk.Frame(master)
        main_frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        # 3. พื้นที่แสดงผลภาพ/วิดีโอ 
        self.video_label = tk.Label(main_frame, text="Select an option to start.", 
                                      bg='black', fg='white', relief=tk.RAISED)
        self.video_label.pack(side=tk.LEFT, padx=10, fill=tk.BOTH, expand=True)
        
        # 4. Control Panel (ปุ่มต่าง ๆ)
        control_frame = tk.Frame(main_frame)
        control_frame.pack(side=tk.RIGHT, padx=10, fill=tk.Y)

        # ปุ่ม Start Camera
        self.btn_camera = tk.Button(control_frame, text="🔴 Start Camera", command=self.start_camera, width=20)
        self.btn_camera.pack(pady=10)
        
        # ปุ่ม Stop Camera
        self.btn_stop = tk.Button(control_frame, text="◼ Stop Camera", command=self.stop_camera, width=20, state=tk.DISABLED)
        self.btn_stop.pack(pady=10)

        # ปุ่ม Upload Image
        self.btn_upload = tk.Button(control_frame, text="🖼️ Upload Image", command=self.upload_image, width=20)
        self.btn_upload.pack(pady=10)
        
        # Label แสดงสถานะ
        self.status_label = tk.Label(control_frame, text="Status: Ready", fg="blue", wraplength=150)
        self.status_label.pack(pady=20, fill=tk.X)
        
        # ผูกฟังก์ชัน On Close
        master.protocol("WM_DELETE_WINDOW", self.on_close)

    # --- ฟังก์ชันควบคุม ---

    def start_camera(self):
        """เริ่มต้นการจับภาพจากกล้องวิดีโอ"""
        if self.is_running_camera:
            return

        self.cap = cv2.VideoCapture(CAMERA_ID)
        if not self.cap.isOpened():
            self.status_label.config(text="Status: ERROR - Cannot open camera", fg="red")
            return

        self.is_running_camera = True
        self.btn_camera.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.NORMAL)
        self.btn_upload.config(state=tk.DISABLED)
        self.status_label.config(text="Status: Camera Running...", fg="green")
        self.prev_frame_time = time.time()
        self.update_frame() 

    def stop_camera(self):
        """หยุดการทำงานของกล้อง"""
        if self.is_running_camera and self.cap:
            self.cap.release()
        
        self.is_running_camera = False
        self.btn_camera.config(state=tk.NORMAL)
        self.btn_stop.config(state=tk.DISABLED)
        self.btn_upload.config(state=tk.NORMAL)
        self.status_label.config(text="Status: Camera Stopped", fg="blue")
        
        if self._after_id:
            self.master.after_cancel(self._after_id) 
            self._after_id = None

    def upload_image(self):
        """อนุญาตให้ผู้ใช้อัปโหลดไฟล์ภาพเพื่อประมวลผล"""
        if self.is_running_camera:
            self.stop_camera()

        filepath = filedialog.askopenfilename(
            title="Select an Image File",
            filetypes=(("Image files", "*.jpg *.jpeg *.png"), ("All files", "*.*"))
        )
        
        if filepath:
            self.status_label.config(text=f"Status: Processing {os.path.basename(filepath)}", fg="darkorange")
            image = cv2.imread(filepath)
            if image is not None:
                # ส่งภาพไปประมวลผล
                processed_frame = self._process_frame(image, is_camera=False)
                # แสดงผลใน GUI
                self._display_frame(processed_frame)
                self.status_label.config(text="Status: Image Processed", fg="green")
            else:
                self.status_label.config(text="Status: ERROR - Could not load image", fg="red")
        else:
            self.status_label.config(text="Status: Ready", fg="blue")

    # --- ฟังก์ชันการประมวลผลภาพหลัก ---

    def _process_frame(self, frame, is_camera=True):
        """ฟังก์ชันหลักในการตรวจจับและจดจำใบหน้า พร้อมแสดงผลใน Terminal"""
        
        if is_camera:
            frame = cv2.flip(frame, 1) 
        
        # 1. ตรวจจับใบหน้า
        detected_results = detect_faces(frame)
        boxes = [box for (box, conf) in detected_results]
        
        # 2. จดจำใบหน้า
        if boxes:
            names, lbph_confidences = recognize_faces_lbph(frame, boxes)
        else:
            names = []
            lbph_confidences = []

        
        # --- แสดงผลใน Terminal (สำหรับ Debug) ---
        if names:
            print("\n--- Detection Results ---")
            for i in range(len(names)):
                name = names[i]
                conf = lbph_confidences[i]
                match_status = f"Match: {conf:.2f}" 
                if name != "Unknown":
                    print(f"✅ Found: {name} | Confidence (Lower is better): {conf:.2f}")
                else:
                    print(f"❓ Unknown | Confidence: {conf:.2f}")
            print("-------------------------")
        
        # --- วาดผลลัพธ์ลงบนภาพ ---
        for i, (x, y, w, h) in enumerate(boxes):
            name = names[i]
            
            text_name = name
            text_lbph_conf = f"Match: {lbph_confidences[i]:.2f}" 
            
            color = (0, 0, 255) if name == "Unknown" else (0, 255, 0) # BGR
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
            # พื้นหลังข้อความ
            y_text_bg = y - 40 if y - 40 > 0 else y + 10
            cv2.rectangle(frame, (x, y_text_bg), (x + w, y), color, cv2.FILLED)
            
            cv2.putText(frame, text_name, (x + 6, y - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(frame, text_lbph_conf, (x + 6, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        return frame

    def _display_frame(self, frame):
        """แปลงและแสดงเฟรมภาพใน Tkinter Label พร้อมปรับขนาด"""
        
        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        img_h, img_w, _ = cv2image.shape
        label_w = self.video_label.winfo_width()
        label_h = self.video_label.winfo_height()

        if label_w <= 1 or label_h <= 1: 
             scale_factor = 1 
        else:
             scale_factor = min(label_w / img_w, label_h / img_h)

        new_w = int(img_w * scale_factor)
        new_h = int(img_h * scale_factor)
        
        cv2image = cv2.resize(cv2image, (new_w, new_h), interpolation=cv2.INTER_AREA)

        img = Image.fromarray(cv2image)
        self.photo = ImageTk.PhotoImage(image=img)
        
        self.video_label.config(image=self.photo, text="")
        self.video_label.image = self.photo 

    # --- Loop สำหรับกล้อง ---

    def update_frame(self):
        """Loop สำหรับดึงเฟรมจากกล้องและอัปเดต GUI"""
        if self.is_running_camera and self.cap:
            ret, frame = self.cap.read()
            if ret:
                # 1. ประมวลผลภาพ 
                processed_frame = self._process_frame(frame, is_camera=True)
                
                # คำนวณ FPS
                new_frame_time = time.time()
                try:
                    fps = 1 / (new_frame_time - self.prev_frame_time)
                except ZeroDivisionError:
                    fps = 0
                self.prev_frame_time = new_frame_time
                cv2.putText(processed_frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                # 2. แสดงผลใน GUI
                self._display_frame(processed_frame)
            
            # เรียกตัวเองซ้ำหลังจากหน่วงเวลา
            self._after_id = self.master.after(UPDATE_DELAY_MS, self.update_frame)
            
    def on_close(self):
        """ฟังก์ชันที่ถูกเรียกเมื่อผู้ใช้กดปิดหน้าต่าง"""
        print("[INFO] Closing application...")
        self.stop_camera() 
        self.master.destroy() 

if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("1000x700") 
    app = FaceRecognitionApp(root)
    root.mainloop()
