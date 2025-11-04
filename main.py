import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import time
import numpy as np
import os
import sys

from core.detector import detect_faces
from dataset.Thinnakorn.recognizer import recognize_faces_lbph

APP_TITLE = "Face Recognition Application (GUI)"
CAMERA_ID = 0
UPDATE_DELAY_MS = 15

class FaceRecognitionApp:
    def __init__(self, master):
        self.master = master
        master.title(APP_TITLE)
        self.cap = None
        self.is_running_camera = False
        self.prev_frame_time = time.time()

        main_frame = tk.Frame(master)
        main_frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        self.video_label = tk.Label(main_frame, text="Select an option to start.",
                                    bg='black', fg='white', relief=tk.RAISED)
        self.video_label.pack(side=tk.LEFT, padx=10, fill=tk.BOTH, expand=True)

        control_frame = tk.Frame(main_frame)
        control_frame.pack(side=tk.RIGHT, padx=10, fill=tk.Y)

        self.btn_camera = tk.Button(control_frame, text="🔴 Start Camera", command=self.start_camera, width=20)
        self.btn_camera.pack(pady=10)

        self.btn_stop = tk.Button(control_frame, text="◼ Stop Camera", command=self.stop_camera, width=20, state=tk.DISABLED)
        self.btn_stop.pack(pady=10)

        self.btn_upload = tk.Button(control_frame, text="🖼️ Upload Image", command=self.upload_image, width=20)
        self.btn_upload.pack(pady=10)

        self.status_label = tk.Label(control_frame, text="Status: Ready", fg="blue", wraplength=150)
        self.status_label.pack(pady=20, fill=tk.X)

    def start_camera(self):
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
        if self.is_running_camera and self.cap:
            self.cap.release()
        self.is_running_camera = False
        self.btn_camera.config(state=tk.NORMAL)
        self.btn_stop.config(state=tk.DISABLED)
        self.btn_upload.config(state=tk.NORMAL)
        self.status_label.config(text="Status: Camera Stopped", fg="blue")
        if hasattr(self, "_after_id"):
            self.master.after_cancel(self._after_id)

    def upload_image(self):
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
                processed_frame = self._process_frame(image, is_camera=False)
                self._display_frame(processed_frame)
                self.status_label.config(text="Status: Image Checked", fg="green")
            else:
                self.status_label.config(text="Status: ERROR - Could not load image", fg="red")
        else:
            self.status_label.config(text="Status: Ready", fg="blue")

    def _process_frame(self, frame, is_camera=True):
        if is_camera:
            frame = cv2.flip(frame, 1)
        detected_results = detect_faces(frame)
        boxes = [box for (box, conf) in detected_results]

        if boxes:
            names, lbph_confidences = recognize_faces_lbph(frame, boxes)
        else:
            names, lbph_confidences = [], []

        for i, (x, y, w, h) in enumerate(boxes):
            name = names[i]
            conf = lbph_confidences[i]
            if name != "Unknown":
                text_name = f"Matched: {name}"
                color = (0, 255, 0)
            else:
                text_name = "Unknown (No match found)"
                color = (0, 0, 255)

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.rectangle(frame, (x, y - 45 if y - 45 > 0 else y + 10), (x + w, y), color, cv2.FILLED)
            cv2.putText(frame, text_name, (x + 6, y - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            cv2.putText(frame, f"Conf: {conf:.2f}", (x + 6, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        if not boxes and is_camera:
            cv2.putText(frame, "No face detected / Not Matched", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return frame

    def _display_frame(self, frame):
        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_h, img_w, _ = cv2image.shape
        label_w = self.video_label.winfo_width()
        label_h = self.video_label.winfo_height()
        if label_w == 1 or label_h == 1:
            scale_factor = 1
        else:
            scale_factor = min(label_w / img_w, label_h / img_h)
        new_w, new_h = int(img_w * scale_factor), int(img_h * scale_factor)
        cv2image = cv2.resize(cv2image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        img = Image.fromarray(cv2image)
        self.photo = ImageTk.PhotoImage(image=img)
        self.video_label.config(image=self.photo, text="")
        self.video_label.image = self.photo

    def update_frame(self):
        if self.is_running_camera and self.cap:
            ret, frame = self.cap.read()
            if ret:
                processed_frame = self._process_frame(frame, is_camera=True)
                new_frame_time = time.time()
                fps = 1 / (new_frame_time - self.prev_frame_time)
                self.prev_frame_time = new_frame_time
                cv2.putText(processed_frame, f"FPS: {int(fps)}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                self._display_frame(processed_frame)
            self._after_id = self.master.after(UPDATE_DELAY_MS, self.update_frame)

if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("1000x700")
    app = FaceRecognitionApp(root)
    root.mainloop()
