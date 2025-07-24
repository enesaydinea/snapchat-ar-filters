import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
from threading import Thread

# Efekt görsellerini yükle
effects = [
    cv2.imread("gozluk1.png", cv2.IMREAD_UNCHANGED),
    cv2.imread("gozluk2.png", cv2.IMREAD_UNCHANGED),
    cv2.imread("emoji.png", cv2.IMREAD_UNCHANGED),
    cv2.imread("biyik.png", cv2.IMREAD_UNCHANGED)
]
selected_index = 0
stop_flag = False

# MediaPipe tanımlamaları
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

def run_camera():
    global selected_index, stop_flag

    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        if stop_flag:
            break

        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            for landmarks in results.multi_face_landmarks:
                left_eye = landmarks.landmark[33]
                right_eye = landmarks.landmark[263]

                x1 = int(left_eye.x * w)
                y1 = int(left_eye.y * h)
                x2 = int(right_eye.x * w)
                y2 = int(right_eye.y * h)

                effect = effects[selected_index]
                if effect is None:
                    continue

                if selected_index in [0, 1]:  # Gözlük
                    effect_width = int(1.6 * abs(x2 - x1))
                    effect_height = int(effect_width * effect.shape[0] / effect.shape[1])
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    top_left_x = center_x - effect_width // 2
                    top_left_y = center_y - int(0.45 * effect_height)

                elif selected_index == 2:  # Emoji
                    x_min = int(landmarks.landmark[234].x * w) - 30
                    y_min = int(landmarks.landmark[10].y * h) - 80
                    x_max = int(landmarks.landmark[454].x * w) + 30
                    y_max = int(landmarks.landmark[152].y * h) + 60
                    effect_width = x_max - x_min
                    effect_height = y_max - y_min
                    top_left_x = x_min
                    top_left_y = y_min

                elif selected_index == 3:  # Bıyık
                    nose = landmarks.landmark[1]
                    nose_x = int(nose.x * w)
                    nose_y = int(nose.y * h)
                    effect_width = int(1.8 * abs(x2 - x1))
                    effect_height = int(effect_width * effect.shape[0] / effect.shape[1])
                    top_left_x = nose_x - effect_width // 2 + 1
                    top_left_y = nose_y - int(0.15 * effect_height) + 15

                resized = cv2.resize(effect, (effect_width, effect_height))

                for i in range(effect_height):
                    for j in range(effect_width):
                        if 0 <= top_left_y + i < h and 0 <= top_left_x + j < w:
                            alpha = resized[i, j, 3] / 255.0
                            for c in range(3):
                                frame[top_left_y + i, top_left_x + j, c] = \
                                    (1 - alpha) * frame[top_left_y + i, top_left_x + j, c] + alpha * resized[i, j, c]

        cv2.imshow("Kamera", frame)

        # ESC tuşu ile çıkış
        if cv2.waitKey(1) & 0xFF == 27:
            stop_flag = True
            break

        # Pencere çarpı tuşuna basılmışsa çık
        if cv2.getWindowProperty("Kamera", cv2.WND_PROP_VISIBLE) < 1:
            stop_flag = True
            break

    cap.release()
    cv2.destroyAllWindows()

# GUI fonksiyonları
def select_effect(index):
    global selected_index
    selected_index = index

def on_close():
    global stop_flag
    stop_flag = True
    root.destroy()

# GUI arayüzü
root = tk.Tk()
root.title("Efekt Seçimi")
root.protocol("WM_DELETE_WINDOW", on_close)

tk.Button(root, text="Gözlük 1", width=20, command=lambda: select_effect(0)).pack(pady=5)
tk.Button(root, text="Gözlük 2", width=20, command=lambda: select_effect(1)).pack(pady=5)
tk.Button(root, text="Emoji", width=20, command=lambda: select_effect(2)).pack(pady=5)
tk.Button(root, text="Bıyık", width=20, command=lambda: select_effect(3)).pack(pady=5)

# Kamera thread
import threading
camera_thread = threading.Thread(target=run_camera)
camera_thread.start()

# GUI döngüsü
root.mainloop()

# Thread bitince GUI kapandıktan sonra OpenCV penceresi kapanır
stop_flag = True

