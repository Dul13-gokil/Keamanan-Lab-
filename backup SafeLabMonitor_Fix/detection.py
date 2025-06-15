import cv2
import torch
import mediapipe as mp
import os
import time
from datetime import datetime
import face_recognition
import numpy as np
import csv
from flask import Response, jsonify

# Gunakan GPU jika tersedia
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"[INFO] Menggunakan device: {device}")

# Muat wajah yang dikenal
known_face_encodings = []
known_face_names = []
for filename in os.listdir("known_faces"):
    if filename.endswith(('.jpg', '.png', '.jpeg')):
        image = face_recognition.load_image_file(f"known_faces/{filename}")
        encodings = face_recognition.face_encodings(image)
        if encodings:
            known_face_encodings.append(encodings[0])
            known_face_names.append(os.path.splitext(filename)[0])

# Buat folder dan file log jika belum ada
os.makedirs("bukti", exist_ok=True)
log_file = "log_mencurigakan.csv"
if not os.path.isfile(log_file):
    with open(log_file, mode='w', newline='') as f:
        csv.writer(f).writerow(["Waktu", "Nama", "Alasan"])

# Load model YOLOv5 hanya untuk deteksi orang
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True).to(device)
model.classes = [0]  # Hanya manusia

# Inisialisasi Mediapipe pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=0,
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Setup kamera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Variabel kontrol
last_save_time = 0
save_interval = 5  # Detik antar simpan
frame_count = 0
cache_expiry = 15
face_cache = []
face_cache_counter = 0
suspicious_events = []
last_suspicious_time = 0
suspicious_logged = False

def generate_frames():
    global frame_count, last_save_time, face_cache, face_cache_counter
    global suspicious_events, last_suspicious_time, suspicious_logged

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 480))
        clean_frame = frame.copy()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = model(frame)
        detections = results.xyxy[0].cpu().numpy()

        # Deteksi wajah setiap beberapa frame
        if frame_count % 5 == 0:
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
            new_face_cache = []

            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                name = "Unknown"
                distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                if len(distances) > 0:
                    best_match = np.argmin(distances)
                    if distances[best_match] < 0.5:
                        name = known_face_names[best_match]
                new_face_cache.append(((top, right, bottom, left), name))

            face_cache = new_face_cache
            face_cache_counter = 0
        else:
            face_cache_counter += 1
            if face_cache_counter >= cache_expiry:
                face_cache = []

        for det in detections:
            x1, y1, x2, y2, conf, cls = det.astype(int)

            # Abaikan jika terlalu kecil (kemungkinan hanya kepala atau bagian tubuh)
            if (y2 - y1) < 150:
                continue

            person_roi = frame[y1:y2, x1:x2]
            if person_roi.size == 0:
                continue

            roi_rgb = cv2.cvtColor(person_roi, cv2.COLOR_BGR2RGB)
            result = pose.process(roi_rgb)

            if result.pose_landmarks:
                landmarks = result.pose_landmarks.landmark
                mp_drawing.draw_landmarks(person_roi, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                frame[y1:y2, x1:x2] = person_roi

                # Landmark penting
                lw = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
                rw = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
                lh = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
                rh = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
                lk = landmarks[mp_pose.PoseLandmark.LEFT_KNEE]
                rk = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE]

                # Deteksi apakah membungkuk (hindari false positive)
                def is_bending(lk, rk, lh, rh):
                    return (abs(lk.y - lh.y) < 0.1 and abs(rk.y - rh.y) < 0.1)

                def is_carrying(w1, w2):
                    return abs(w1.x - w2.x) < 0.25 and abs(w1.y - w2.y) < 0.25

                def is_putting_into_pocket(wrist, hip):
                    return wrist.visibility > 0.7 and hip.visibility > 0.7 and abs(wrist.x - hip.x) < 0.1 and abs(wrist.y - hip.y) < 0.1

                visible_points = [lm for lm in landmarks if lm.visibility > 0.5]
                if len(visible_points) < 15:
                    continue

                # Cek gerakan mencurigakan
                reason = None
                if not is_bending(lk, rk, lh, rh):
                    if is_carrying(lw, rw):
                        is_suspicious = True
                        reason = "Mengangkat/Memindahkan barang"
                    elif is_putting_into_pocket(lw, lh) or is_putting_into_pocket(rw, rh):
                        is_suspicious = True
                        reason = "Memasukkan sesuatu kedalam saku"
                    else:
                        is_suspicious = False
                else:
                    is_suspicious = False

                if is_suspicious:
                    current_time = time.time()
                    if not suspicious_logged or current_time - last_suspicious_time > save_interval:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"bukti/bukti_{timestamp}.jpg"
                        cv2.imwrite(filename, clean_frame)

                        name_detected = "Unknown"
                        for (top, right, bottom, left), name in face_cache:
                            name_detected = name
                            break

                        alasan = f"Gerakan mencurigakan ({reason})" if reason else "Gerakan mencurigakan"
                        log_entry = [datetime.now().strftime("%Y-%m-%d %H:%M:%S"), name_detected, alasan]
                        suspicious_events.append({'time': log_entry[0], 'event': f"{name_detected} - {alasan}"})

                        with open(log_file, mode='a', newline='') as f:
                            csv.writer(f).writerow(log_entry)

                        last_suspicious_time = current_time
                        suspicious_logged = True
                else:
                    suspicious_logged = False

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

        for (top, right, bottom, left), name in face_cache:
            color = (0, 0, 255) if name == "Unknown" else (0, 255, 0)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        frame_count += 1

def get_recent_logs():
    return suspicious_events[-10:][::-1]

def get_status():
    return jsonify({
        'system': 'Aktif',
        'camera': 'Terhubung',
        'notification': 'Siaga'
    })