import cv2
import mediapipe as mp
import pygame
import math
import csv
from datetime import datetime
import numpy as np
import pyttsx3
import time

# ---------------- Initialize ---------------- #
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1)

# Init pygame for beep
pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)

def generate_beep(frequency=1000, duration=2.0, volume=0.5):  # 2-second beep
    sample_rate = 44100
    n_samples = int(sample_rate * duration)
    t = np.linspace(0, duration, n_samples, False)
    tone = np.sin(frequency * 2 * np.pi * t)
    audio = np.int16(tone * 32767 * volume)
    sound = pygame.sndarray.make_sound(np.column_stack((audio, audio)))
    return sound

beep_sound = generate_beep()

# Init TTS
engine = pyttsx3.init()
engine.setProperty('rate', 170)
engine.setProperty('volume', 1.0)

def speak_alert(text):
    engine.say(text)
    engine.runAndWait()

# ---------------- Helper Functions ---------------- #
def dist(p1, p2):
    return math.dist(p1, p2)

def eye_aspect_ratio(landmarks, eye_indices):
    p = [landmarks[i] for i in eye_indices]
    A = dist(p[1], p[5])
    B = dist(p[2], p[4])
    C = dist(p[0], p[3])
    return (A + B) / (2.0 * C)

def mouth_aspect_ratio(landmarks, mouth_indices):
    p = [landmarks[i] for i in mouth_indices]
    A = dist(p[1], p[5])
    B = dist(p[2], p[4])
    C = dist(p[0], p[3])
    return (A + B) / (2.0 * C)

def log_event(frame_no, reason, ear, mar):
    with open("sleepiness_log.csv", mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                         frame_no, reason, f"{ear:.2f}", f"{mar:.2f}"])

# ---------------- Landmarks & Thresholds ---------------- #
LEFT_EYE = [33, 160, 158, 133, 153, 144, 163, 7]
RIGHT_EYE = [362, 385, 387, 263, 373, 380, 386, 249]
MOUTH = [61, 81, 13, 311, 14, 308, 78, 82, 312, 308]

EAR_THRESHOLD = 0.25
MAR_THRESHOLD = 0.65
CLOSED_FRAMES = 20      # for eyes closed
YAWN_FRAMES = 8         # for yawning (faster)

ear_buffer = []
mar_buffer = []
buffer_size = 5

counter = 0
reason = ""
frame_no = 0

# ---------------- Blink & Yawn Counters ---------------- #
ear_low_counter = 0
mar_high_counter = 0
MIN_CONSEC_FRAMES = 3

# ---------------- Calibration ---------------- #
calibration_frames = 0
CALIBRATION_PERIOD = 50
ear_calibration = []
mar_calibration = []

# ---------------- Alert Timing ---------------- #
last_alert_time = 0
ALERT_COOLDOWN = 6   # seconds between alarms

# ---------------- Webcam ---------------- #
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_no += 1
    frame = cv2.resize(frame, (640, 480))
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    if result.multi_face_landmarks:
        for face in result.multi_face_landmarks:
            landmarks = [(lm.x * w, lm.y * h) for lm in face.landmark]

            # Compute EAR & MAR
            left_ear = eye_aspect_ratio(landmarks, LEFT_EYE)
            right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE)
            ear = (left_ear + right_ear) / 2
            mar = mouth_aspect_ratio(landmarks, MOUTH)

            # Add to smoothing buffers
            ear_buffer.append(ear)
            mar_buffer.append(mar)
            if len(ear_buffer) > buffer_size:
                ear_buffer.pop(0)
            if len(mar_buffer) > buffer_size:
                mar_buffer.pop(0)

            smoothed_ear = sum(ear_buffer) / len(ear_buffer)
            smoothed_mar = sum(mar_buffer) / len(mar_buffer)

            # ---------------- Calibration Phase ---------------- #
            if calibration_frames < CALIBRATION_PERIOD:
                ear_calibration.append(smoothed_ear)
                mar_calibration.append(smoothed_mar)
                calibration_frames += 1
                cv2.putText(frame, "Calibrating... Please stay awake", (50, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                continue
            elif calibration_frames == CALIBRATION_PERIOD:
                EAR_THRESHOLD = min(0.3, sum(ear_calibration)/len(ear_calibration) * 0.8)
                MAR_THRESHOLD = max(0.65, sum(mar_calibration)/len(mar_calibration) * 1.2)
                calibration_frames += 1

            # ---------------- Blink/Yawn Filtering ---------------- #
            if smoothed_ear < EAR_THRESHOLD:
                ear_low_counter += 1
            else:
                ear_low_counter = 0

            if smoothed_mar > MAR_THRESHOLD:
                mar_high_counter += 1
            else:
                mar_high_counter = 0

            # Check sleepiness
            prolonged_sleep = False
            if ear_low_counter >= MIN_CONSEC_FRAMES:
                prolonged_sleep = True
                reason = "Eyes Closed"
            elif mar_high_counter >= MIN_CONSEC_FRAMES:
                prolonged_sleep = True
                reason = "Yawning"

            # ---------------- Main Counter ---------------- #
            if prolonged_sleep:
                counter += 1
            else:
                counter = 0
                reason = ""

            # ---------------- Alerts ---------------- #
            if (reason == "Eyes Closed" and counter > CLOSED_FRAMES) or \
               (reason == "Yawning" and counter > YAWN_FRAMES):

                cv2.putText(frame, f"⚠️ SLEEPINESS ALERT: {reason}", (50, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

                if time.time() - last_alert_time > ALERT_COOLDOWN:
                    # Play 2s beep
                    beep_sound.play()
                    time.sleep(2)

                    # Speak message
                    speak_alert("Stay alert, don't drift off!")

                    # Log event
                    log_event(frame_no, reason, smoothed_ear, smoothed_mar)
                    last_alert_time = time.time()

            elif counter > CLOSED_FRAMES // 2 and reason == "Eyes Closed":
                cv2.putText(frame, f"⚠️ Warning: {reason}", (50, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)

            # ---------------- Display Metrics ---------------- #
            cv2.putText(frame, f"EAR: {smoothed_ear:.2f}", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"MAR: {smoothed_mar:.2f}", (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    cv2.imshow("Sleepiness Detector (Eyes & Yawn)", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
