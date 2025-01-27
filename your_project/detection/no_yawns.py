import cv2
import time
from ultralytics import YOLO
import pygame
from scipy.spatial import distance as dist
from imutils import face_utils
import numpy as np
import dlib

# Initialize pygame mixer for playing sound
pygame.mixer.init()

# Load the trained YOLO model
model_path = "best_models/best (5).pt"  # Path to your trained model
model = YOLO(model_path)

# Class labels (update as per your dataset)
class_labels = ['LeftRight', 'child', 'drowsy', 'mobile', 'person', 'talking']

# Initialize webcam
webcam_index = 0  # Change if the webcam is not detected (0 is usually the default USB webcam)
cap = cv2.VideoCapture(webcam_index)

if not cap.isOpened():
    print("Error: Unable to access the webcam.")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1280)

# Define alarm sounds
drowsy_alarm_sound = "drowsy_alarm.wav"
mobile_alarm_sound = "mobile_alarm.wav"

drowsy_alarm = pygame.mixer.Sound(drowsy_alarm_sound)
mobile_alarm = pygame.mixer.Sound(mobile_alarm_sound)

def play_alarm(alarm):
    if not pygame.mixer.get_busy():
        alarm.play()

def stop_alarm(alarm):
    alarm.stop()

# Initialize state variables
drowsy_score = 0
mobile_score = 0
drowsy_alarm_count = 0
mobile_alarm_count = 0
drowsy_threshold = 30
mobile_threshold = 50
decay_rate = 15
increment_rate = 10
prev_drowsy_state = False
prev_mobile_state = False
start_time = time.time()

# Blink detection setup
EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 2
blink_counter = 0
total_blinks = 0

print("[INFO] Loading facial landmark predictor...")
shape_predictor_path = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shape_predictor_path)
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

print("Starting real-time detection. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to read from webcam.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)

    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0

        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        if ear < EYE_AR_THRESH:
            blink_counter += 1
        else:
            if blink_counter >= EYE_AR_CONSEC_FRAMES:
                total_blinks += 1
            blink_counter = 0

    results = model.predict(source=frame, imgsz=640, conf=0.20, device=0, save=False, save_txt=False)
    drowsy_detected_now = False
    mobile_detected_now = False

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            detected_label = class_labels[cls]

            if detected_label == 'drowsy' and conf > 0.4:
                drowsy_detected_now = True
            elif detected_label == 'mobile' and conf > 0.2:
                mobile_detected_now = True

    if drowsy_detected_now:
        drowsy_score += increment_rate
    else:
        drowsy_score = max(0, drowsy_score - decay_rate)

    if mobile_detected_now:
        mobile_score += increment_rate
    else:
        mobile_score = max(0, mobile_score - decay_rate)

    if drowsy_score >= drowsy_threshold:
        if not prev_drowsy_state:
            drowsy_alarm_count += 1
            play_alarm(drowsy_alarm)
        prev_drowsy_state = True
    else:
        if prev_drowsy_state:
            stop_alarm(drowsy_alarm)
        prev_drowsy_state = False

    if mobile_score >= mobile_threshold:
        if not prev_mobile_state:
            mobile_alarm_count += 1
            play_alarm(mobile_alarm)
        prev_mobile_state = True
    else:
        if prev_mobile_state:
            stop_alarm(mobile_alarm)
        prev_mobile_state = False

    total_time = time.time() - start_time

    cv2.putText(frame, f"Drowsy Score: {drowsy_score}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, f"Drowsy Alarms: {drowsy_alarm_count}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(frame, f"Mobile Alarms: {mobile_alarm_count}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    cv2.putText(frame, f"Blinks: {total_blinks}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    cv2.putText(frame, f"Driving Time: {int(total_time)}s", (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imshow("Driver Drowsiness Detection", frame)
    

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()

cv2.destroyAllWindows()
