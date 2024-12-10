import cv2
import time
from ultralytics import YOLO
import pygame

# Initialize pygame mixer for playing sound
pygame.mixer.init()

# Load the trained YOLO model
model_path = "best_models/best (5).pt"  # Path to your trained model
model = YOLO(model_path)

# Class labels (update as per your dataset)
class_labels = ['LeftRight', 'child', 'drowsy', 'mobile', 'person', 'talking']  # Replace with actual class names

# Initialize webcam
webcam_index = 0  # Change if the webcam is not detected (0 is usually the default USB webcam)
cap = cv2.VideoCapture(webcam_index)

if not cap.isOpened():
    print("Error: Unable to access the webcam.")
    exit()

# Set video capture properties
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Set width
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  # Set height

# Define alarm sounds
drowsy_alarm_sound = "drowsy_alarm.wav"  # Replace with the path to your drowsy alarm sound file
mobile_alarm_sound = "mobile_alarm.wav"  # Replace with the path to your mobile alarm sound file

# Load alarm sounds using pygame
drowsy_alarm = pygame.mixer.Sound(drowsy_alarm_sound)
mobile_alarm = pygame.mixer.Sound(mobile_alarm_sound)

# Function to play alarm sound
def play_alarm(alarm):
    if not pygame.mixer.get_busy():  # Only play if not already playing
        alarm.play()

# Function to stop the alarm sound
def stop_alarm(alarm):
    alarm.stop()

# Initialize state variables
drowsy_score = 0  # Drowsiness score
mobile_score = 0  # Mobile usage score

# Tracking metrics
drowsy_alarm_count = 0
mobile_alarm_count = 0

# Thresholds and flags
drowsy_threshold = 30  # Threshold to trigger the drowsy alarm
mobile_threshold = 50  # Threshold to trigger the mobile alarm

# Decay rates and increment rates
decay_rate = 15
increment_rate = 10

# State tracking for alarms
prev_drowsy_state = False
prev_mobile_state = False
start_time = time.time()

print("Starting real-time detection. Press 'q' to quit.")

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to read from webcam.")
        break

    # Run inference on the frame
    results = model.predict(source=frame, imgsz=640, conf=0.20, device=0, save=False, save_txt=False)

    # Annotate frame with detection results
    drowsy_detected_now = False
    mobile_detected_now = False

    for result in results:
        for box in result.boxes:
            # Extract box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])  # Confidence
            cls = int(box.cls[0])  # Class index
            detected_label = class_labels[cls]

            # Check for specific detections and update flags
            if detected_label == 'drowsy' and conf > 0.4:
                drowsy_detected_now = True
            elif detected_label == 'mobile' and conf > 0.2:
                mobile_detected_now = True

    # Update drowsy score
    if drowsy_detected_now:
        drowsy_score += increment_rate
    else:
        drowsy_score = max(0, drowsy_score - decay_rate)

    # Update mobile score
    if mobile_detected_now:
        mobile_score += increment_rate
    else:
        mobile_score = max(0, mobile_score - decay_rate)

    # Handle drowsy alarm logic
    if drowsy_score >= drowsy_threshold:
        if not prev_drowsy_state:
            drowsy_alarm_count += 1
            play_alarm(drowsy_alarm)
        prev_drowsy_state = True
    else:
        if prev_drowsy_state:
            stop_alarm(drowsy_alarm)
        prev_drowsy_state = False

    # Handle mobile alarm logic
    if mobile_score >= mobile_threshold:
        if not prev_mobile_state:
            mobile_alarm_count += 1
            play_alarm(mobile_alarm)
        prev_mobile_state = True
    else:
        if prev_mobile_state:
            stop_alarm(mobile_alarm)
        prev_mobile_state = False

    # Calculate total driving time
    total_time = time.time() - start_time

    # Display scores and metrics
    cv2.putText(frame, f"Drowsy Score: {drowsy_score}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, f"Mobile Score: {mobile_score}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Drowsy Alarms: {drowsy_alarm_count}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(frame, f"Mobile Alarms: {mobile_alarm_count}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    cv2.putText(frame, f"Driving Time: {int(total_time)}s", (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("Driver Drowsiness Detection", frame)

    # Quit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
