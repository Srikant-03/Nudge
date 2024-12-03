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

# Define alarm sound
alarm_sound = "alarm.wav"  # Replace with the path to your .wav alarm sound file

# Load the alarm sound using pygame
alarm = pygame.mixer.Sound(alarm_sound)

# Function to play alarm sound
def play_alarm():
    if not pygame.mixer.get_busy():  # Only play if not already playing
        alarm.play()

# Function to stop the alarm sound
def stop_alarm():
    alarm.stop()

# Initialize state variables
drowsy_score = 0  # Score indicating drowsiness level
score_threshold = 50  # Threshold to trigger the alarm
decay_rate = 1  # How fast the score decays when not drowsy
increment_rate = 5  # How much the score increases per detection

print("Starting real-time detection. Press 'q' to quit.")

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to read from webcam.")
        break

    # Run inference on the frame
    results = model.predict(source=frame, imgsz=640, conf=0.25, device=0, save=False, save_txt=False)

    # Annotate frame with detection results
    drowsy_detected_now = False  # Flag for this frame's drowsy detection

    for result in results:
        for box in result.boxes:
            # Extract box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])  # Confidence
            cls = int(box.cls[0])  # Class index

            # Draw bounding box and label
            label = f"{class_labels[cls]}: {conf:.2f}"
            color = (0, 255, 0)  # Green for bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # If "drowsy" class is detected, update the flag
            if class_labels[cls] == 'drowsy' and conf > 0.5:
                drowsy_detected_now = True

    # Update the drowsy score
    if drowsy_detected_now:
        drowsy_score += increment_rate  # Increase the score when drowsy is detected
    else:
        drowsy_score = max(0, drowsy_score - decay_rate)  # Decrease the score (but keep it non-negative)

    # Check if the score exceeds the threshold
    if drowsy_score >= score_threshold:
        play_alarm()  # Play the alarm if score exceeds threshold
    else:
        stop_alarm()  # Stop the alarm if score drops below threshold

    # Display the frame
    cv2.putText(frame, f"Drowsy Score: {drowsy_score}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow("Driver Drowsiness Detection", frame)

    # Quit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
