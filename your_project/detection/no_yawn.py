import cv2
import time
from ultralytics import YOLO
import dlib
from imutils import face_utils
from scipy.spatial import distance as dist
import numpy as np
import threading

class DetectionProcessor:
    def __init__(self):
        self.frame = None
        self.running = True
        self._camera_connected = False
        self._lock = threading.Lock()  # Add thread safety
        
        # Add to DetectionProcessor class
        @property
        def start_time(self):
            with self._lock:
                return self._start_time

        @start_time.setter
        def start_time(self, value):
            with self._lock:
                self._start_time = value
        
        try:
            # Detection components
            self.model = YOLO("best_models/best (5).pt")
            self.detector = dlib.get_frontal_face_detector()
            self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
            
            # Video capture
            # In DetectionProcessor __init__ method
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
            # Try different camera indexes
                self.cap = cv2.VideoCapture(1)
            if not self.cap.isOpened():
                print("Error: Could not open any camera")
                self._camera_connected = False
            else:
                self._camera_connected = True
    # Set camera properties after successful open
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            
            # Detection parameters
            self.EYE_AR_THRESH = 0.3
            self.EYE_AR_CONSEC_FRAMES = 2
            self.drowsy_threshold = 30
            self.mobile_threshold = 50
            self.decay_rate = 15
            self.increment_rate = 10
            
            # State variables
            self.reset_state()
            
        except Exception as e:
            print(f"Initialization error: {str(e)}")
            self.running = False
            raise

    def reset_state(self):
        """Reset all state variables to their initial values."""
        with self._lock:
            self.drowsy_score = 0
            self.mobile_score = 0
            self.drowsy_alarm_count = 0
            self.mobile_alarm_count = 0
            self.prev_drowsy_state = False
            self.prev_mobile_state = False
            self.blink_counter = 0
            self.total_blinks = 0
            self.start_time = time.time()
            self.drowsy_alarm = False
            self.mobile_alarm = False

    def is_camera_connected(self):
        """Check if the camera is connected and working."""
        return self._camera_connected

    def get_metrics(self):
        """Get current detection metrics."""
        with self._lock:
            return {
                'drowsy_score': self.drowsy_score,
                'mobile_score': self.mobile_score,
                'drowsy_alarms': self.drowsy_alarm_count,
                'mobile_alarms': self.mobile_alarm_count,
                'total_blinks': self.total_blinks,
                'driving_time': int(time.time() - self.start_time)
            }

    @staticmethod
    def eye_aspect_ratio(eye):
        """Calculate the eye aspect ratio."""
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        C = dist.euclidean(eye[0], eye[3])
        return (A + B) / (2.0 * C)

    def process_eyes(self, frame):
        """Process eye detection and tracking."""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rects = self.detector(gray, 0)
            (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
            (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

            for rect in rects:
                shape = self.predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)

                left_eye = shape[lStart:lEnd]
                right_eye = shape[rStart:rEnd]
                left_ear = self.eye_aspect_ratio(left_eye)
                right_ear = self.eye_aspect_ratio(right_eye)
                ear = (left_ear + right_ear) / 2.0

                with self._lock:
                    if ear < self.EYE_AR_THRESH:
                        self.blink_counter += 1
                    else:
                        if self.blink_counter >= self.EYE_AR_CONSEC_FRAMES:
                            self.total_blinks += 1
                        self.blink_counter = 0

                # Draw eye contours
                left_eye_hull = cv2.convexHull(left_eye)
                right_eye_hull = cv2.convexHull(right_eye)
                cv2.drawContours(frame, [left_eye_hull], -1, (0, 255, 0), 1)
                cv2.drawContours(frame, [right_eye_hull], -1, (0, 255, 0), 1)

            return frame
        except Exception as e:
            print(f"Eye processing error: {str(e)}")
            return frame

    def update_scores(self, drowsy_detected, mobile_detected):
        """Update detection scores and alarm states."""
        with self._lock:
            # Update drowsy score
            if drowsy_detected:
                self.drowsy_score = min(self.drowsy_score + self.increment_rate, 100)
            else:
                self.drowsy_score = max(self.drowsy_score - self.decay_rate, 0)

            # Update mobile score
            if mobile_detected:
                self.mobile_score = min(self.mobile_score + self.increment_rate, 100)
            else:
                self.mobile_score = max(self.mobile_score - self.decay_rate, 0)

            # Update alarm states
            self.drowsy_alarm = self.drowsy_score >= self.drowsy_threshold
            self.mobile_alarm = self.mobile_score >= self.mobile_threshold

            # Update alarm counts
            if self.drowsy_alarm and not self.prev_drowsy_state:
                self.drowsy_alarm_count += 1
            if self.mobile_alarm and not self.prev_mobile_state:
                self.mobile_alarm_count += 1

            self.prev_drowsy_state = self.drowsy_alarm
            self.prev_mobile_state = self.mobile_alarm

    def draw_annotations(self, frame):
        """Draw detection information on the frame."""
        metrics = self.get_metrics()
        annotations = [
            (f"Drowsy Score: {metrics['drowsy_score']:.1f}", (10, 30)),
            (f"Drowsy Alarms: {metrics['drowsy_alarms']}", (10, 70)),
            (f"Blinks: {metrics['total_blinks']}", (10, 110)),
            (f"Mobile Alarms: {metrics['mobile_alarms']}", (10, 150)),
            (f"Driving Time: {metrics['driving_time']}s", (10, 190))
        ]

        for text, pos in annotations:
            cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 1, 
                       (255, 255, 255) if "Driving Time" in text else (0, 0, 255), 2)
        return frame

    def process_frame(self, frame):
        """Process a single frame for all detections."""
        try:
            frame = self.process_eyes(frame)
            
            # YOLO detection
            results = self.model.predict(source=frame, imgsz=640, conf=0.20, device=0)
            drowsy_detected = False
            mobile_detected = False

            for result in results:
                for box in result.boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    label = result.names[cls]

                    if label == 'drowsy' and conf > 0.4:
                        drowsy_detected = True
                    elif label == 'mobile' and conf > 0.2:
                        mobile_detected = True

            self.update_scores(drowsy_detected, mobile_detected)
            return self.draw_annotations(frame)
        except Exception as e:
            print(f"Frame processing error: {str(e)}")
            return frame

    def run(self):
        while self.running and self._camera_connected:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to grab frame. Retrying...")
                # Attempt to reconnect the camera
                self.cap.release()
                self.cap = cv2.VideoCapture(0)
                if not self.cap.isOpened():
                    print("Camera reconnection failed. Exiting detection loop.")
                    self._camera_connected = False
                    break
                continue

            print("Frame captured successfully.")
        
            # Process the frame (eye detection, YOLO, etc.)
            processed_frame = self.process_frame(frame)
            if processed_frame is None:
                print("Processed frame is None, skipping update.")
                continue

            with self._lock:
                self.frame = processed_frame

            print("Frame processed and updated at", time.time())
            time.sleep(0.033)  # ~30 FPS

        self.cleanup()


    def cleanup(self):
        """Clean up resources."""
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()
        self._camera_connected = False

    def stop(self):
        """Stop the detection process."""
        self.running = False
        time.sleep(0.1)  # Give time for the run loop to finish
        self.cleanup()