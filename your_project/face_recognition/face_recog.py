import os
import cv2
import numpy as np
import pickle
import torch
import json
from facenet_pytorch import InceptionResnetV1, MTCNN

# Load video dictionary
video_dict_file = "video_dict.json"
def load_video_dict(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            return json.load(f)
    return {}
video_dict = load_video_dict(video_dict_file)

# Initialize MTCNN and InceptionResnetV1
mtcnn = MTCNN(keep_all=True)
resnet = InceptionResnetV1(pretrained='vggface2').eval()

# Function to detect and encode faces
def detect_and_encode(image):
    with torch.no_grad():
        boxes, _ = mtcnn.detect(image)
        if boxes is not None:
            faces = []
            for box in boxes:
                face = image[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
                if face.size == 0:
                    continue
                face = cv2.resize(face, (160, 160))
                face = np.transpose(face, (2, 0, 1)).astype(np.float32) / 255.0
                face_tensor = torch.tensor(face).unsqueeze(0)
                encoding = resnet(face_tensor).detach().numpy().flatten()
                faces.append((encoding, box))
            return faces
    return []

# Function to learn faces from a video
def learn_from_video(video_path, label, known_face_encodings, known_face_names):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    encodings = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_encodings = detect_and_encode(frame_rgb)

        if face_encodings:
            encodings.extend([enc[0] for enc in face_encodings])

        frame_count += 1
        if frame_count % 10 == 0:  # Process every 10th frame for efficiency
            print(f"Processed {frame_count} frames")

    cap.release()

    if encodings:
        mean_encoding = np.mean(encodings, axis=0)
        known_face_encodings.append(mean_encoding)
        known_face_names.append(label)

    return known_face_encodings, known_face_names

# Save encodings and names to a file
def save_encodings(file_path, encodings, names):
    with open(file_path, 'wb') as f:
        pickle.dump({'encodings': encodings, 'names': names}, f)

# Load encodings and names from a file
def load_encodings(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            return data['encodings'], data['names']
    return [], []

# Recognize faces in real-time
def recognize_faces(known_encodings, known_names, test_encodings, threshold=0.6):
    recognized_names = []
    for test_encoding in test_encodings:
        distances = np.linalg.norm(known_encodings - test_encoding, axis=1)
        min_distance_idx = np.argmin(distances)
        if distances[min_distance_idx] < threshold:
            recognized_names.append(known_names[min_distance_idx])
        else:
            recognized_names.append('Not Recognized')
    return recognized_names

# File path to save and load encodings
encodings_file = 'face_encodings.pkl'

# Load previously saved encodings
known_face_encodings, known_face_names = load_encodings(encodings_file)

# If new videos are available, train on them
for name, video_path in video_dict.items():
    if name not in known_face_names:
        print(f"Training on {name}'s video...")
        known_face_encodings, known_face_names = learn_from_video(video_path, name, known_face_encodings, known_face_names)

# Save the updated encodings
save_encodings(encodings_file, known_face_encodings, known_face_names)

# Start video capture for recognition
cap = cv2.VideoCapture(0)
threshold = 0.6

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    test_face_encodings = detect_and_encode(frame_rgb)

    if test_face_encodings and known_face_encodings:
        names = recognize_faces(np.array(known_face_encodings), known_face_names, [enc[0] for enc in test_face_encodings], threshold)
        for name, (_, box) in zip(names, test_face_encodings):
            (x1, y1, x2, y2) = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow('Face Recognition', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
