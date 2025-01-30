# utils/face_recognition.py
import os
import cv2
import numpy as np
import torch
import hashlib
from datetime import datetime
from facenet_pytorch import InceptionResnetV1, MTCNN

# Global flag that can be imported by other modules
FACE_RECOGNITION_ENABLED = False

# Learning cache dictionary
LEARNING_CACHE = {}

# Initialize models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
try:
    mtcnn = MTCNN(keep_all=True, device=device)
    resnet = InceptionResnetV1(pretrained='vggface2', device=device).eval()
    FACE_RECOGNITION_ENABLED = True
    print(f"Using device: {device}")
    print("Face recognition models loaded successfully")
except Exception as e:
    print(f"Error initializing face recognition models: {e}")
    mtcnn = None
    resnet = None

def get_video_hash(video_path):
    """Generate a hash of the video file based on modification time and size"""
    if not os.path.exists(video_path):
        return None
    stats = os.stat(video_path)
    hash_string = f"{video_path}_{stats.st_mtime}_{stats.st_size}"
    return hashlib.md5(hash_string.encode()).hexdigest()

def should_learn_face(username, video_path, cache_duration):
    """Check if we need to learn face encodings for this video"""
    current_time = datetime.now()
    video_hash = get_video_hash(video_path)
    if video_hash is None:
        return False
    if username in LEARNING_CACHE:
        cache_time, cached_hash = LEARNING_CACHE[username]
        if (current_time - cache_time) < cache_duration and cached_hash == video_hash:
            return False
    return True

def detect_and_encode(image):
    if not FACE_RECOGNITION_ENABLED:
        raise ValueError("Face recognition is not available")
        
    with torch.no_grad():
        try:
            boxes, _ = mtcnn.detect(image)
            if boxes is not None:
                faces = []
                for box in boxes:
                    if box is None:
                        continue
                    face = image[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
                    if face.size == 0:
                        continue
                    face = cv2.resize(face, (160, 160))
                    face = np.transpose(face, (2, 0, 1)).astype(np.float32) / 255.0
                    face_tensor = torch.tensor(face).unsqueeze(0).to(device)
                    encoding = resnet(face_tensor).cpu().numpy().flatten()
                    faces.append((encoding, box))
                return faces
        except Exception as e:
            print(f"Error in detect_and_encode: {e}")
            raise
    return []

def learn_from_video(video_path, username, encodings_file, cache_duration, load_encodings_fn, save_encodings_fn):
    """Learn face encodings from a user's video"""
    if not FACE_RECOGNITION_ENABLED:
        raise ValueError("Face recognition is not available")
        
    if not should_learn_face(username, video_path, cache_duration):
        print(f"Using cached encodings for user: {username}")
        return True
        
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
            
        encodings = []
        frame_count = 0
        
        while cap.isOpened() and frame_count < 30:
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_encodings = detect_and_encode(frame_rgb)
            if face_encodings:
                encodings.extend([enc[0] for enc in face_encodings])
            frame_count += 1
        cap.release()
        
        if not encodings:
            print(f"No faces found in video for user: {username}")
            return False
            
        mean_encoding = np.mean(encodings, axis=0)
        known_encodings, known_usernames = load_encodings_fn(encodings_file)
        
        if username in known_usernames:
            idx = known_usernames.index(username)
            known_encodings[idx] = mean_encoding
        else:
            known_encodings.append(mean_encoding)
            known_usernames.append(username)
            
        save_encodings_fn(known_encodings, known_usernames, encodings_file)
        LEARNING_CACHE[username] = (datetime.now(), get_video_hash(video_path))
        
        print(f"Successfully learned face encoding for user: {username}")
        return True
        
    except Exception as e:
        print(f"Error learning from video for user {username}: {e}")
        return False

def recognize_face(image, threshold, encodings_file, load_encodings_fn):
    if not FACE_RECOGNITION_ENABLED:
        raise ValueError("Face recognition is not available")
    
    known_encodings, usernames = load_encodings_fn(encodings_file)
    if not known_encodings:
        raise ValueError("No registered faces found in the system")

    face_encodings = detect_and_encode(image)
    if not face_encodings:
        return None

    test_encoding = face_encodings[0][0]
    distances = np.linalg.norm(np.array(known_encodings) - test_encoding, axis=1)
    min_distance_idx = np.argmin(distances)
    if distances[min_distance_idx] < threshold:
        return usernames[min_distance_idx]
    
    return None
