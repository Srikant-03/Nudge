from flask import Flask, render_template, request, redirect, url_for, jsonify, session, Response
import json
import os
import subprocess
import cv2
import pickle
import numpy as np
from facenet_pytorch import InceptionResnetV1, MTCNN
import torch
from werkzeug.utils import secure_filename
from datetime import datetime, timedelta
import hashlib


app = Flask(__name__)
app.secret_key = os.urandom(24)

def load_users():
    if os.path.exists("users.json"):
        try:
            with open("users.json", "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return {}
    return {}  # If the file doesn't exist, return an empty dictionary

def save_users(data):
    with open("users.json", "w") as f:
        json.dump(data, f, indent=4)

@app.route("/")
def main():
    return render_template("main.html")

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        license_number = request.form["license"]
        password = request.form["password"]
        confirm_password = request.form["confirm_password"]
        
        users = load_users()
        
        if license_number in users:
            return "User already exists. Try logging in."

        if password != confirm_password:
            return "Passwords do not match."

        users[license_number] = {"password": password, "face_data": None}
        save_users(users)

        return redirect(url_for("face_signup"))
    return render_template("signup.html")

@app.route("/face_signup")
def face_signup():
    return render_template("face_signup.html")

def generate_frames():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Add CAP_DSHOW for Windows
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
        
    try:
        while True:
            success, frame = cap.read()
            if not success:
                break
            else:
                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    finally:
        cap.release()

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), 
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/start_face_registration", methods=["POST"])
def start_face_registration():
    try:
        users = load_users()
        license_number = list(users.keys())[-1]  # Get last registered user
        
        folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'recorded_video')
        os.makedirs(folder, exist_ok=True)
        video_path = os.path.join(folder, f"{license_number}.mp4")

        cap = None
        for index in [0, 1]:  # Try both 0 and 1 as camera indices
            cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)  # Add CAP_DSHOW for Windows
            if cap.isOpened():
                break
        
        if not cap or not cap.isOpened():
            return jsonify({
                "status": "error",
                "message": "Could not access webcam. Please check if it's connected and not in use by another application."
            }), 500

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Change from mp4v to XVID
        out = cv2.VideoWriter(video_path, fourcc, 20.0, (frame_width, frame_height))
        
        frame_count = 0
        fps = 20
        duration = 15
        total_frames = fps * duration
        
        while cap.isOpened() and frame_count < total_frames:
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
            frame_count += 1
        
        cap.release()
        out.release()
        cv2.destroyAllWindows()

        if frame_count > 0:
            users[license_number]["face_data"] = video_path
            save_users(users)
            return jsonify({
                "status": "success",
                "message": "Face registration completed successfully"
            })
        else:
            return jsonify({
                "status": "error",
                "message": "No frames were recorded. Please try again."
            }), 500
            
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"An error occurred: {str(e)}"
        }), 500


UPLOAD_FOLDER = 'uploads'
ENCODINGS_FILE = 'face_encodings.pkl'
USERS_FILE = 'users.json'
LEARNING_CACHE = {}
CACHE_DURATION = timedelta(hours=1)

os.makedirs(UPLOAD_FOLDER, exist_ok=True)



# Initialize face recognition models
try:
    import torch
    from facenet_pytorch import InceptionResnetV1, MTCNN
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize the models
    mtcnn = MTCNN(
        keep_all=True,
        device=device
    )
    resnet = InceptionResnetV1(
        pretrained='vggface2',
        device=device
    ).eval()
    
    FACE_RECOGNITION_ENABLED = True
    print("Face recognition models loaded successfully")
    
except Exception as e:
    print(f"Error initializing face recognition models: {e}")
    FACE_RECOGNITION_ENABLED = False
    mtcnn = None
    resnet = None

def load_users():
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, 'r') as f:
            return json.load(f)
    return {}

def get_video_hash(video_path):
    """Generate a hash of the video file based on modification time and size"""
    if not os.path.exists(video_path):
        return None
    stats = os.stat(video_path)
    hash_string = f"{video_path}_{stats.st_mtime}_{stats.st_size}"
    return hashlib.md5(hash_string.encode()).hexdigest()

def should_learn_face(username, video_path):
    """Check if we need to learn face encodings for this video"""
    current_time = datetime.now()
    video_hash = get_video_hash(video_path)
    
    if video_hash is None:
        return False
        
    if username in LEARNING_CACHE:
        cache_time, cached_hash = LEARNING_CACHE[username]
        # Return False if cache is still valid and video hasn't changed
        if (current_time - cache_time) < CACHE_DURATION and cached_hash == video_hash:
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

def load_encodings():
    if os.path.exists(ENCODINGS_FILE):
        with open(ENCODINGS_FILE, 'rb') as f:
            data = pickle.load(f)
            return data['encodings'], data['usernames']
    return [], []

def save_encodings(encodings, usernames):
    with open(ENCODINGS_FILE, 'wb') as f:
        pickle.dump({
            'encodings': encodings,
            'usernames': usernames
        }, f)

def learn_from_video(video_path, username):
    """Learn face encodings from a user's video"""
    if not FACE_RECOGNITION_ENABLED:
        raise ValueError("Face recognition is not available")
        
    # Check if learning is needed
    if not should_learn_face(username, video_path):
        print(f"Using cached encodings for user: {username}")
        return True
        
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
            
        encodings = []
        frame_count = 0
        
        while cap.isOpened() and frame_count < 30:  # Process up to 30 frames
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
        known_encodings, known_usernames = load_encodings()
        
        # Update or add new encoding
        if username in known_usernames:
            idx = known_usernames.index(username)
            known_encodings[idx] = mean_encoding
        else:
            known_encodings.append(mean_encoding)
            known_usernames.append(username)
            
        save_encodings(known_encodings, known_usernames)
        
        # Update the learning cache
        LEARNING_CACHE[username] = (datetime.now(), get_video_hash(video_path))
        
        print(f"Successfully learned face encoding for user: {username}")
        return True
        
    except Exception as e:
        print(f"Error learning from video for user {username}: {e}")
        return False

def update_face_encodings():
    """Update face encodings for all users with video data"""
    if not FACE_RECOGNITION_ENABLED:
        print("Face recognition is not available. Skipping encoding updates.")
        return
        
    users = load_users()
    updates_needed = False
    
    for username, user_data in users.items():
        if user_data.get('face_data'):
            video_path = user_data['face_data']
            if os.path.exists(video_path) and should_learn_face(username, video_path):
                updates_needed = True
                try:
                    learn_from_video(video_path, username)
                except Exception as e:
                    print(f"Error updating encodings for {username}: {e}")
    
    if not updates_needed:
        print("No face encoding updates needed")
        
def recognize_face(image, threshold=0.6):
    if not FACE_RECOGNITION_ENABLED:
        raise ValueError("Face recognition is not available")
        
    known_encodings, usernames = load_encodings()
    
    if not known_encodings:
        raise ValueError("No registered faces found in the system")

    face_encodings = detect_and_encode(image)
    
    if not face_encodings:
        return None

    # Use the first detected face
    test_encoding = face_encodings[0][0]
    distances = np.linalg.norm(np.array(known_encodings) - test_encoding, axis=1)
    min_distance_idx = np.argmin(distances)
    
    if distances[min_distance_idx] < threshold:
        return usernames[min_distance_idx]
    
    return None

@app.route("/face_login")
def face_login():
    if not FACE_RECOGNITION_ENABLED:
        return jsonify({
            "status": "error",
            "message": "Face recognition is not available"
        }), 500
        
    # Update face encodings before showing the login page
    update_face_encodings()
    return render_template("face_login.html")

@app.route("/verify_face", methods=["POST"])
def verify_face():
    if not FACE_RECOGNITION_ENABLED:
        return jsonify({
            "status": "error",
            "message": "Face recognition is not available"
        }), 500
        
    try:
        if 'image' not in request.files:
            return jsonify({
                "status": "error",
                "message": "No image provided"
            }), 400

        image_file = request.files['image']
        if image_file.filename == '':
            return jsonify({
                "status": "error",
                "message": "No image selected"
            }), 400

        # Save and process the image
        filename = secure_filename(image_file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        image_file.save(filepath)

        # Read and process the image
        image = cv2.imread(filepath)
        if image is None:
            raise ValueError("Could not read uploaded image")

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Clean up the temporary file
        os.remove(filepath)

        # Perform face recognition
        username = recognize_face(image_rgb)

        if username:
            # Verify user exists in users.json
            users = load_users()
            if username in users:
                session["user"] = username
                return jsonify({
                    "status": "success",
                    "redirect": url_for("dashboard")
                })

        return jsonify({
            "status": "error",
            "message": "Face not recognized. Please try again."
        }), 400

    except Exception as e:
        print(f"Error in verify_face: {e}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500
        
@app.route("/dashboard")
def dashboard():
    if "user" not in session:
        return redirect(url_for("face_login"))
    return render_template("dashboard1.html")

@app.route("/start_detection")
def start_detection():
    subprocess.Popen(["python", "detection/no_yawn.py"])
    return jsonify({"status": "Detection Started"})

@app.route("/stop_detection")
def stop_detection():
    os.system("pkill -f no_yawn.py")  
    return jsonify({"status": "Detection Stopped"})

@app.route("/history")
def history():
    if os.path.exists("history/history.json"):
        with open("history/history.json", "r") as f:
            data = json.load(f)
        return render_template("history.html", history=data)
    return "No history available."

if __name__ == "__main__":
    app.run(debug=True)