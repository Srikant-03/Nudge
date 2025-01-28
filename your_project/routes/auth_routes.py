# routes/auth_routes.py
import os
import datetime
from flask import Blueprint, render_template, request, redirect, url_for, jsonify, session
from werkzeug.utils import secure_filename

# Import the module itself
import utils.file_utils as file_utils
# Or alternatively, explicit imports
from utils.file_utils import load_users, save_users, load_encodings, save_encodings
from utils.video_utils import record_video
from utils.face_recognition import learn_from_video, FACE_RECOGNITION_ENABLED, recognize_face

from config import USERS_FILE, UPLOAD_FOLDER
auth_bp = Blueprint("auth", __name__)

@auth_bp.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        license_number = request.form["license"]
        password = request.form["password"]
        confirm_password = request.form["confirm_password"]
        
        users = file_utils.load_users(USERS_FILE)
        if license_number in users:
            return "User already exists. Try logging in."

        if password != confirm_password:
            return "Passwords do not match."

        users[license_number] = {"password": password, "face_data": None}
        file_utils.save_users(users, USERS_FILE)
        
        session['user'] = license_number
        session['license_number'] = license_number
        
        return redirect(url_for("auth.face_signup"))
    return render_template("signup.html")



@auth_bp.route("/face_signup")
def face_signup():
    return render_template("face_signup.html")

@auth_bp.route("/start_face_registration", methods=["POST"])
def start_face_registration():
    try:
        # Ensure user is logged in or in signup process
        if 'user' not in session:
            return jsonify({
                "status": "error", 
                "message": "User not logged in"
            }), 403

        # Get the user's license number from session
        license_number = session.get('license_number')
        if not license_number:
            return jsonify({
                "status": "error", 
                "message": "License number not found"
            }), 400

        # Create a videos directory if it doesn't exist
        os.makedirs("recorded_video", exist_ok=True)
        

        # Record a 15-second video for face registration
        video_path = record_video(
            license_number, 
            duration=15, 
            fps=30, 
            folder="recorded_video"
        )

        encodings_file = os.path.join(os.getcwd(), "face_encodings.pkl")
        success = learn_from_video(
            video_path, 
            username=license_number, 
            encodings_file=encodings_file, 
            cache_duration=datetime.timedelta(days=30),
            load_encodings_fn=load_encodings,
            save_encodings_fn=save_encodings
        )
        
        if not success:
            return jsonify({
                "status": "error", 
                "message": "Failed to learn face encodings"
            }), 500
            
        users = file_utils.load_users(USERS_FILE)
        if license_number in users:
            users[license_number]["face_data"] = video_path  # Save video path
            file_utils.save_users(users, USERS_FILE)
    
        return jsonify({
            "status": "success", 
            "message": "Face registration video recorded successfully",
            "video_path": video_path
        })

    except Exception as e:
        print(f"Face registration error: {str(e)}")
        return jsonify({
            "status": "error", 
            "message": f"Face registration failed: {str(e)}"
        }), 500
        
        
@auth_bp.route("/face_login")
def face_login():
    # (You might want to update face encodings before showing the page)
    # For simplicity, we assume the update happens elsewhere
    if not FACE_RECOGNITION_ENABLED:
        return jsonify({
            "status": "error",
            "message": "Face recognition is not available"
        }), 500
    return render_template("face_login.html")

@auth_bp.route("/verify_face", methods=["POST"])
def verify_face():
    if not FACE_RECOGNITION_ENABLED:
        return jsonify({
            "status": "error",
            "message": "Face recognition is not available"
        }), 500
    try:
        if 'image' not in request.files:
            return jsonify({"status": "error", "message": "No image provided"}), 400

        image_file = request.files['image']
        if image_file.filename == '':
            return jsonify({"status": "error", "message": "No image selected"}), 400

        filename = secure_filename(image_file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        image_file.save(filepath)

        import cv2  # local import
        image = cv2.imread(filepath)
        if image is None:
            raise ValueError("Could not read uploaded image")

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        os.remove(filepath)

        username = recognize_face(image_rgb, threshold=0.6, encodings_file=os.path.join(os.getcwd(), "face_encodings.pkl"), load_encodings_fn=file_utils.load_encodings)
        if username:
            users = file_utils.load_users(USERS_FILE)
            if username in users:
                session["user"] = username
                return jsonify({
                    "status": "success",
                    "redirect": url_for("dashboard.dashboard")
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
