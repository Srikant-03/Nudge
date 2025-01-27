# config.py
import os
from datetime import timedelta

# Flask configuration
SECRET_KEY = os.urandom(24)

# File paths and folders
USERS_FILE = os.path.join(os.getcwd(), "users.json")
ENCODINGS_FILE = os.path.join(os.getcwd(), "face_encodings.pkl")
UPLOAD_FOLDER = os.path.join(os.getcwd(), "uploads")

# Cache duration for face learning
CACHE_DURATION = timedelta(hours=1)

# Video capture settings
VIDEO_WIDTH = 480
VIDEO_HEIGHT = 480
