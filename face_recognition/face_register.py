import cv2
import os
import json
import sys

def load_users():
    if os.path.exists("users.json"):
        with open("users.json", "r") as f:
            return json.load(f)
    return {}

def save_users(users):
    with open("users.json", "w") as f:
        json.dump(users, f, indent=4)

def record_video(license_number):
    folder = r'F:\Drowsiness_Iot\recorded_video'
    os.makedirs(folder, exist_ok=True)
    video_path = os.path.join(folder, f"{license_number}.mp4")
    
    cap = cv2.VideoCapture(0)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, 20.0, (frame_width, frame_height))
    
    print("Recording for 15 seconds...")
    frame_count = 0
    fps = 20
    duration = 15
    total_frames = fps * duration
    
    while cap.isOpened() and frame_count < total_frames:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
        cv2.imshow('Recording', frame)
        frame_count += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    users = load_users()
    if license_number in users:
        users[license_number]["face_data"] = video_path
        save_users(users)
        print(f"Face video saved as {video_path}")
    else:
        print("Error: License number not found in users.json")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python face_register.py <license_number>")
        sys.exit(1)
    
    license_number = sys.argv[1]
    record_video(license_number)
