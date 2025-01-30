# utils/video_utils.py
import cv2
import os
import time

def generate_frames(video_width, video_height):
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, video_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, video_height)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return

    try:
        while True:
            success, frame = cap.read()
            if not success:
                break
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    finally:
        cap.release()

def record_video(license_number, duration, fps, folder="recorded_video"):
    os.makedirs(folder, exist_ok=True)
    video_path = os.path.join(folder, f"{license_number}.mp4")

    cap = None
    for index in [0, 1]:
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if cap.isOpened():
            break

    if not cap or not cap.isOpened():
        raise Exception("Could not access webcam. Check connection or if another application is using it.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(video_path, fourcc, fps, (frame_width, frame_height))

    frame_count = 0
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
        return video_path
    else:
        raise Exception("No frames recorded. Please try again.")
