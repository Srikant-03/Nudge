import cv2
import numpy as np
import time
from flask import Response, Flask

app = Flask(__name__)

@app.route('/dummy_video_feeds')
def dummy_video_feeds():
    def generate():
        while True:
            # Create a blank image with "Initializing..." text
            frame = np.zeros((480, 640, 3), np.uint8)
            cv2.putText(frame, "Initializing...", (100, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            ret, jpeg = cv2.imencode('.jpg', frame)
            if not ret:
                print("Failed to encode dummy frame")
                continue
            # Debug: print a message in the server log
            print("Yielding a dummy frame")
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
            time.sleep(0.1)  # 10 FPS for testing

    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)