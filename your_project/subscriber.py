import paho.mqtt.client as mqtt
import numpy as np
import json
import base64
import time
import cv2
from threading import Thread
import queue

class WebcamSubscriber:
    def __init__(self, broker="192.168.169.1", topic="webcam/frames", buffer_size=2):
        self.frame_queue = queue.Queue(maxsize=buffer_size)
        self.latest_frame = None
        self.ret = False
        self.running = True
        
        # Initialize MQTT client
        self.client = mqtt.Client()
        self.client.on_message = self.on_message
        self.client.connect(broker, 1883, 60)
        self.topic = topic
        
    def on_message(self, client, userdata, message):
        try:
            msg_data = json.loads(message.payload.decode())
            frame_data = base64.b64decode(msg_data["data"])
            frame = np.frombuffer(frame_data, dtype=np.dtype(msg_data["dtype"]))
            frame = frame.reshape(msg_data["shape"])
            
            if not self.frame_queue.full():
                self.frame_queue.put_nowait(frame)
            
        except Exception as e:
            print(f"Error processing frame: {e}")
    
    def read(self):
        try:
            if not self.frame_queue.empty():
                self.latest_frame = self.frame_queue.get_nowait()
                self.ret = True
            return self.ret, self.latest_frame
        except queue.Empty:
            return self.ret, self.latest_frame
        
    def start(self):
        self.client.subscribe(self.topic)
        self.client.loop_start()
        
    def stop(self):
        self.running = False
        self.client.loop_stop()
        self.client.disconnect()

if __name__ == "__main__":
    subscriber = WebcamSubscriber()
    subscriber.start()
    
    last_time = time.time()
    frame_count = 0
    
    try:
        while True:
            ret, frame = subscriber.read()
            if ret:
                cv2.imshow('MQTT Camera Feed', frame)
                frame_count += 1
                
                # Calculate and display FPS
                if frame_count % 30 == 0:
                    current_time = time.time()
                    fps = 30 / (current_time - last_time)
                    print(f"Display FPS: {fps:.1f}")
                    last_time = current_time
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
    except KeyboardInterrupt:
        pass
    finally:
        subscriber.stop()
        cv2.destroyAllWindows()