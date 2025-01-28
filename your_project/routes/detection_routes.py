# routes/detection_routes.py
import os
import time
import json
import threading
import numpy as np
import cv2
from datetime import datetime
import csv
from flask import Blueprint, render_template, jsonify, Response, session, request, redirect, url_for

# from detection.no_yawn import DetectionProcessor

from detection.data_logger import DataLogger

from detection.mqtt_detection import MQTTDetectionProcessor


detection_bp = Blueprint("detection", __name__)

# Global variables to manage detection
detection_processor = None
detection_thread = None
is_detection_running = False
detector = None
data_logger = None

@detection_bp.route("/start_detection", methods=['POST'])
def start_detection():
    global detection_processor, detection_thread, is_detection_running, data_logger
    try:
        if not is_detection_running:

            # detection_processor = DetectionProcessor()
            detection_processor = MQTTDetectionProcessor()
            # Initialize data logger with user's license number
            data_logger = DataLogger(session['user'], detection_processor)
            print(f"Logs will be stored at: {data_logger.get_log_path()}")
            detection_thread = threading.Thread(target=detection_processor.run)
            detection_thread.daemon = True
            detection_thread.start()
            is_detection_running = True
            return jsonify({
                "status": "success",
                "message": "Detection Started Successfully"
            })
        return jsonify({
            "status": "warning",
            "message": "Detection Already Running"
        })
    except Exception as e:
        print(f"Start detection error: {str(e)}")
        return jsonify({
            "status": "error",
            "message": f"Failed to start detection: {str(e)}"
        }), 500

@detection_bp.route("/stop_detection", methods=['POST'])
def stop_detection():
    global detection_processor, detection_thread, is_detection_running, data_logger
    try:
        if detection_processor and is_detection_running:
            if data_logger:
                data_logger.stop()
                data_logger = None
            detection_processor.stop()
            if detection_thread and detection_thread.is_alive():
                detection_thread.join(timeout=1.0)
            detection_processor = None
            detection_thread = None
            is_detection_running = False
            save_detection_history()
            return jsonify({
                "status": "success",
                "message": "Detection Stopped Successfully"
            })
        return jsonify({
            "status": "warning",
            "message": "Detection Not Running"
        })
    except Exception as e:
        print(f"Stop detection error: {str(e)}")
        return jsonify({
            "status": "error",
            "message": f"Failed to stop detection: {str(e)}"
        }), 500

@detection_bp.route("/video_feed")
def video_feed():
    from utils.video_utils import generate_frames
    from config import VIDEO_WIDTH, VIDEO_HEIGHT
    return Response(generate_frames(VIDEO_WIDTH, VIDEO_HEIGHT),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@detection_bp.route("/video_feeds")
def video_feeds():
    def generate():
        while True:
            frame = None
            try:
                if detection_processor is not None and detection_processor.frame is not None:
                    frame = detection_processor.frame.copy()
            except Exception as e:
                print(f"Error copying detection_processor.frame: {e}")
            if frame is None:
                frame = np.zeros((480, 640, 3), np.uint8)
                cv2.putText(frame, "Initializing...", (200, 240), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            ret, jpeg = cv2.imencode('.jpg', frame)
            if not ret:
                print("Failed to encode frame")
                continue
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')
            time.sleep(0.033)
    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@detection_bp.route("/stream")
def stream():
    return alarm_stream()
def alarm_stream():
    def event_generator():
        while True:
            try:
                if detection_processor:
                    try:
                        metrics = detection_processor.get_metrics()
                        
                        # Separate handling for each alarm type
                        if detection_processor.drowsy_alarm:
                            try:
                                yield "event: drowsy-alarm\ndata: {}\n\n"
                            except Exception as drowsy_alarm_error:
                                print(f"Error generating drowsy alarm event: {drowsy_alarm_error}")
                        
                        if detection_processor.mobile_alarm:
                            try:
                                yield "event: mobile-alarm\ndata: {}\n\n"
                            except Exception as mobile_alarm_error:
                                print(f"Error generating mobile alarm event: {mobile_alarm_error}")
                        
                        # Metrics generation with error handling
                        try:
                            yield f"data: metrics:{json.dumps(metrics)}\n\n"
                        except Exception as metrics_error:
                            print(f"Error generating metrics event: {metrics_error}")
                    
                    except Exception as processor_error:
                        print(f"Error processing detection metrics: {processor_error}")
            
            except Exception as general_error:
                print(f"Unhandled error in alarm stream: {general_error}")
            
            time.sleep(0.1)
    
    return Response(event_generator(), mimetype="text/event-stream")

def save_detection_history():
    """Save detection session history to JSON file."""
    if not detection_processor:
        return
    try:
        metrics = detection_processor.get_metrics()
        history_data = {
            "timestamp": time.strftime("%H:%M:%S"),
            "duration": metrics.get("driving_time", 0),
            "drowsy_alarms": metrics.get("drowsy_alarms", 0),
            "mobile_alarms": metrics.get("mobile_alarms", 0),
            "blinks": metrics.get("total_blinks", 0),
            "max_drowsy_score": metrics.get("drowsy_score", 0),
            "max_mobile_score": metrics.get("mobile_score", 0)
        }
        os.makedirs("history", exist_ok=True)
        history_file = os.path.join("history", f"{session['user']}_history.json")
        existing_data = []
        if os.path.exists(history_file):
            with open(history_file, "r") as f:
                existing_data = json.load(f)
        existing_data.append(history_data)
        with open(history_file, "w") as f:
            json.dump(existing_data, f, indent=4)
    except Exception as e:
        print(f"Error saving history: {str(e)}")
        
@detection_bp.route("/csv_stats")
def csv_stats():
    """
    Read the latest entry from the user's CSV file and return the key metrics.
    """
    try:
        # Build the file path using the session's license number
        csv_path = os.path.join(os.getcwd(), 'data_logs', f"{session['user']}.csv")
        if not os.path.exists(csv_path):
            return jsonify({"error": "CSV log file not found."}), 404

        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            if not rows:
                return jsonify({"error": "No data found in CSV."}), 404
            last_row = rows[-1]

        # Helper to convert HH:MM:SS to seconds.
        def hms_to_seconds(time_str):
            h, m, s = map(int, time_str.split(':'))
            return h * 3600 + m * 60 + s

        total_driving_time = last_row.get('total_driving_time', "00:00:00")
        total_seconds = hms_to_seconds(total_driving_time)
        drowsy_time = float(last_row.get('drowsy_time', 0))
        drowsy_alarm = int(last_row.get('drowsy_alarm', 0))
        blink_rate = float(last_row.get('blink_rate', 0))

        return jsonify({
            'drowsy_alarm': drowsy_alarm,
            'blink_rate': blink_rate,
            'total_driving_time': total_driving_time,
            'drowsy_time': drowsy_time,
            'total_driving_seconds': total_seconds
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@detection_bp.route("/mqtt_status")
def mqtt_status():
    if detection_processor:
        return jsonify(detection_processor.get_mqtt_status())
    return jsonify({"status": "Detection not running"})
    
    
def get_last_ride_data(csv_path):
    if not os.path.exists(csv_path):
        return {}
    
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        rows = [row for row in reader]
    
    rides = []
    current_ride = []
    prev_time = None
    
    for row in rows:
        try:
            current_time = datetime.strptime(row['timestamp'], "%Y-%m-%d %H:%M:%S")
        except ValueError:
            continue
        
        if prev_time and (current_time - prev_time).total_seconds() > 120:
            if current_ride:
                rides.append(current_ride)
                current_ride = []
        current_ride.append(row)
        prev_time = current_time
    
    if current_ride:
        rides.append(current_ride)
    
    return {
        'total_driving_time': rides[-1][-1]['total_driving_time'] if rides else '00:00:00',
        'drowsy_alarms': sum(int(r['drowsy_alarm']) for ride in rides[-1:] for r in ride) if rides else 0,
        'blink_rate': rides[-1][-1]['blink_rate'] if rides else 0,
        'mobile_alarms': sum(int(r['mobile_alarm']) for ride in rides[-1:] for r in ride) if rides else 0
    } if rides else {}

def process_rides(csv_path):
    rides = []
    if not os.path.exists(csv_path):
        return rides
    
    try:
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        current_ride = []
        prev_time = None
        
        for row in rows:
            try:
                current_time = datetime.strptime(row['timestamp'], "%Y-%m-%d %H:%M:%S")
            except (KeyError, ValueError):
                continue
                
            if prev_time is None:
                current_ride.append(row)
            else:
                time_diff = (current_time - prev_time).total_seconds()
                if time_diff > 120:
                    if current_ride:
                        rides.append(current_ride)
                        current_ride = []
                current_ride.append(row)
            
            prev_time = current_time

        if current_ride:
            rides.append(current_ride)

    except Exception as e:
        print(f"Error processing rides: {str(e)}")
    
    return rides


def get_users():
    """Helper function to load users from JSON file"""
    try:
        with open('users.json', 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading users: {str(e)}")
        return {}
