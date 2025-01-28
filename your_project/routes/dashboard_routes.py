# routes/dashboard_routes.py
import os
import json
import time
from flask import Blueprint, render_template, session, redirect, url_for, jsonify, request
from utils import file_utils

dashboard_bp = Blueprint("dashboard", __name__)

@dashboard_bp.route("/dashboard")
def dashboard():
    if "user" not in session:
        return redirect(url_for("auth.face_login"))
    return render_template("dashboard1.html", user=session.get('user'))

@dashboard_bp.route("/history")
def history():
    try:
        if "user" not in session:
            return redirect(url_for("auth.face_login"))
        history_file = os.path.join("history", f"{session['user']}_history.json")
        if os.path.exists(history_file):
            with open(history_file, "r") as f:
                data = json.load(f)
            return render_template("history.html", history=data)
        return render_template("history.html", history=[])
    except Exception as e:
        print(f"History error: {str(e)}")
        return render_template("error.html", error="Failed to load history", details=str(e))

@dashboard_bp.route("/get_detection_status")
def get_detection_status():
    # This route returns the detection status. The detection processor is managed
    # in the detection routes module (global variable shared among detection routes)
    from routes.detection_routes import detection_processor, is_detection_running
    status = {
        "is_running": is_detection_running,
        "camera_connected": detection_processor.is_camera_connected() if detection_processor else False,
        "metrics": detection_processor.get_metrics() if detection_processor else {}
    }
    return jsonify(status)
