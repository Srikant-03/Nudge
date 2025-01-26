# app.py
from flask import Flask
from config import SECRET_KEY, UPLOAD_FOLDER
import os

def create_app():
    app = Flask(__name__)
    app.secret_key = SECRET_KEY
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

    # Import and register blueprints
    from routes.main_routes import main_bp
    from routes.auth_routes import auth_bp
    from routes.dashboard_routes import dashboard_bp
    from routes.detection_routes import detection_bp
    

    app.register_blueprint(main_bp)
    app.register_blueprint(auth_bp)
    app.register_blueprint(dashboard_bp)
    app.register_blueprint(detection_bp)
    

    return app

if __name__ == "__main__":
    app = create_app()
    app.run(debug=True)
