import csv
import json
import os
import threading
import time
from datetime import datetime
from pathlib import Path
import stat

class DataLogger:
    def __init__(self, license_number, detector):
        """
        Initialize the data logger for a specific user session.
        
        Args:
            license_number (str): User's license number for file identification
            detector (DetectionProcessor): Reference to the detection processor instance
        """
        self.license_number = license_number
        self.detector = detector
        
        # Create project directory structure with full permissions
        self.base_dir = os.path.join(os.getcwd(),'data_logs')
        self.file_path = os.path.join(self.base_dir, f"{self.license_number}.csv")
        self.last_timestamp = None
        self.running = True
        self.session_start_time = time.time()
        self.file_lock = threading.Lock()
        self.last_total_blinks = 0  # Track blinks between intervals
        
        try:
            self._setup_log_file()
            self.thread = threading.Thread(target=self._logging_loop, daemon=False)
            self.thread.start()
        except Exception as e:
            print(f"Initialization error: {str(e)}")
            raise

    def _ensure_directory_permissions(self, path):
        """Ensure directory exists with full permissions."""
        try:
            # Create directory if it doesn't exist
            os.makedirs(path, exist_ok=True)
            
            # Set full permissions (read/write/execute for owner, group, others)
            os.chmod(path, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
            
            # Set permissions for parent directories
            parent = Path(path).parent
            while str(parent) != parent.root:
                try:
                    os.chmod(str(parent), stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
                    parent = parent.parent
                except:
                    break
                    
        except Exception as e:
            print(f"Permission setting error for {path}: {str(e)}")
            raise

    def _setup_log_file(self):
        """Set up the CSV file with headers if it doesn't exist and get last timestamp."""
        try:
            # Ensure directory exists with proper permissions
            self._ensure_directory_permissions(self.base_dir)
            
            file_exists = os.path.exists(self.file_path)
            mode = 'a' if file_exists else 'w'
            
            # Create or append to file
            with open(self.file_path, mode, newline='') as f:
                if not file_exists:
                    writer = csv.writer(f)
                    writer.writerow([
                        'timestamp', 'date', 'day', 'drowsy_alarm', 
                        'mobile_alarm', 'total_driving_time', 'drowsy_time',
                        'blink_rate', 'break_time'
                    ])
            
            # Set file permissions
            os.chmod(self.file_path, stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IWGRP | stat.S_IROTH | stat.S_IWOTH)
            
            # Get last timestamp if file exists
            if file_exists:
                with open(self.file_path, 'r') as f:
                    lines = list(csv.reader(f))
                    if len(lines) > 1:
                        last_line = lines[-1]
                        if last_line[0] != 'timestamp':
                            self.last_timestamp = datetime.strptime(
                                last_line[0], "%Y-%m-%d %H:%M:%S"
                            ).timestamp()
                            self.last_total_blinks = self.detector.total_blinks
                            
        except Exception as e:
            print(f"File setup error in {self.base_dir}: {str(e)}")
            raise

    def _calculate_blink_rate(self, current_total_blinks, elapsed_minutes):
        """Calculate the blink rate for the current interval."""
        if elapsed_minutes == 0:
            return 0
            
        blinks_in_interval = current_total_blinks - self.last_total_blinks
        rate = blinks_in_interval / elapsed_minutes
        self.last_total_blinks = current_total_blinks
        return rate

    def _log_data(self):
        """Log current detection metrics to CSV file."""
        try:
            current_time = time.time()
            current_dt = datetime.fromtimestamp(current_time)
        
            with self.detector._lock:
                drowsy_alarms = self.detector.drowsy_alarm_count
                mobile_alarms = self.detector.mobile_alarm_count
                total_blinks = self.detector.total_blinks
        
            break_time = 0
            if self.last_timestamp and (current_time - self.last_timestamp) >= 3600:
                break_time = (current_time - self.last_timestamp) / 60
        
            # Calculate time elapsed since last log
            elapsed_minutes = 1  # Default to 1 minute for regular intervals
            if self.last_timestamp:
                elapsed_minutes = max(1, (current_time - self.last_timestamp) / 60)
        
            # Calculate total driving time in HH:MM:SS format
            elapsed_time = current_time - self.session_start_time
            hours = int(elapsed_time // 3600)
            minutes = int((elapsed_time % 3600) // 60)
            seconds = int(elapsed_time % 60)
            total_driving_time = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        
            drowsy_time = 2 * drowsy_alarms
            blink_rate = self._calculate_blink_rate(total_blinks, elapsed_minutes)
        
            row = [
                current_dt.strftime("%Y-%m-%d %H:%M:%S"),  # Ensure full timestamp
                current_dt.strftime("%Y-%m-%d"),
                current_dt.strftime("%A"),
                drowsy_alarms,
                mobile_alarms,
                total_driving_time,  # Now formatted as HH:MM:SS
                drowsy_time,
                round(blink_rate, 2),
                round(break_time, 2)
            ]

            with self.file_lock:
                with open(self.file_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(row)
        
            self.last_timestamp = current_time
        
        except Exception as e:
            print(f"Data logging error: {str(e)}")

    def _logging_loop(self):
        """Main logging loop that runs in a separate thread."""
        while self.running:
            try:
                self._log_data()
                time.sleep(60 - time.time() % 60)  # Sync to minute boundaries
            except Exception as e:
                print(f"Logging error: {str(e)}")
                time.sleep(1)

    def stop(self):
        """Stop the logging thread gracefully."""
        self.running = False
        if hasattr(self, 'thread') and self.thread.is_alive():
            self.thread.join(timeout=2.0)

    def get_log_path(self):
        """Return the current log file path."""
        return self.file_path