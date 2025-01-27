from paho.mqtt.client import Client as MQTTClient
from detection.no_yawn import DetectionProcessor
import threading
import time

class MQTTDetectionProcessor(DetectionProcessor):
    def __init__(self):
        super().__init__()
        
        # MQTT Configuration
        self.broker = "192.168.169.1"
        self.port = 1883
        self.drowsy_topic = "drowsiness/detected"
        self.mobile_topic = "mobile/detected"
        
        # Connection state
        self._connected = False
        self._connection_lock = threading.Lock()
        
        # MQTT Client setup with clean_session=True for fresh start
        self.mqtt_client = MQTTClient(client_id=f"detection_processor_{int(time.time())}", clean_session=True)
        self.mqtt_client.on_connect = self._on_connect
        self.mqtt_client.on_disconnect = self._on_disconnect
        self.mqtt_client.on_publish = self._on_publish
        
        # Set up automatic reconnection
        self.mqtt_client.reconnect_delay_set(min_delay=1, max_delay=30)
        
        # Previous states for MQTT
        self._prev_drowsy_mqtt_state = False
        self._prev_mobile_mqtt_state = False
        
        # Start connection monitor thread
        self._monitor_thread = threading.Thread(target=self._connection_monitor, daemon=True)
        self._monitor_thread.start()
        
        # Initial connection
        self._connect_mqtt()
    
    def _connect_mqtt(self):
        """Establish MQTT connection with retry logic"""
        try:
            print(f"Attempting to connect to MQTT broker at {self.broker}:{self.port}")
            self.mqtt_client.connect(self.broker, self.port, keepalive=60)
            self.mqtt_client.loop_start()
        except Exception as e:
            print(f"Initial MQTT connection error: {str(e)}")
            self._connected = False
    
    def _connection_monitor(self):
        """Monitor and maintain MQTT connection"""
        while self.running:  # Use the existing DetectionProcessor running flag
            if not self._connected:
                try:
                    print("Connection monitor: Attempting to reconnect...")
                    self.mqtt_client.reconnect()
                except Exception as e:
                    print(f"Reconnection attempt failed: {str(e)}")
            time.sleep(5)  # Check every 5 seconds
    
    def _on_connect(self, client, userdata, flags, rc):
        """Callback when connected to MQTT broker"""
        if rc == 0:
            with self._connection_lock:
                self._connected = True
            print("Successfully connected to MQTT broker")
            # Send test message to confirm connection
            self.mqtt_client.publish(self.drowsy_topic, "system_online", qos=1)
        else:
            print(f"Failed to connect to MQTT broker with code: {rc}")
            self._connected = False
    
    def _on_disconnect(self, client, userdata, rc):
        """Callback when disconnected from MQTT broker"""
        with self._connection_lock:
            self._connected = False
        print(f"Disconnected from MQTT broker with code: {rc}")
    
    def _on_publish(self, client, userdata, mid):
        """Callback when message is published"""
        print(f"Message {mid} published successfully")
    
    def publish_message(self, topic, message):
        """Publish message with retry logic"""
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries and self.running:
            if self._connected:
                try:
                    result = self.mqtt_client.publish(topic, message, qos=1)
                    if result.rc == 0:
                        print(f"Successfully published '{message}' to {topic}")
                        return True
                except Exception as e:
                    print(f"Error publishing message: {str(e)}")
            else:
                print("Not connected to MQTT broker, waiting...")
                time.sleep(1)
            retry_count += 1
        
        return False
    
    def update_scores(self, drowsy_detected, mobile_detected):
        """Override update_scores to include MQTT publishing"""
        # Call parent method
        super().update_scores(drowsy_detected, mobile_detected)
        
        with self._lock:
            # Handle drowsiness detection
            current_drowsy_state = self.drowsy_score >= self.drowsy_threshold
            if current_drowsy_state != self._prev_drowsy_mqtt_state:
                message = "drowsy" if current_drowsy_state else "false"
                if self.publish_message(self.drowsy_topic, message):
                    self._prev_drowsy_mqtt_state = current_drowsy_state
            
            # Handle mobile detection
            current_mobile_state = self.mobile_score >= self.mobile_threshold
            if current_mobile_state != self._prev_mobile_mqtt_state:
                message = "mobile" if current_mobile_state else "false"
                if self.publish_message(self.mobile_topic, message):
                    self._prev_mobile_mqtt_state = current_mobile_state
    
    def cleanup(self):
        """Override cleanup to include MQTT cleanup"""
        try:
            if self._connected:
                # Send offline messages
                self.publish_message(self.drowsy_topic, "system_offline")
                self.publish_message(self.mobile_topic, "system_offline")
                # Stop MQTT client
                self.mqtt_client.loop_stop()
                self.mqtt_client.disconnect()
                print("MQTT client disconnected cleanly")
        except Exception as e:
            print(f"MQTT cleanup error: {str(e)}")
        finally:
            super().cleanup()