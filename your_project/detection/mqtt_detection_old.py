from paho.mqtt.client import Client as MQTTClient
import threading
from detection.no_yawn import DetectionProcessor

class MQTTDetectionProcessor(DetectionProcessor):
    def __init__(self):
        super().__init__()
        
        # MQTT Configuration
        self.broker = "192.168.169.1"
        self.port = 1883
        self.drowsy_topic = "drowsiness/detected"
        self.mobile_topic = "mobile/detected"
        
        # MQTT Client setup
        self.mqtt_client = MQTTClient()
        self.mqtt_client.on_connect = self._on_connect
        self.mqtt_client.on_publish = self._on_publish
        
        # Previous states for MQTT
        self._prev_drowsy_mqtt_state = False
        self._prev_mobile_mqtt_state = False
        
        # Connect to MQTT broker
        try:
            self.mqtt_client.connect(self.broker, self.port, 60)
            self.mqtt_client.loop_start()
            print("MQTT client connected successfully")
        except Exception as e:
            print(f"MQTT connection error: {str(e)}")
    
    def _on_connect(self, client, userdata, flags, rc):
        print(f"MQTT Connected with result code {rc}")
    
    def _on_publish(self, client, userdata, mid):
        print(f"MQTT Message published: {mid}")
    
    def update_scores(self, drowsy_detected, mobile_detected):
        """Override update_scores to include MQTT publishing"""
        # Call parent method to maintain original functionality
        super().update_scores(drowsy_detected, mobile_detected)
        
        # Handle MQTT publishing for drowsiness
        with self._lock:
            current_drowsy_state = self.drowsy_score >= self.drowsy_threshold
            if current_drowsy_state != self._prev_drowsy_mqtt_state:
                self.mqtt_client.publish(
                    self.drowsy_topic, 
                    "drowsy" if current_drowsy_state else "false"
                )
                self._prev_drowsy_mqtt_state = current_drowsy_state
            
            # Handle MQTT publishing for mobile detection
            current_mobile_state = self.mobile_score >= self.mobile_threshold
            if current_mobile_state != self._prev_mobile_mqtt_state:
                self.mqtt_client.publish(
                    self.mobile_topic, 
                    "mobile" if current_mobile_state else "false"
                )
                self._prev_mobile_mqtt_state = current_mobile_state
    
    def cleanup(self):
        """Override cleanup to include MQTT cleanup"""
        try:
            self.mqtt_client.loop_stop()
            self.mqtt_client.disconnect()
            print("MQTT client disconnected")
        except Exception as e:
            print(f"MQTT cleanup error: {str(e)}")
        finally:
            super().cleanup()