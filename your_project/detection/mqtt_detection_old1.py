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
        
        # Debug flags
        self.last_published_drowsy = None
        self.last_published_mobile = None
        self.message_count = 0
        
        # MQTT Client setup
        self.mqtt_client = MQTTClient()
        self.mqtt_client.on_connect = self._on_connect
        self.mqtt_client.on_publish = self._on_publish
        self.mqtt_client.on_disconnect = self._on_disconnect
        
        # Previous states for MQTT
        self._prev_drowsy_mqtt_state = False
        self._prev_mobile_mqtt_state = False
        
        # Connect to MQTT broker
        try:
            print(f"Attempting to connect to MQTT broker at {self.broker}:{self.port}")
            self.mqtt_client.connect(self.broker, self.port, 60)
            self.mqtt_client.loop_start()
            print("MQTT client connected successfully")
            
            # Test message
            self.mqtt_client.publish(self.drowsy_topic, "test_connection")
            print("Test message published")
        except Exception as e:
            print(f"MQTT connection error: {str(e)}")
    
    def _on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            print("Successfully connected to MQTT broker")
            # Subscribe to our own topics for testing
            client.subscribe([(self.drowsy_topic, 0), (self.mobile_topic, 0)])
        else:
            print(f"Failed to connect to MQTT broker with code: {rc}")
    
    def _on_disconnect(self, client, userdata, rc):
        print(f"Disconnected from MQTT broker with code: {rc}")
        if rc != 0:
            print("Unexpected disconnection. Attempting to reconnect...")
            try:
                self.mqtt_client.reconnect()
            except Exception as e:
                print(f"Reconnection failed: {str(e)}")
    
    def _on_publish(self, client, userdata, mid):
        self.message_count += 1
        print(f"Message {mid} published successfully. Total messages: {self.message_count}")
    
    def publish_with_confirmation(self, topic, message):
        """Publish message with additional logging"""
        try:
            result = self.mqtt_client.publish(topic, message, qos=1)
            result.wait_for_publish()
            if result.is_published():
                print(f"Successfully published message '{message}' to topic '{topic}'")
                return True
            else:
                print(f"Failed to publish message '{message}' to topic '{topic}'")
                return False
        except Exception as e:
            print(f"Error publishing message: {str(e)}")
            return False
    
    def update_scores(self, drowsy_detected, mobile_detected):
        """Override update_scores to include MQTT publishing"""
        # Call parent method to maintain original functionality
        super().update_scores(drowsy_detected, mobile_detected)
        
        current_time = time.strftime("%H:%M:%S")
        
        # Handle MQTT publishing for drowsiness
        with self._lock:
            current_drowsy_state = self.drowsy_score >= self.drowsy_threshold
            if current_drowsy_state != self._prev_drowsy_mqtt_state:
                message = "drowsy" if current_drowsy_state else "false"
                print(f"\n[{current_time}] Drowsy state changed to: {message}")
                print(f"Drowsy score: {self.drowsy_score}, Threshold: {self.drowsy_threshold}")
                
                if self.publish_with_confirmation(self.drowsy_topic, message):
                    self._prev_drowsy_mqtt_state = current_drowsy_state
                    self.last_published_drowsy = message
            
            # Handle MQTT publishing for mobile detection
            current_mobile_state = self.mobile_score >= self.mobile_threshold
            if current_mobile_state != self._prev_mobile_mqtt_state:
                message = "mobile" if current_mobile_state else "false"
                print(f"\n[{current_time}] Mobile state changed to: {message}")
                print(f"Mobile score: {self.mobile_score}, Threshold: {self.mobile_threshold}")
                
                if self.publish_with_confirmation(self.mobile_topic, message):
                    self._prev_mobile_mqtt_state = current_mobile_state
                    self.last_published_mobile = message
    
    def get_mqtt_status(self):
        """Get current MQTT status for debugging"""
        return {
            'connected': self.mqtt_client.is_connected(),
            'message_count': self.message_count,
            'last_drowsy': self.last_published_drowsy,
            'last_mobile': self.last_published_mobile,
            'drowsy_state': self._prev_drowsy_mqtt_state,
            'mobile_state': self._prev_mobile_mqtt_state
        }
    
    def cleanup(self):
        """Override cleanup to include MQTT cleanup"""
        try:
            # Publish final offline messages
            self.publish_with_confirmation(self.drowsy_topic, "false")
            self.publish_with_confirmation(self.mobile_topic, "false")
            
            self.mqtt_client.loop_stop()
            self.mqtt_client.disconnect()
            print("MQTT client disconnected")
        except Exception as e:
            print(f"MQTT cleanup error: {str(e)}")
        finally:
            super().cleanup()