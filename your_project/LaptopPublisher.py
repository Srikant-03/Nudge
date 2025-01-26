import threading
import paho.mqtt.client as mqtt
import time

# MQTT Broker details
broker = "192.168.169.1"  # Replace with your broker IP
port = 1883
topic = "drowsiness/detected"

# Define the MQTT client
client = mqtt.Client()

# Set up the username and password for broker if needed
# client.username_pw_set("your_username", "your_password")

# Define callback for connection
def on_connect(client, userdata, flags, rc):
    print(f"Connected with result code {rc}")
    client.subscribe(topic)

# Callback for publishing messages
def on_publish(client, userdata, mid):
    print(f"Message published: {mid}")

# Callback for receiving messages (not needed for publisher but can be useful for testing)
def on_message(client, userdata, msg):
    print(f"Received message: {msg.payload.decode()}")

# Setup MQTT callbacks
client.on_connect = on_connect
client.on_publish = on_publish
client.on_message = on_message

# Connect to the MQTT broker
client.connect(broker, port, 60)

# Start the loop to maintain connection
client.loop_start()

# Shared global variable to track drowsiness status
drowsiness_detected = False

# Flag to control graceful shutdown
stop_thread = False

# MQTT publishing logic (runs in a separate thread)
def mqtt_publish():
    global drowsiness_detected, stop_thread
    prev_detection = False

    while not stop_thread:
        # Check drowsiness status

        # If drowsiness detected and not previously detected, send "true"
        if drowsiness_detected and not prev_detection:
            client.publish(topic, "drowsy")
            prev_detection = True
            print("Drowsiness detected! Sending 'true' to subscriber.")

        # If no drowsiness detected and previously detected, send "false"
        elif not drowsiness_detected and prev_detection:
            client.publish(topic, "false")
            prev_detection = False
            print("No drowsiness detected. Sending 'false' to subscriber.")

        # Delay before checking again
        time.sleep(1)

# Function to simulate continuous drowsiness detection (for demonstration purposes)
def simulate_drowsiness_detection():
    global drowsiness_detected, stop_thread
    while not stop_thread:
        # Actual drowsiness detection logic goes here
        # For now, we're simulating continuous detection
        drowsiness_detected = False
        time.sleep(10)
        drowsiness_detected = True
        time.sleep(10)

# Create threads for both tasks
mqtt_thread = threading.Thread(target=mqtt_publish)
drowsiness_thread = threading.Thread(target=simulate_drowsiness_detection)

# Start the threads
mqtt_thread.start()
drowsiness_thread.start()

try:
    # Keep the main thread alive while both threads are running
    while True:
        time.sleep(1)

except KeyboardInterrupt:
    print("Exiting... Gracefully shutting down...")

finally:
    # Set the flag to stop the threads
    stop_thread = True

    # Join the threads to ensure they finish gracefully
    mqtt_thread.join()
    drowsiness_thread.join()

    # Stop the MQTT client loop and disconnect
    client.loop_stop()
    client.disconnect()
    print("MQTT client disconnected and threads joined. Exiting.")