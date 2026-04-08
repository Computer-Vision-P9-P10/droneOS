import paho.mqtt.client as mqtt
import json
import time
import threading
from Telemetry_Generator.telemetryGen import FlightPathSimulator

BROKER = "localhost"
PORT = 1883

TELEMETRY_TOPIC = "drone/telemetry"
STATUS_TOPIC = "drone/status"
COMMAND_TOPIC = "drone/command"
DETECTION_TOPIC = "drone/detection"


telemetry_running = threading.Event()
telemetry_thread = None
telemetry_lock = threading.Lock()
simulator = FlightPathSimulator(start_lat=57.048, start_lon=9.918)


def perform_return_home():
    print("Returning home...")


def perform_land():
    print("Landing now...")


def perform_hover():
    print("Hovering...")


def perform_circle():
    print("Circling around...")


def telemetry_worker():
    while telemetry_running.is_set():
        with telemetry_lock:
            simulator.update()
            telemetry = simulator.get_telemetry()
        client.publish(TELEMETRY_TOPIC, json.dumps(telemetry), qos=0)
        time.sleep(1)


def perform_start_telemetry():
    global telemetry_thread
    if telemetry_running.is_set():
        print("Telemetry already running.")
        return

    print("Starting telemetry...")
    telemetry_running.set()
    telemetry_thread = threading.Thread(target=telemetry_worker, daemon=True)
    telemetry_thread.start()


def perform_stop_telemetry():
    if not telemetry_running.is_set():
        print("Telemetry already stopped.")
        return

    print("Stopping telemetry...")
    telemetry_running.clear()


def perform_start_cv():
    print("Starting computer vision...")
    threading.Thread(target=cv_worker, daemon=True).start()


def cv_worker():
    while True:
        with telemetry_lock:
            telemetry = simulator.get_telemetry()
        event = {
            "type": "cv_detection",
            "detected": "person",
            "lat": telemetry["lat"],
            "lon": telemetry["lon"],
            "timestamp": telemetry["timestamp"],
        }
        client.publish(DETECTION_TOPIC, json.dumps(event), qos=1)
        time.sleep(5)


COMMAND_MAP = {
    "RETURN_HOME": perform_return_home,
    "LAND": perform_land,
    "HOVER": perform_hover,
    "CIRCLE": perform_circle,
    "START_CV": perform_start_cv,
    "START_TELEMETRY": perform_start_telemetry,
    "STOP_TELEMETRY": perform_stop_telemetry,
}


def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Connected to MQTT broker.")
        client.subscribe(COMMAND_TOPIC, qos=1)
        client.publish(STATUS_TOPIC, "online", qos=1, retain=True)
        print(f"Subscribed to {COMMAND_TOPIC}")
    else:
        print(f"Failed to connect, rc={rc}")


def on_disconnect(client, userdata, rc):
    if rc != 0:
        print("Unexpected disconnection. Triggering RTH.")
        perform_return_home()


def on_message(client, userdata, message):
    if message.topic != COMMAND_TOPIC:
        return

    try:
        payload = message.payload.decode("utf-8")
        data = json.loads(payload)
        cmd = data.get("cmd")
        action = COMMAND_MAP.get(cmd)

        if action:
            print(f"Received command: {cmd}")
            action()
        else:
            print(f"Unknown command: {cmd}")
    except json.JSONDecodeError:
        print(f"Invalid JSON on {COMMAND_TOPIC}: {message.payload!r}")
    except Exception as e:
        print(f"Command handling error: {e}")


client = mqtt.Client(client_id="droneOS")
client.will_set(STATUS_TOPIC, "offline", qos=1, retain=True)

client.on_connect = on_connect
client.on_disconnect = on_disconnect
client.on_message = on_message

client.connect(BROKER, PORT, keepalive=10)
client.loop_start()

try:
    while True:
        time.sleep(0.2)
except KeyboardInterrupt:
    print("Shutting down...")
finally:
    telemetry_running.clear()
    client.publish(STATUS_TOPIC, "offline", qos=1, retain=True)
    client.loop_stop()
    client.disconnect()
