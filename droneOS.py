import paho.mqtt.client as mqtt
import json
import time
import threading
import uuid
from datetime import datetime, timezone
from Telemetry_Generator.telemetryGen import FlightPathSimulator
import config
from detect import run_detector

BROKER = "localhost"
PORT = 1883

TELEMETRY_TOPIC = "drone/telemetry"
STATUS_TOPIC = "drone/status"
COMMAND_TOPIC = "drone/command"
DETECTION_TOPIC = "drone/detection"
MISSION_TOPIC = "drone/mission"


telemetry_running = threading.Event()
telemetry_thread = None
telemetry_lock = threading.Lock()
simulator = FlightPathSimulator(start_lat=57.048, start_lon=9.918)
cv_running = threading.Event()
cv_stop_event = threading.Event()
cv_thread = None
current_mission_id = None
current_mission_started_at = None


def _now_iso():
    return datetime.now(timezone.utc).isoformat()


def _to_json_safe(value):
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, dict):
        return {k: _to_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json_safe(v) for v in value]
    return value


def _publish_mission_started(source="START_CV"):
    global current_mission_id, current_mission_started_at

    current_mission_id = str(uuid.uuid4())
    current_mission_started_at = _now_iso()
    event = {
        "event": "mission_started",
        "mission_id": current_mission_id,
        "started_at": current_mission_started_at,
        "source": source,
        "model": getattr(config, "MODEL_PATH", None),
        "video_source": getattr(config, "VIDEO_PATH", None),
    }

    client.publish(MISSION_TOPIC, json.dumps(_to_json_safe(event)), qos=1)


def _publish_mission_stopped(reason):
    global current_mission_id, current_mission_started_at

    if current_mission_id is None:
        return

    event = {
        "event": "mission_stopped",
        "mission_id": current_mission_id,
        "started_at": current_mission_started_at,
        "stopped_at": _now_iso(),
        "reason": reason,
    }
    current_mission_id = None
    current_mission_started_at = None

    client.publish(MISSION_TOPIC, json.dumps(_to_json_safe(event)), qos=1)


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
    global cv_thread
    if cv_running.is_set():
        print("Computer vision already running.")
        return

    print("Starting computer vision...")
    _publish_mission_started(source="START_CV")
    time.sleep(1)
    cv_stop_event.clear()
    cv_running.set()
    cv_thread = threading.Thread(target=cv_worker, daemon=True)
    cv_thread.start()


def perform_stop_cv():
    global cv_thread
    if not cv_running.is_set():
        print("Computer vision already stopped.")
        return

    print("Stopping computer vision...")
    cv_stop_event.set()
    if cv_thread is not None and cv_thread.is_alive():
        cv_thread.join(timeout=5)


def publish_cv_detection(detection_payload):
    with telemetry_lock:
        telemetry = simulator.get_telemetry()

    event = {
        "event": "cv_detection",
        "mission_id": current_mission_id,
        "person_id": detection_payload.get("person_id"),
        "state": detection_payload.get("state", "unknown"),
        "frame_count": detection_payload.get("frame_count"),
        "bbox": detection_payload.get("last_box"),
        "lat": telemetry["lat"],
        "lon": telemetry["lon"],
        "timestamp": telemetry["timestamp"],
    }
    client.publish(DETECTION_TOPIC, json.dumps(_to_json_safe(event)), qos=1)


def cv_worker():
    stop_reason = "stream_end"
    try:
        run_detector(stop_event=cv_stop_event, on_person_state_change=publish_cv_detection)
    except Exception as e:
        stop_reason = "error"
        print(f"CV worker error: {e}")
    finally:
        if stop_reason != "error" and cv_stop_event.is_set():
            stop_reason = "command_stop"
        _publish_mission_stopped(reason=stop_reason)
        cv_running.clear()


COMMAND_MAP = {
    "RETURN_HOME": perform_return_home,
    "LAND": perform_land,
    "HOVER": perform_hover,
    "CIRCLE": perform_circle,
    "START_CV": perform_start_cv,
    "STOP_CV": perform_stop_cv,
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
    cv_stop_event.set()
    client.publish(STATUS_TOPIC, "offline", qos=1, retain=True)
    client.loop_stop()
    client.disconnect()
