import paho.mqtt.client as mqtt
import json
import time
import threading
import uuid
import os
import cv2
import hmac
import hashlib
import secrets
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs
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
IMAGE_REQUEST_TOPIC = "drone/image_request"
IMAGE_RESPONSE_TOPIC = "drone/image_response"
CAPTURE_ROOT = "captures"
IMAGE_HTTP_BIND_HOST = getattr(config, "IMAGE_HTTP_BIND_HOST", "0.0.0.0")
IMAGE_HTTP_PORT = int(getattr(config, "IMAGE_HTTP_PORT", 8081))
IMAGE_HTTP_PUBLIC_HOST = getattr(config, "IMAGE_HTTP_PUBLIC_HOST", "localhost")
IMAGE_URL_TTL_SECONDS = int(getattr(config, "IMAGE_URL_TTL_SECONDS", 30))
IMAGE_URL_SIGNING_SECRET = getattr(config, "IMAGE_URL_SIGNING_SECRET", None)


telemetry_running = threading.Event()
telemetry_thread = None
telemetry_lock = threading.Lock()
# Cassiopeia as start location
simulator = FlightPathSimulator(start_lat=57.01239104461584, start_lon=9.991556328232413)
cv_running = threading.Event()
cv_stop_event = threading.Event()
cv_thread = None
current_mission_id = None
current_mission_started_at = None
next_violation_id = 1
image_http_server = None
image_http_thread = None
IMAGE_URL_SIGNING_SECRET = str(IMAGE_URL_SIGNING_SECRET or secrets.token_hex(32))


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
    global current_mission_id, current_mission_started_at, next_violation_id

    current_mission_id = str(uuid.uuid4())
    current_mission_started_at = _now_iso()
    next_violation_id = 1
    os.makedirs(os.path.join(CAPTURE_ROOT, current_mission_id), exist_ok=True)
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


def _save_violation_snapshot(mission_id, violation_id, frame):
    mission_dir = os.path.join(CAPTURE_ROOT, mission_id)
    os.makedirs(mission_dir, exist_ok=True)
    image_path = os.path.join(mission_dir, f"violation_{violation_id}.jpg")
    success = cv2.imwrite(image_path, frame)
    return image_path if success else None


def _publish_image_response(payload):
    client.publish(IMAGE_RESPONSE_TOPIC, json.dumps(_to_json_safe(payload)), qos=1)


def _build_image_path(mission_id, violation_id):
    return os.path.join(CAPTURE_ROOT, mission_id, f"violation_{violation_id}.jpg")


def _build_image_url(mission_id, violation_id):
    exp = int(time.time()) + IMAGE_URL_TTL_SECONDS
    sig = _sign_image_url(mission_id, violation_id, exp)
    return (
        f"http://{IMAGE_HTTP_PUBLIC_HOST}:{IMAGE_HTTP_PORT}/image/"
        f"{mission_id}/{violation_id}.jpg?exp={exp}&sig={sig}"
    )


def _sign_image_url(mission_id, violation_id, exp):
    payload = f"{mission_id}:{violation_id}:{exp}".encode("utf-8")
    secret_key = str(IMAGE_URL_SIGNING_SECRET).encode("utf-8")
    return hmac.new(
        secret_key,
        payload,
        hashlib.sha256,
    ).hexdigest()


def _is_valid_image_token(mission_id, violation_id, exp, sig):
    try:
        exp_int = int(exp)
    except (TypeError, ValueError):
        return False

    if exp_int < int(time.time()):
        return False

    expected = _sign_image_url(mission_id, violation_id, exp_int)
    return hmac.compare_digest(expected, sig)


class _ImageRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        parts = parsed.path.strip("/").split("/")
        if len(parts) != 3 or parts[0] != "image" or not parts[2].endswith(".jpg"):
            self.send_response(404)
            self.end_headers()
            return

        mission_id = parts[1]
        violation_part = parts[2][:-4]
        try:
            violation_id = int(violation_part)
        except ValueError:
            self.send_response(400)
            self.end_headers()
            return

        query = parse_qs(parsed.query)
        exp = query.get("exp", [None])[0]
        sig = query.get("sig", [None])[0]
        if exp is None or sig is None or not _is_valid_image_token(mission_id, violation_id, exp, sig):
            self.send_response(403)
            self.end_headers()
            return

        image_path = _build_image_path(mission_id, violation_id)
        if not os.path.exists(image_path):
            self.send_response(404)
            self.end_headers()
            return

        try:
            with open(image_path, "rb") as f:
                image_data = f.read()
        except OSError:
            self.send_response(500)
            self.end_headers()
            return

        self.send_response(200)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(image_data)))
        self.end_headers()
        self.wfile.write(image_data)

    def log_message(self, format, *args):
        return


def _start_image_server():
    global image_http_server, image_http_thread

    image_http_server = ThreadingHTTPServer(
        (IMAGE_HTTP_BIND_HOST, IMAGE_HTTP_PORT),
        _ImageRequestHandler,
    )
    image_http_thread = threading.Thread(
        target=image_http_server.serve_forever,
        daemon=True,
    )
    image_http_thread.start()
    print(f"Image HTTP server running on {IMAGE_HTTP_BIND_HOST}:{IMAGE_HTTP_PORT}")


def _stop_image_server():
    global image_http_server
    if image_http_server is not None:
        image_http_server.shutdown()
        image_http_server.server_close()
        image_http_server = None


def _handle_image_request(data):
    mission_id = data.get("mission_id")
    violation_id = data.get("violation_id")

    if not mission_id or violation_id is None:
        _publish_image_response(
            {
                "event": "image_response",
                "status": "error",
                "error": "mission_id and violation_id are required",
                "timestamp": _now_iso(),
            }
        )
        return

    try:
        violation_id = int(violation_id)
    except (TypeError, ValueError):
        _publish_image_response(
            {
                "event": "image_response",
                "status": "error",
                "mission_id": mission_id,
                "error": "violation_id must be an integer",
                "timestamp": _now_iso(),
            }
        )
        return

    image_path = _build_image_path(mission_id, violation_id)
    if not os.path.exists(image_path):
        _publish_image_response(
            {
                "event": "image_response",
                "status": "error",
                "mission_id": mission_id,
                "violation_id": violation_id,
                "error": "image_not_found",
                "timestamp": _now_iso(),
            }
        )
        return

    _publish_image_response(
        {
            "event": "image_response",
            "status": "ok",
            "mission_id": mission_id,
            "violation_id": violation_id,
            "image_url": _build_image_url(mission_id, violation_id),
            "timestamp": _now_iso(),
        }
    )


def perform_return_home():
    print("Returning home...")


def telemetry_worker():
    while telemetry_running.is_set():
        with telemetry_lock:
            simulator.update()
            telemetry = simulator.get_telemetry()
        client.publish(TELEMETRY_TOPIC, json.dumps(telemetry), qos=0)

        # Boundary check: if a square boundary has been configured for the simulator,
        # publish a boundary warning event when the telemetry leaves that area.
        try:
            lat = telemetry.get("lat")
            lon = telemetry.get("lon")
            if lat is not None and lon is not None and simulator.point_outside_boundary(lat, lon):
                warning_event = {
                    "event": "boundary_warning",
                    "mission_id": current_mission_id,
                    "warning": "out_of_bounds",
                    "lat": lat,
                    "lon": lon,
                    "timestamp": telemetry.get("timestamp"),
                    "source": "telemetry_path_boundary",
                }
                client.publish(DETECTION_TOPIC, json.dumps(_to_json_safe(warning_event)), qos=1)
        except Exception as e:
            # non-fatal; log and continue
            print(f"Boundary check error: {e}")

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


def publish_cv_detection(detection_payload, frame=None):
    global next_violation_id

    with telemetry_lock:
        telemetry = simulator.get_telemetry()

    violation_id = None
    if (
        detection_payload.get("state") == "violation"
        and current_mission_id is not None
        and frame is not None
    ):
        violation_id = next_violation_id
        saved_path = _save_violation_snapshot(current_mission_id, violation_id, frame)
        if saved_path is not None:
            next_violation_id += 1
        else:
            violation_id = None

    event = {
        "event": "cv_detection",
        "mission_id": current_mission_id,
        "violation_id": violation_id,
        "person_id": detection_payload.get("person_id"),
        "state": detection_payload.get("state", "unknown"),
        "violation_type": detection_payload.get("violation_type", "none"),
        "person_confidence": detection_payload.get("person_confidence", 0.0),
        "helmet_confidence": detection_payload.get("helmet_confidence", 0.0),
        "vest_confidence": detection_payload.get("vest_confidence", 0.0),
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


def perform_start_telemetry_path(following: bool = True):
    end_lat = getattr(config, "TELEMETRY_PATH_END_LAT", None)
    end_lon = getattr(config, "TELEMETRY_PATH_END_LON", None)
    # try to coerce to floats if provided
    try:
        if end_lat is not None:
            end_lat = float(end_lat)
        if end_lon is not None:
            end_lon = float(end_lon)
    except Exception:
        end_lat = end_lon = None

    # Apply the two-point path
    if following:
        simulator.set_path_follow(end_lat=end_lat, end_lon=end_lon)
    else:
        simulator.set_path_follow_fail(end_lat=end_lat, end_lon=end_lon)

    # Build a small square (2m half-size) around the two-point path and use it as the configured boundary
    # but do NOT switch waypoints to patrol that square (keep two-point path).
    boundary_meters = getattr(config, "TELEMETRY_PATH_BOUNDARY_METERS", 2)
    simulator.set_square_boundary(meters=boundary_meters, end_lat=end_lat, end_lon=end_lon, set_waypoints=False)

    if telemetry_running.is_set():
        # already running: ensure the simulator picks up the new path and boundary immediately
        with telemetry_lock:
            simulator.last_update = time.time()
        print(f"Telemetry running: switched to two-point path with square boundary ({boundary_meters}m).")
        return

    # not running: start telemetry (which will use the two-point path we just set)
    print(f"Starting telemetry on two-point path with square boundary ({boundary_meters}m)...")
    telemetry_running.set()
    global telemetry_thread
    telemetry_thread = threading.Thread(target=telemetry_worker, daemon=True)
    telemetry_thread.start()

COMMAND_MAP = {
    "START_CV": perform_start_cv,
    "STOP_CV": perform_stop_cv,
    "START_TELEMETRY": perform_start_telemetry,
    "STOP_TELEMETRY": perform_stop_telemetry,
    "START_TELEMETRY_PATH": lambda: perform_start_telemetry_path(following=True),
    "START_TELEMETRY_PATH_FAIL": lambda: perform_start_telemetry_path(following=False),
}


def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Connected to MQTT broker.")
        client.subscribe(COMMAND_TOPIC, qos=1)
        client.subscribe(IMAGE_REQUEST_TOPIC, qos=1)
        client.publish(STATUS_TOPIC, "online", qos=1, retain=True)
        print(f"Subscribed to {COMMAND_TOPIC} and {IMAGE_REQUEST_TOPIC}")
    else:
        print(f"Failed to connect, rc={rc}")


def on_disconnect(client, userdata, rc):
    if rc != 0:
        print("Unexpected disconnection. Triggering RTH.")
        perform_return_home()


def on_message(client, userdata, message):
    try:
        payload = message.payload.decode("utf-8")
        data = json.loads(payload)

        if message.topic == COMMAND_TOPIC:
            cmd = data.get("cmd")
            action = COMMAND_MAP.get(cmd)

            if action:
                print(f"Received command: {cmd}")
                action()
            else:
                print(f"Unknown command: {cmd}")
        elif message.topic == IMAGE_REQUEST_TOPIC:
            _handle_image_request(data)
        else:
            return
    except json.JSONDecodeError:
        print(f"Invalid JSON on {message.topic}: {message.payload!r}")
    except Exception as e:
        print(f"Command handling error: {e}")


client = mqtt.Client(client_id="droneOS")
client.will_set(STATUS_TOPIC, "offline", qos=1, retain=True)

client.on_connect = on_connect
client.on_disconnect = on_disconnect
client.on_message = on_message

_start_image_server()
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
    _stop_image_server()
