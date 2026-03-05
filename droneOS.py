import paho.mqtt.client as mqtt
import json
import time

BROKER = "localhost"
PORT = 1883

TELEMETRY_TOPIC = "drone/telemetry"
STATUS_TOPIC = "drone/status"
COMMAND_TOPIC = "drone/command"

def perform_return_home():
    print("Returning home...")

def perform_land():
    print("Landing now...")

def perform_hover():
    print("Hovering...")

COMMAND_MAP = {
    "RETURN_HOME": perform_return_home,
    "LAND": perform_land,
    "HOVER": perform_hover,
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
        telemetry = {
            "lat": 57.048,
            "lon": 9.918,
            "altitude": 120,
            "battery": 87
        }
        client.publish(TELEMETRY_TOPIC, json.dumps(telemetry), qos=0)
        time.sleep(1)
except KeyboardInterrupt:
    print("Shutting down...")
finally:
    client.publish(STATUS_TOPIC, "offline", qos=1, retain=True)
    client.loop_stop()
    client.disconnect()
