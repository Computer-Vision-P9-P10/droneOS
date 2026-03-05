import paho.mqtt.client as mqtt
import json, time

BROKER = "localhost"
PORT = 1883

client = mqtt.Client(client_id="droneOS")
client.connect(BROKER, PORT)
client.loop_start()

while True:
    telemetry = {
        "lat": 57.048,
        "lon": 9.918,
        "altitude": 120,
        "battery": 87
    }
    client.publish("drone/telemetry", json.dumps(telemetry))
    time.sleep(1)  # Publishing speedz
