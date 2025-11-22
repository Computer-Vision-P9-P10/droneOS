import requests
import time
import httpx
import asyncio
import cv2
from ultralytics import YOLO

model = YOLO("yolo12n.pt")
cam = cv2.VideoCapture(0) # change number depending on how it sees the drone
backend_host = "0.0.0.0"

frame_interval = 1  # control report frequency
frame_count = 0 

def validate_connection():
    pass

def alert():
    pass

async def on_person_detected():
    print("Person detected! Executing action...")
    url = f"{backend_host}/violation"
    data = {"event": "person_detected", "timestamp": time.time()}
    async with httpx.AsyncClient() as client:
        try:
            response = client.post(url, json=data)
            if response.status_code == 200:
                print("API request successful")
            else:
                print(f"API returned status code {response.status_code}")
        except Exception as e:
            print(f"Failed to send API request: {e}")


while True:
    ret, frame = cam.read()
    if not ret:
        break

    results = model(frame)

    frame_count += 1

    if frame_count % frame_interval == 0:
        for detection in results[0].boxes.data:
            class_id = int(detection[5])
            if class_id == 0:
                asyncio.run(on_person_detected())
                break

    cv2.imshow('YOLO Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
