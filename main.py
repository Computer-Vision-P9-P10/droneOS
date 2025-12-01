import requests
import time
import httpx
import asyncio
import cv2
from ultralytics import YOLO
import numpy as np

model = YOLO("yolo12s.pt")
# cap = cv2.VideoCapture(0) # change number depending on how it sees the drone
cap = cv2.VideoCapture("./videos/DJI_0017.MP4")
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit(1)
# Set to highest available resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, cap.get(cv2.CAP_PROP_FRAME_WIDTH))
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

backend_host = "0.0.0.0"

# Parameters
confidence=0.25
frame_interval = 1  # control report frequency
frame_count = 0
on_person_detected_count = 0


person_history = {}  # {person_id: {"frames": n, "vest_frames": m, "helmet_frames": k}}
next_person_id = 0

def get_centroid(box):
    x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
    return ((x1 + x2) // 2, (y1 + y2) // 2)

def validate_connection():
    pass

def alert():
    pass

async def on_person_detected():
    print("Person detected! Executing action...")
    # url = f"{backend_host}/violation"
    # data = {"event": "person_detected", "timestamp": time.time()}
    # async with httpx.AsyncClient() as client:
    #     try:
    #         response = client.post(url, json=data)
    #         if response.status_code == 200:
    #             print("API request successful")
    #         else:
    #             print(f"API returned status code {response.status_code}")
    #     except Exception as e:
    #         print(f"Failed to send API request: {e}")


def boxes_overlap(box1, box2):
    x1_min, y1_min, x1_max, y1_max = box1[0], box1[1], box1[2], box1[3]
    x2_min, y2_min, x2_max, y2_max = box2[0], box2[1], box2[2], box2[3]
    return not (x1_max < x2_min or x2_max < x1_min or y1_max < y2_min or y2_max < y1_min)


def non_max_suppression(boxes, iou_threshold=0.5):
    if len(boxes) == 0:
        return []
    boxes = sorted(boxes, key=lambda x: x[4], reverse=True)
    keep = []
    while boxes:
        box = boxes.pop(0)
        keep.append(box)
        boxes = [
            b for b in boxes
            if int(b[5]) != int(box[5]) or iou(box, b) < iou_threshold
        ]
    return keep

# intersection handling
def iou(box1, box2):
    x1_min, y1_min, x1_max, y1_max = box1[0], box1[1], box1[2], box1[3]
    x2_min, y2_min, x2_max, y2_max = box2[0], box2[1], box2[2], box2[3]
    xi1 = max(x1_min, x2_min)
    yi1 = max(y1_min, y2_min)
    xi2 = min(x1_max, x2_max)
    yi2 = min(y1_max, y2_max)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocessing: mild denoising and strong contrast enhancement for small objects like helmets
    frame = cv2.GaussianBlur(frame, (3, 3), 0)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(frame)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))  # Increased clipLimit for more contrast
    l_channel = clahe.apply(l_channel)
    frame = cv2.merge((l_channel, a, b))
    frame = cv2.cvtColor(frame, cv2.COLOR_LAB2BGR)

    results = model(frame, conf=confidence)

    frame_count += 1

    if frame_count % frame_interval == 0:
        boxes = results[0].boxes.data
        class_names = model.names if hasattr(model, "names") else {}
        person_boxes = []
        vest_boxes = []
        helmet_boxes = []
        for box in boxes:
            class_id = int(box[5])
            label = class_names.get(class_id, str(class_id)).lower()
            if label == "person":
                person_boxes.append(box)
            elif label == "vest":
                vest_boxes.append(box)
            elif label == "helmet":
                helmet_boxes.append(box)

        # Assign IDs to persons using centroid proximity (simple tracking)
        current_persons = []
        for p_box in person_boxes:
            p_centroid = get_centroid(p_box)
            matched_id = None
            for pid, hist in person_history.items():
                last_centroid = hist.get("last_centroid")
                if last_centroid and np.linalg.norm(np.array(p_centroid) - np.array(last_centroid)) < 50:
                    matched_id = pid
                    break
            if matched_id is None:
                matched_id = next_person_id
                person_history[matched_id] = {"frames": 0, "vest_frames": 0, "helmet_frames": 0}
                next_person_id += 1
            person_history[matched_id]["frames"] += 1
            person_history[matched_id]["last_centroid"] = p_centroid

            # Check overlaps
            vest_overlap = any(boxes_overlap(p_box, v_box) for v_box in vest_boxes)
            helmet_overlap = any(boxes_overlap(p_box, h_box) for h_box in helmet_boxes)
            if vest_overlap:
                person_history[matched_id]["vest_frames"] += 1
            if helmet_overlap:
                person_history[matched_id]["helmet_frames"] += 1

        # Trigger only if vest or helmet detected on person for >60% of frames AND at least 10 seconds in frame
        for pid, hist in person_history.items():
            time_in_frame = hist["frames"] / cap.get(cv2.CAP_PROP_FPS)
            overlap_frames = hist["vest_frames"] + hist["helmet_frames"]
            if hist["frames"] > 0 and overlap_frames / hist["frames"] > 0.6 and time_in_frame >= 10:
                on_person_detected_count += 1
                asyncio.run(on_person_detected())
                break

    for box in boxes:
        x1, y1, x2, y2, conf, class_id = box
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        class_names = model.names if hasattr(model, "names") else {}
        color = tuple(int(x) for x in list(cv2.cvtColor(
            np.uint8([[[(int(class_id) * 50) % 255, 255, 255]]]), cv2.COLOR_HSV2BGR
        )[0][0])) 
        label = f"{class_names.get(int(class_id), str(int(class_id)))} {conf:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    cv2.imshow('YOLO Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"FPS: {fps:.2f}")
print(f"'on_person_detected' called: {on_person_detected_count} times")
