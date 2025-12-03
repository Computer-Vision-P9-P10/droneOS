import httpx
import asyncio
import cv2
from ultralytics import YOLO
import numpy as np
import time

model = YOLO("yolo12n.pt")
# cap = cv2.VideoCapture(0) # change number depending on how it sees the drone
cap = cv2.VideoCapture("./videos/DJI_0017.MP4")
# cap = cv2.VideoCapture("./videos/amar/DJI_0017.MP4")
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit(1)

backend_host = "0.0.0.0"

# Parameters
person_conf = 0.4
vest_conf = 0.4
helmet_conf = 0.2
boots_conf = 0.1
gloves_conf = 0.1

confidence=0.1
frame_interval = 1  # control report frequency
frame_count = 0
on_person_detected_count = 0
filters_on = False
console_output = True

# Zoom
zoom_enabled = True
zoomed_in = False
zoom_factor = 2
zoom_min_duration = 0.5 # seconds
zoom_only_person_frames = 0
zoom_person_frame_threshold = 10
zoom_size_min_threshold = 0.4  # Min percent of person height in frame
zoom_size_max_threshold = 0.6  # Max percent of person height in frame
max_zoom_factor = 4.0
zoom_step = 0.5

# Tracking
person_history = {}  # {person_id: {"frames": n, "vest_frames": m, "helmet_frames": k}}
next_person_id = 0
person_boxes = []
vest_boxes = []
helmet_boxes = []
boots_boxes = []
gloves_boxes = []

def apply_filters(frame):
    frame = cv2.GaussianBlur(frame, (3, 3), 0)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(frame)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l_channel = clahe.apply(l_channel)
    frame = cv2.merge((l_channel, a, b))
    frame = cv2.cvtColor(frame, cv2.COLOR_LAB2BGR)
    return frame

def get_centroid(box):
    x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
    return ((x1 + x2) // 2, (y1 + y2) // 2)

# this needed?
def validate_connection():
    pass

async def violation_detected(violation:str):
    message = f"Missing {violation} detected!"
    url = f"{backend_host}/violation"
    data = {"message": message, "timestamp": ""}
    async with httpx.AsyncClient() as client:
        try:
            response = client.post(url, json=data)
            if response.status_code == 200:
                print("API request successful")
            else:
                print(f"API returned status code {response.status_code}")
        except Exception as e:
            print(f"Failed to send API request: {e}")

async def on_person_detected():
    print("Person detected! Executing action...")

def boxes_overlap(box1, box2):
    x1_min, y1_min, x1_max, y1_max = box1[0], box1[1], box1[2], box1[3]
    x2_min, y2_min, x2_max, y2_max = box2[0], box2[1], box2[2], box2[3]
    return not (x1_max < x2_min or x2_max < x1_min or y1_max < y2_min or y2_max < y1_min)

def define_boxes(boxes):
    class_names = model.names if hasattr(model, "names") else {}
    person_boxes.clear()
    vest_boxes.clear()
    helmet_boxes.clear()
    boots_boxes.clear()
    gloves_boxes.clear()
    for box in boxes:
        class_id = int(box[5])
        label = class_names.get(class_id, str(class_id)).lower()
        if label == "person":
            person_boxes.append(box)
        elif label == "vest":
            vest_boxes.append(box)
        elif label == "helmet":
            helmet_boxes.append(box)
        elif label == "boots":
            boots_boxes.append(box)
        elif label == "gloves":
            gloves_boxes.append(box)
    return person_boxes, vest_boxes, helmet_boxes, boots_boxes, gloves_boxes

def track_and_update_persons(person_boxes, vest_boxes, helmet_boxes, boots_boxes, gloves_boxes, person_history, next_person_id):
    if not person_boxes:
        return next_person_id

    current_centroids = [(p_box, get_centroid(p_box)) for p_box in person_boxes]

    # To track which person IDs have PPE counted this frame
    ppe_counted_ids = set()

    for p_box, p_centroid in current_centroids:
        best_match_id = None
        best_score = float('inf')

        for pid, hist in person_history.items():
            if "last_centroid" not in hist or "last_area" not in hist:
                continue

            last_centroid = hist["last_centroid"]
            last_area = hist["last_area"]

            centroid_dist = np.linalg.norm(np.array(p_centroid) - np.array(last_centroid))
            current_area = (p_box[2] - p_box[0]) * (p_box[3] - p_box[1])
            area_ratio = abs(current_area - last_area) / max(current_area, last_area)

            match_score = centroid_dist * (1 + area_ratio * 0.5)

            if match_score < 75:
                if match_score < best_score:
                    best_score = match_score
                    best_match_id = pid

        if best_match_id is None:
            best_match_id = next_person_id
            person_history[best_match_id] = {
                "frames": 0, "vest_frames": 0, "helmet_frames": 0,
                "boots_frames": 0, "gloves_frames": 0, "detected": False,
                "last_centroid": None, "last_area": 0
            }
            next_person_id += 1

        hist = person_history[best_match_id]
        hist["frames"] += 1
        hist["last_centroid"] = p_centroid
        hist["last_area"] = (p_box[2] - p_box[0]) * (p_box[3] - p_box[1])

        # Count PPE once per person per frame
        if best_match_id not in ppe_counted_ids:
            if any(boxes_overlap(p_box, v_box) for v_box in vest_boxes):
                hist["vest_frames"] += 1
            if any(boxes_overlap(p_box, h_box) for h_box in helmet_boxes):
                hist["helmet_frames"] += 1
            if any(boxes_overlap(p_box, b_box) for b_box in boots_boxes):
                hist["boots_frames"] += 1
            if any(boxes_overlap(p_box, g_box) for g_box in gloves_boxes):
                hist["gloves_frames"] += 1
            ppe_counted_ids.add(best_match_id)

    return next_person_id

def trigger_on_person_detected(person_history, cap, on_person_detected_count):
    for pid, hist in person_history.items():
        time_in_frame = hist["frames"] / cap.get(cv2.CAP_PROP_FPS)
        overlap_frames = hist["vest_frames"] + hist["helmet_frames"]
        if (
            hist["frames"] > 0
            and overlap_frames / hist["frames"] > 0.6
            and time_in_frame >= 10
            and not hist.get("detected", False)
        ):
            on_person_detected_count += 1
            asyncio.run(on_person_detected())
            hist["frames"] = 0
            hist["vest_frames"] = 0
            hist["helmet_frames"] = 0
            hist["boots_frames"] = 0
            hist["gloves_frames"] = 0
            hist["detected"] = True
            break
    return on_person_detected_count

def draw_boxes_on_frame(frame, boxes, model):
    class_names = model.names if hasattr(model, "names") else {}
    for box in boxes:
        x1, y1, x2, y2, conf, class_id = box
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        color = tuple(int(x) for x in list(cv2.cvtColor(
            np.uint8([[[(int(class_id) * 50) % 255, 255, 255]]]), cv2.COLOR_HSV2BGR
        )[0][0]))
        label = f"{class_names.get(int(class_id), str(int(class_id)))} {conf:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    return frame

def show_frame(frame):
    cv2.imshow('YOLO Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        return False
    return True

def print_detection_summary(person_history, boxes):
    class_names = model.names if hasattr(model, "names") else {}
    for pid, hist in person_history.items():
        print("Detection Summary:")
        print(f"Person ID: {pid}")
        print(f"  Frames in view: {hist['frames']}")
        print(f"  Vest frames: {hist['vest_frames']}")
        print(f"  Helmet frames: {hist['helmet_frames']}")
        for box in boxes:
            conf = box[4]
            class_id = int(box[5])
            obj_name = class_names.get(class_id, str(class_id))
            print(f"    Object class: {obj_name}, Confidence: {conf:.2f}")
        print("-" * 30)

start_time = time.time()
processed_frames = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    if filters_on:
        frame = apply_filters(frame)

    if len(person_boxes) > 0:
        h, w = frame.shape[:2]
        largest_person = max(person_boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
        px, py = get_centroid(largest_person)
        person_area = (largest_person[2] - largest_person[0]) * (largest_person[3] - largest_person[1])
        frame_area = w * h

        person_height = largest_person[3] - largest_person[1]
        person_ratio = person_height / h
        should_zoom = person_ratio < zoom_size_min_threshold or person_ratio > zoom_size_max_threshold

        if 'last_zoom_change' not in globals():
            last_zoom_change = 0
        now = time.time()
        can_change_zoom = (now - last_zoom_change) >= zoom_min_duration

        if zoom_enabled and can_change_zoom:
            if should_zoom:
                if person_ratio < zoom_size_min_threshold and (not zoomed_in or zoom_factor < max_zoom_factor):
                    zoomed_in = True
                    zoom_factor = min(zoom_factor + zoom_step, max_zoom_factor)
                    last_zoom_change = now
                elif person_ratio > zoom_size_max_threshold and (zoomed_in and zoom_factor > 1.0):
                    zoom_factor = max(zoom_factor - zoom_step, 1.0)
                    if zoom_factor == 1.0:
                        zoomed_in = False
                    last_zoom_change = now
            else:
                zoom_only_person_frames = 0

        if zoomed_in:
            nh, nw = int(h / zoom_factor), int(w / zoom_factor)

            if 'zoom_center' not in globals():
                zoom_center = [px, py]
            else:
                alpha = 0.15
                zoom_center[0] = int(zoom_center[0] * (1 - alpha) + px * alpha)
                zoom_center[1] = int(zoom_center[1] * (1 - alpha) + py * alpha)

            cx, cy = zoom_center
            x1 = int(max(0, min(w - nw, cx - nw // 2)))
            y1 = int(max(0, min(h - nh, cy - nh // 2)))
            frame = frame[y1:y1+nh, x1:x1+nw]
            frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR)
        else:
            zoom_only_person_frames = 0

    results = model(frame, conf=confidence, iou=0.5, imgsz=640, verbose=False)

    frame_count += 1
    processed_frames += 1

    if frame_count % frame_interval == 0:
        boxes = results[0].boxes.data
        filtered_boxes = []
        class_names = model.names if hasattr(model, "names") else {}
        for box in boxes:
            class_id = int(box[5])
            conf = box[4]
            label = class_names.get(class_id, str(class_id)).lower()
            if label == "person" and conf >= person_conf:
                filtered_boxes.append(box)
            elif label == "vest" and conf >= vest_conf:
                filtered_boxes.append(box)
            elif label == "helmet" and conf >= helmet_conf:
                filtered_boxes.append(box)
            elif label == "boots" and conf >= boots_conf:
                filtered_boxes.append(box)
            elif label == "gloves" and conf >= gloves_conf:
                filtered_boxes.append(box)

        person_boxes, vest_boxes, helmet_boxes, boots_boxes, gloves_boxes = define_boxes(filtered_boxes)
        if len(person_boxes) > 0 and len(vest_boxes) == 0 and len(helmet_boxes) == 0:
            zoom_only_person_frames += 1
        else:
            zoom_only_person_frames = 0

        if zoom_enabled:
            if zoom_only_person_frames >= zoom_person_frame_threshold and not zoomed_in:
                zoomed_in = True
            elif zoomed_in:
                if len(person_boxes) == 0:
                    zoomed_in = False
                    zoom_only_person_frames = 0

        next_person_id = track_and_update_persons(person_boxes, vest_boxes, helmet_boxes, boots_boxes, gloves_boxes, person_history, next_person_id)
        on_person_detected_count = trigger_on_person_detected(person_history, cap, on_person_detected_count)

    if console_output and frame_count % 30 == 0:
        fps = processed_frames / (time.time() - start_time) if (time.time() - start_time) > 0 else 0.0
        detected = len(filtered_boxes)
        detected_names = [model.names.get(int(box[5]), str(int(box[5]))) for box in filtered_boxes]
        print(f"[Frame {frame_count}] FPS: {fps:.1f} - Detected: {detected} ({', '.join(detected_names)})")
    else:
        if not console_output:
            frame = draw_boxes_on_frame(frame, filtered_boxes, model)
            if not show_frame(frame):
                break

cap.release()
cv2.destroyAllWindows()

elapsed = time.time() - start_time
real_fps = processed_frames / elapsed if elapsed > 0 else 0.0

print("\n\n" + "=" * 20 + " Results " + "=" * 20)
print(f"\nProcessed FPS (measured): {real_fps:.2f}")
print(f"'on_person_detected' called: {on_person_detected_count} times")

if person_history:
    print("\n=== PERSON HISTORY ===")
    print("Confidence thresholds:")
    print(f"  Person: {person_conf}")
    print(f"  Vest: {vest_conf}")
    print(f"  Helmet: {helmet_conf}")
    print(f"  Boots: {boots_conf}")
    print(f"  Gloves: {gloves_conf}")
    print("-" * 40)

    for pid, hist in person_history.items():
        total_frames = hist["frames"]
        vest_frames = hist["vest_frames"]
        helmet_frames = hist["helmet_frames"]
        vest_ratio = (vest_frames / total_frames) if total_frames > 0 else 0.0
        helmet_ratio = (helmet_frames / total_frames) if total_frames > 0 else 0.0

        print(f"Person ID {pid}:")
        print(f"  Frames in view: {total_frames}")
        print(f"  Vest frames: {vest_frames} ({vest_ratio:.1%})")
        print(f"  Helmet frames: {helmet_frames} ({helmet_ratio:.1%})")
        boots_frames = hist.get("boots_frames", 0)
        gloves_frames = hist.get("gloves_frames", 0)
        boots_ratio = (boots_frames / total_frames) if total_frames > 0 else 0.0
        gloves_ratio = (gloves_frames / total_frames) if total_frames > 0 else 0.0
        print(f"  Boots frames: {boots_frames} ({boots_ratio:.1%})")
        print(f"  Gloves frames: {gloves_frames} ({gloves_ratio:.1%})")
        print(f"  Detected flag: {hist.get('detected', False)}")
        print("-" * 40)
else:
    print("No persons tracked.")
