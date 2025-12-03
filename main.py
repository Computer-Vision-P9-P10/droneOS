import asyncio
import cv2
from ultralytics import YOLO
import numpy as np
import time

from zoom_controller import ZoomController
from events import trigger_on_person_detected
import config
from filters import apply_filters
from box_utils import define_boxes, person_boxes, vest_boxes, helmet_boxes, boots_boxes, gloves_boxes

start_time = time.time()

# Choose your tracker import here
USE_SORT_TRACKER = config.USE_SORT_TRACKER
sort_tracker = None
if USE_SORT_TRACKER:
    from sort_tracker import SortTracker, sort_track_and_update_persons, boxes_overlap, get_centroid
    sort_tracker = SortTracker(max_age=30, min_hits=3, iou_threshold=0.3)
else:
    from centroid_tracker import track_and_update_persons, boxes_overlap, get_centroid

model = YOLO(config.MODEL_PATH)
cap = cv2.VideoCapture(config.VIDEO_PATH)
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit(1)

backend_host = config.BACKEND_HOST

person_conf = config.PERSON_CONF
vest_conf = config.VEST_CONF
helmet_conf = config.HELMET_CONF
boots_conf = config.BOOTS_CONF
gloves_conf = config.GLOVES_CONF
confidence = config.CONFIDENCE
frame_interval = config.FRAME_INTERVAL

zoom_controller = ZoomController(
    zoom_enabled=config.ZOOM_ENABLED,
    zoom_factor=config.ZOOM_FACTOR,
    zoom_min_duration=config.ZOOM_MIN_DURATION,
    zoom_person_frame_threshold=config.ZOOM_PERSON_FRAME_THRESHOLD,
    zoom_size_min_threshold=config.ZOOM_SIZE_MIN_THRESHOLD,
    zoom_size_max_threshold=config.ZOOM_SIZE_MAX_THRESHOLD,
    max_zoom_factor=config.MAX_ZOOM_FACTOR,
    zoom_step=config.ZOOM_STEP,
)

frame_count = 0
on_person_detected_count = 0
processed_frames = 0

person_history = {}
next_person_id = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if config.FILTERS_ON:
        frame = apply_filters(frame)

    frame = zoom_controller.update_zoom(frame, person_boxes)

    results = model(frame, conf=confidence, iou=0.3, imgsz=640, verbose=False)

    frame_count += 1
    processed_frames += 1

    if frame_count % frame_interval == 0:
        boxes = results[0].boxes.data.cpu().numpy()
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

        define_boxes(filtered_boxes, model)

        if len(person_boxes) > 0 and len(vest_boxes) == 0 and len(helmet_boxes) == 0:
            zoom_controller.increment_zoom_only_person_frames()
        else:
            zoom_controller.reset_zoom_only_person_frames()

        if zoom_controller.should_zoom_in():
            zoom_controller.zoomed_in = True

        if zoom_controller.zoomed_in:
            if len(person_boxes) == 0:
                zoom_controller.disable_zoom()

        if USE_SORT_TRACKER:
            sort_tracker, person_history = sort_track_and_update_persons(
                person_boxes, vest_boxes, helmet_boxes, boots_boxes, gloves_boxes,
                person_history, sort_tracker
            )
            next_person_id = max(person_history.keys(), default=-1) + 1 if person_history else 0
        else:
            next_person_id = track_and_update_persons(
                person_boxes, vest_boxes, helmet_boxes, boots_boxes, gloves_boxes,
                person_history, next_person_id
            )

        on_person_detected_count = trigger_on_person_detected(person_history, cap, on_person_detected_count, backend_host)

    if config.CONSOLE_OUTPUT and frame_count % 30 == 0:
        fps = processed_frames / (time.time() - start_time) if (time.time() - start_time) > 0 else 0.0
        detected = len(filtered_boxes)
        detected_names = [model.names.get(int(box[5]), str(int(box[5]))) for box in filtered_boxes]
        print(f"[Frame {frame_count}] FPS: {fps:.1f} - Detected: {detected} ({', '.join(detected_names)})")

cap.release()
cv2.destroyAllWindows()

elapsed = time.time() - zoom_controller.last_zoom_change
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
