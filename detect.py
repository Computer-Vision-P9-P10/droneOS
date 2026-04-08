import cv2
from ultralytics import YOLO
import numpy as np
import time

from visualization import draw_top_left_overlay, compliance_color_from_state
from person_state import get_current_state, is_compliant, cleanup_person_history
from ppe_matching import best_region_match
from zoom_controller import ZoomController
from events import trigger_event
import config
from box_utils import (
    define_boxes,
    person_boxes,
    vest_boxes,
    helmet_boxes,
)

start_time = time.time()

model = YOLO(config.MODEL_PATH, task="detect")
cap = cv2.VideoCapture(config.VIDEO_PATH)
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit(1)

PERSON_CONF = getattr(config, "PERSON_CONF", 0.6)
VEST_CONF = getattr(config, "VEST_CONF", 0.4)
HELMET_CONF = getattr(config, "HELMET_CONF", 0.4)
CONFIDENCE = getattr(config, "CONFIDENCE", 0.1)
IOU = getattr(config, "IOU", 0.6)
FRAME_INTERVAL = getattr(config, "FRAME_INTERVAL", 1)

TRACKER_YAML = getattr(config, "TRACKER_YAML", "custom_bytetrack.yaml")
MIN_TRACK_FRAMES = getattr(config, "MIN_TRACK_FRAMES", 20)
STALE_TRACK_FRAMES = getattr(config, "STALE_TRACK_FRAMES", 120)
PPE_COMPLIANCE_THRESHOLD = getattr(config, "PPE_COMPLIANCE_THRESHOLD", 0.70)
PPE_COMPLIANCE_MIN_FRAMES = getattr(config, "PPE_COMPLIANCE_MIN_FRAMES", 10)

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
event_count = 0
processed_frames = 0
person_history = {}

while True:
    t0 = time.time()
    ret, frame = cap.read()
    t1 = time.time()
    if not ret:
        break

    frame = zoom_controller.update_zoom(frame, person_boxes)
    t4 = time.time()

    track_kwargs = dict(
        conf=CONFIDENCE,
        iou=IOU,
        imgsz=640,
        verbose=False,
        persist=True,
        tracker=TRACKER_YAML,
    )

    if config.DEVICE == "cpu":
        results = model.track(frame, **track_kwargs)
    elif config.DEVICE == "cuda" or config.DEVICE == "gpu":
        results = model.track(frame, device=0, **track_kwargs)
    else:
        print("Please enter a supported device.")
        break
    t5 = time.time()

    frame_count += 1
    processed_frames += 1

    current_frame_boxes = []
    current_frame_people = []
    current_frame_track_ids = []
    class_names = model.names if hasattr(model, "names") else {}

    if frame_count % FRAME_INTERVAL == 0:
        result = results[0]
        boxes_obj = result.boxes

        if boxes_obj is not None and len(boxes_obj) > 0:
            xyxy = boxes_obj.xyxy.cpu().numpy()
            confs = boxes_obj.conf.cpu().numpy()
            clss = boxes_obj.cls.cpu().numpy().astype(int)

            if boxes_obj.is_track and boxes_obj.id is not None:
                track_ids = boxes_obj.id.int().cpu().tolist()
            else:
                track_ids = [None] * len(xyxy)

            for i in range(len(xyxy)):
                x1, y1, x2, y2 = xyxy[i]
                conf = float(confs[i])
                class_id = int(clss[i])
                label = class_names.get(class_id, str(class_id)).lower()

                box_row = np.array([x1, y1, x2, y2, conf, class_id], dtype=float)

                if label == "person" and conf >= PERSON_CONF:
                    current_frame_boxes.append(box_row)
                    current_frame_people.append({
                        "track_id": track_ids[i],
                        "box": [x1, y1, x2, y2],
                        "conf": conf,
                        "has_vest_now": False,
                        "has_helmet_now": False,
                    })
                elif label == "vest" and conf >= VEST_CONF:
                    current_frame_boxes.append(box_row)
                elif label == "helmet" and conf >= HELMET_CONF:
                    current_frame_boxes.append(box_row)

        define_boxes(current_frame_boxes, model)

        for person in current_frame_people:
            track_id = person["track_id"]
            if track_id is None:
                continue

            pid = int(track_id)
            person_box = person["box"]

            vest_match = best_region_match(person_box, vest_boxes, region="vest", min_iou=0.15)
            helmet_match = best_region_match(person_box, helmet_boxes, region="helmet", min_iou=0.10)

            person["has_vest_now"] = vest_match is not None
            person["has_helmet_now"] = helmet_match is not None

            hist = person_history.setdefault(pid, {
                "frames": 0,
                "vest_frames": 0,
                "helmet_frames": 0,
                "state": "unknown",
                "state_changed_at_frame": frame_count,
                "last_sent_state": None,
                "last_seen_frame": frame_count,
                "last_box": person_box,
            })

            hist["frames"] += 1
            hist["last_seen_frame"] = frame_count
            hist["last_box"] = person_box

            if vest_match is not None:
                hist["vest_frames"] += 1
            if helmet_match is not None:
                hist["helmet_frames"] += 1

            current_state = is_compliant(hist)
            previous_state = hist.get("state", "unknown")

            if current_state != previous_state:
                hist["state"] = current_state
                hist["state_changed_at_frame"] = frame_count
            else:
                hist["state"] = current_state

        current_frame_track_ids = sorted(
            int(person["track_id"])
            for person in current_frame_people
            if person["track_id"] is not None
        )

        cleanup_person_history(
            person_history,
            frame_count,
            MIN_TRACK_FRAMES,
            STALE_TRACK_FRAMES,
        )

        event_count = trigger_event(
            person_history,
            cap,
            event_count,
        )

    if not config.CONSOLE_OUTPUT:
        for person in current_frame_people:
            x1, y1, x2, y2 = map(int, person["box"])
            pid = person["track_id"]
            person_confidence = person["conf"]

            hist = person_history.get(int(pid)) if pid is not None else None
            current_state = get_current_state(hist) if hist else "unknown"
            color = compliance_color_from_state(current_state)

            label = (
                f"ID {pid} Person:{person_confidence:.2f}"
                if pid is not None
                else f"person P:{person_confidence:.2f}"
            )

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                cv2.LINE_AA,
            )

        for box in current_frame_boxes:
            x1, y1, x2, y2 = map(int, box[:4])
            conf = box[4]
            class_id = int(box[5])
            label = class_names.get(class_id, str(class_id)).lower()

            if label != "person":
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 165, 0), 2)
                cv2.putText(
                    frame,
                    f"{label} {conf:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 165, 0),
                    1,
                    cv2.LINE_AA,
                )

        draw_top_left_overlay(frame, current_frame_people, person_history)
    t6 = time.time()

    if config.CONSOLE_OUTPUT and frame_count % 30 == 0:
        fps = (
            processed_frames / (time.time() - start_time)
            if (time.time() - start_time) > 0
            else 0.0
        )
        detected_names = [
            class_names.get(int(box[5]), str(int(box[5])))
            for box in current_frame_boxes
        ]
        print(
            f"[Frame {frame_count}] FPS: {fps:.1f} - Active IDs: {current_frame_track_ids} - Detected: {len(current_frame_boxes)} ({', '.join(detected_names)})"
        )
        print(
            f"Timing (ms): read={1000 * (t1 - t0):.1f}, zoom={1000 * (t4 - t1):.1f}, inference={1000 * (t5 - t4):.1f}, draw={1000 * (t6 - t5):.1f}"
        )

    if not config.CONSOLE_OUTPUT:
        cv2.imshow("Video", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()

cleanup_person_history(
    person_history,
    frame_count + STALE_TRACK_FRAMES + 1,
    MIN_TRACK_FRAMES,
    STALE_TRACK_FRAMES,
)

elapsed = time.time() - start_time
real_fps = processed_frames / elapsed if elapsed > 0 else 0.0

print("\n\n" + "=" * 20 + " Results " + "=" * 20)
print(f"\nProcessed FPS (measured): {real_fps:.2f}")
print(f"'on_person_detected' called: {event_count} times")

if person_history:
    print("\n=== PERSON HISTORY ===")
    print(f"  PPE compliance threshold: {PPE_COMPLIANCE_THRESHOLD:.0%} over {PPE_COMPLIANCE_MIN_FRAMES}+ frames")
    print(f"  Person conf: {PERSON_CONF} | Vest conf: {VEST_CONF} | Helmet conf: {HELMET_CONF}")
    print("-" * 40)

    for pid, hist in sorted(person_history.items()):
        total_frames = hist["frames"]
        vest_frames = hist["vest_frames"]
        helmet_frames = hist["helmet_frames"]

        vest_ratio = vest_frames / total_frames if total_frames > 0 else 0.0
        helmet_ratio = helmet_frames / total_frames if total_frames > 0 else 0.0

        compliance_state = get_current_state(hist)

        if compliance_state == "compliant":
            compliance_str = "COMPLIANT"
        elif compliance_state == "violation":
            compliance_str = "VIOLATION"
        else:
            compliance_str = "STATE UNKNOWN"

        print(f"Person ID {pid}: [{compliance_str}]")
        print(f"  Frames in view: {total_frames}")
        print(f"  Vest frames: {vest_frames} ({vest_ratio:.1%})")
        print(f"  Helmet frames: {helmet_frames} ({helmet_ratio:.1%})")
        print(f"  Current state: {hist.get('state', 'unknown')}")
        print(f"  Last sent state: {hist.get('last_sent_state', None)}")
        print("-" * 40)
else:
    print("No persons tracked.")
