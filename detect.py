import threading
import time

import cv2
import numpy as np
from ultralytics import YOLO

import config
from box_utils import define_boxes, helmet_boxes, person_boxes, vest_boxes
from events import collect_state_change_events
from person_state import cleanup_person_history, get_current_state, is_compliant
from ppe_matching import best_region_match
from visualization import compliance_color_from_state, draw_top_left_overlay
from zoom_controller import ZoomController


def _to_numpy(array_like):
    if hasattr(array_like, "cpu"):
        return array_like.cpu().numpy()
    return np.asarray(array_like)


def _to_track_ids(track_ids):
    if hasattr(track_ids, "int"):
        int_ids = track_ids.int()
        if hasattr(int_ids, "cpu"):
            return int_ids.cpu().tolist()
        return np.asarray(int_ids).astype(int).tolist()
    return np.asarray(track_ids).astype(int).tolist()


def _build_frame_summary(current_frame_people):
    counts = {
        "person": len(current_frame_people),
        "helmet": len(helmet_boxes),
        "vest": len(vest_boxes),
    }
    detections = [label for label, count in counts.items() if count > 0]
    return {"detections": detections, "counts": counts}


def _get_violation_type(state, vest_ratio, helmet_ratio, threshold):
    if state != "violation":
        return "none"

    missing_vest = vest_ratio < threshold
    missing_helmet = helmet_ratio < threshold

    if missing_vest and missing_helmet:
        return "missing_vest_and_helmet"
    elif missing_vest:
        return "missing_vest"
    elif missing_helmet:
        return "missing_helmet"
    return "unknown_violation"


def run_detector(stop_event, on_person_state_change=None, on_frame_summary=None):
    start_time = time.time()

    model = YOLO(config.MODEL_PATH, task="detect")
    cap = cv2.VideoCapture(config.VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError("Could not open configured video source")

    person_conf = getattr(config, "PERSON_CONF", 0.6)
    vest_conf = getattr(config, "VEST_CONF", 0.4)
    helmet_conf = getattr(config, "HELMET_CONF", 0.4)
    confidence = getattr(config, "CONFIDENCE", 0.1)
    iou = getattr(config, "IOU", 0.6)
    frame_interval = getattr(config, "FRAME_INTERVAL", 1)

    tracker_yaml = getattr(config, "TRACKER_YAML", "custom_bytetrack.yaml")
    min_track_frames = getattr(config, "MIN_TRACK_FRAMES", 20)
    stale_track_frames = getattr(config, "STALE_TRACK_FRAMES", 120)
    ppe_compliance_threshold = getattr(config, "PPE_COMPLIANCE_THRESHOLD", 0.70)
    show_live_feed = getattr(config, "SHOW_LIVE_FEED", False)

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

    try:
        while not stop_event.is_set():
            t0 = time.time()
            ret, frame = cap.read()
            t1 = time.time()
            if not ret:
                break

            frame = zoom_controller.update_zoom(frame, person_boxes)
            t4 = time.time()

            if config.DEVICE == "cpu":
                results = model.track(
                    frame,
                    conf=confidence,
                    iou=iou,
                    imgsz=640,
                    verbose=False,
                    persist=True,
                    tracker=tracker_yaml,
                )
            elif config.DEVICE == "cuda" or config.DEVICE == "gpu":
                results = model.track(
                    frame,
                    conf=confidence,
                    iou=iou,
                    imgsz=640,
                    verbose=False,
                    persist=True,
                    tracker=tracker_yaml,
                    device=0,
                )
            else:
                print("Please enter a supported device.")
                break
            t5 = time.time()

            frame_count += 1
            processed_frames += 1

            current_frame_boxes = []
            current_frame_people = []
            current_frame_track_ids = []
            pending_state_events = []
            class_names = model.names if hasattr(model, "names") else {}

            if frame_count % frame_interval == 0:
                result = results[0]
                boxes_obj = result.boxes

                if boxes_obj is not None and len(boxes_obj) > 0:
                    xyxy = _to_numpy(boxes_obj.xyxy)
                    confs = _to_numpy(boxes_obj.conf)
                    clss = _to_numpy(boxes_obj.cls).astype(int)

                    if boxes_obj.is_track and boxes_obj.id is not None:
                        track_ids = _to_track_ids(boxes_obj.id)
                    else:
                        track_ids = [None] * len(xyxy)

                    for i in range(len(xyxy)):
                        x1, y1, x2, y2 = xyxy[i]
                        conf = float(confs[i])
                        class_id = int(clss[i])
                        label = class_names.get(class_id, str(class_id)).lower()

                        box_row = np.array([x1, y1, x2, y2, conf, class_id], dtype=float)

                        if label == "person" and conf >= person_conf:
                            current_frame_boxes.append(box_row)
                            current_frame_people.append(
                                {
                                    "track_id": track_ids[i],
                                    "box": [x1, y1, x2, y2],
                                    "conf": conf,
                                    "has_vest_now": False,
                                    "has_helmet_now": False,
                                }
                            )
                        elif label == "vest" and conf >= vest_conf:
                            current_frame_boxes.append(box_row)
                        elif label == "helmet" and conf >= helmet_conf:
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

                    hist = person_history.setdefault(
                        pid,
                        {
                            "frames": 0,
                            "vest_frames": 0,
                            "helmet_frames": 0,
                            "state": "unknown",
                            "state_changed_at_frame": frame_count,
                            "last_sent_state": None,
                            "last_seen_frame": frame_count,
                            "last_box": person_box,
                            "last_person_conf": 0.0,
                            "last_vest_conf": 0.0,
                            "last_helmet_conf": 0.0,
                        },
                    )

                    hist["frames"] += 1
                    hist["last_seen_frame"] = frame_count
                    hist["last_box"] = person_box
                    hist["last_person_conf"] = float(person["conf"])
                    hist["last_vest_conf"] = float(vest_match[4]) if vest_match is not None else 0.0
                    hist["last_helmet_conf"] = float(helmet_match[4]) if helmet_match is not None else 0.0

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
                    min_track_frames,
                    stale_track_frames,
                )

                event_count, state_events = collect_state_change_events(
                    person_history,
                    cap,
                    event_count,
                )

                summary = _build_frame_summary(current_frame_people)
                summary["frame_count"] = frame_count
                if on_frame_summary is not None:
                    on_frame_summary(summary)

                if on_person_state_change is not None:
                    for event in state_events:
                        frames = max(event.get("frames", 0), 1)
                        event["vest_ratio"] = event.get("vest_frames", 0) / frames
                        event["helmet_ratio"] = event.get("helmet_frames", 0) / frames
                        event["violation_type"] = _get_violation_type(
                            event.get("state", "unknown"),
                            event["vest_ratio"],
                            event["helmet_ratio"],
                            ppe_compliance_threshold,
                        )
                        event["counts"] = summary["counts"]
                        event["detections"] = summary["detections"]
                        event["frame_count"] = frame_count
                        pending_state_events.append(event)

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

            if on_person_state_change is not None and pending_state_events:
                snapshot_frame = frame.copy()
                for event in pending_state_events:
                    on_person_state_change(event, snapshot_frame)
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

            if show_live_feed:
                cv2.imshow("Video", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    stop_event.set()
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()

        cleanup_person_history(
            person_history,
            frame_count + stale_track_frames + 1,
            min_track_frames,
            stale_track_frames,
        )

    elapsed = time.time() - start_time
    real_fps = processed_frames / elapsed if elapsed > 0 else 0.0

    print("\n\n" + "=" * 20 + " Results " + "=" * 20)
    print(f"\nProcessed FPS (measured): {real_fps:.2f}")
    print(f"'on_person_detected' called: {event_count} times")

    return {
        "event_count": event_count,
        "processed_frames": processed_frames,
        "person_history": person_history,
    }


if __name__ == "__main__":
    local_stop = threading.Event()

    def print_state_event(payload):
        print(payload)

    try:
        run_detector(stop_event=local_stop, on_person_state_change=print_state_event)
    except KeyboardInterrupt:
        local_stop.set()
