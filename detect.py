import cv2
from ultralytics import YOLO
import numpy as np
import time

from zoom_controller import ZoomController
from events import trigger_on_person_detected
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

backend_host = config.BACKEND_HOST

person_conf = config.PERSON_CONF
vest_conf = config.VEST_CONF
helmet_conf = config.HELMET_CONF
confidence = config.CONFIDENCE
iou = config.IOU
frame_interval = config.FRAME_INTERVAL

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
on_person_detected_count = 0
processed_frames = 0

person_history = {}
tracked_people = []


def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih

    a_area = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    b_area = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = a_area + b_area - inter

    return inter / union if union > 0 else 0.0


def sub_box(person_box, y_start_ratio, y_end_ratio):
    x1, y1, x2, y2 = person_box
    h = y2 - y1
    return [x1, y1 + h * y_start_ratio, x2, y1 + h * y_end_ratio]


def best_region_match(person_box, ppe_boxes, region="full", min_iou=0.1):
    if region == "helmet":
        target_box = sub_box(person_box, 0.0, 0.4)
    elif region == "vest":
        target_box = sub_box(person_box, 0.3, 0.75)
    else:
        target_box = person_box

    best = None
    best_score = 0.0
    for ppe in ppe_boxes:
        score = iou_xyxy(target_box, ppe[:4])
        if score > best_score and score >= min_iou:
            best_score = score
            best = ppe
    return best


def is_compliant(hist, threshold=PPE_COMPLIANCE_THRESHOLD, min_frames=PPE_COMPLIANCE_MIN_FRAMES):
    frames = hist.get("frames", 0)
    if frames < min_frames:
        return None
    vest_ratio = hist.get("vest_frames", 0) / frames
    helmet_ratio = hist.get("helmet_frames", 0) / frames
    return vest_ratio >= threshold and helmet_ratio >= threshold


def compliance_color(hist):
    result = is_compliant(hist)
    if result is None:
        return (0, 255, 255)   # yellow
    return (0, 200, 0) if result else (0, 0, 255)  # green or red


def cleanup_person_history(person_history, frame_count, min_frames, stale_frames):
    to_delete = []
    for pid, hist in person_history.items():
        frames = hist.get("frames", 0)
        last_seen = hist.get("last_seen_frame", 0)
        is_stale = (frame_count - last_seen) > stale_frames
        is_short = frames < min_frames and is_stale
        if is_short:
            to_delete.append(pid)

    for pid in to_delete:
        del person_history[pid]


def draw_top_left_overlay(frame, tracked_people, person_history):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.58
    thickness = 2
    line_height = 24
    padding = 10
    margin = 12

    active_ids = sorted([
        int(p["track_id"])
        for p in tracked_people
        if p["track_id"] is not None
    ])

    lines = []
    lines.append("Active IDs: " + (", ".join(map(str, active_ids)) if active_ids else "none"))

    for person in tracked_people:
        pid = person["track_id"]
        conf = person["conf"]
        has_vest_now = person.get("has_vest_now", False)
        has_helmet_now = person.get("has_helmet_now", False)

        if pid is None:
            lines.append(
                f"ID ?  P:{conf:.2f}  V:{'Y' if has_vest_now else 'N'}  H:{'Y' if has_helmet_now else 'N'}"
            )
            continue

        pid = int(pid)
        hist = person_history.get(pid)

        if hist is None:
            lines.append(
                f"ID {pid}  P:{conf:.2f}  V:{'Y' if has_vest_now else 'N'}  H:{'Y' if has_helmet_now else 'N'}"
            )
            continue

        frames = max(hist.get("frames", 0), 1)
        vest_pct = int(100 * hist.get("vest_frames", 0) / frames)
        helmet_pct = int(100 * hist.get("helmet_frames", 0) / frames)

        compliant = is_compliant(hist)
        if compliant is True:
            status = "OK"
        elif compliant is False:
            status = "NO PPE"
        else:
            status = "..."

        lines.append(
            f"ID {pid}  P:{conf:.2f}  V:{'Y' if has_vest_now else 'N'}({vest_pct}%)  H:{'Y' if has_helmet_now else 'N'}({helmet_pct}%)  {status}"
        )

    max_width = 0
    for line in lines:
        (w, _), _ = cv2.getTextSize(line, font, font_scale, thickness)
        max_width = max(max_width, w)

    box_width = max_width + padding * 2
    box_height = len(lines) * line_height + padding * 2

    x1 = margin
    y1 = margin
    x2 = x1 + box_width
    y2 = y1 + box_height

    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    for i, line in enumerate(lines):
        y = y1 + padding + 18 + i * line_height
        cv2.putText(
            frame,
            line,
            (x1 + padding, y),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )


while True:
    t0 = time.time()
    ret, frame = cap.read()
    t1 = time.time()
    if not ret:
        break

    frame = zoom_controller.update_zoom(frame, person_boxes)
    t4 = time.time()

    track_kwargs = dict(
        conf=confidence,
        iou=iou,
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

    filtered_boxes = []
    tracked_people = []
    active_ids = []
    class_names = model.names if hasattr(model, "names") else {}

    if frame_count % frame_interval == 0:
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

                if label == "person" and conf >= person_conf:
                    filtered_boxes.append(box_row)
                    tracked_people.append({
                        "track_id": track_ids[i],
                        "box": [x1, y1, x2, y2],
                        "conf": conf,
                        "class_id": class_id,
                        "has_vest_now": False,
                        "has_helmet_now": False,
                    })
                elif label == "vest" and conf >= vest_conf:
                    filtered_boxes.append(box_row)
                elif label == "helmet" and conf >= helmet_conf:
                    filtered_boxes.append(box_row)

        define_boxes(filtered_boxes, model)

        if len(person_boxes) > 0 and len(vest_boxes) == 0 and len(helmet_boxes) == 0:
            zoom_controller.increment_zoom_only_person_frames()
        else:
            zoom_controller.reset_zoom_only_person_frames()

        if zoom_controller.should_zoom_in():
            zoom_controller.zoomed_in = True

        if zoom_controller.zoomed_in and len(person_boxes) == 0:
            zoom_controller.disable_zoom()

        for person in tracked_people:
            track_id = person["track_id"]
            if track_id is None:
                continue

            pid = int(track_id)
            active_ids.append(pid)
            person_box = person["box"]

            vest_match = best_region_match(person_box, vest_boxes, region="vest", min_iou=0.15)
            helmet_match = best_region_match(person_box, helmet_boxes, region="helmet", min_iou=0.10)

            person["has_vest_now"] = vest_match is not None
            person["has_helmet_now"] = helmet_match is not None

            hist = person_history.setdefault(pid, {
                "frames": 0,
                "vest_frames": 0,
                "helmet_frames": 0,
                "detected": False,
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

        active_ids = sorted(set(active_ids))

        cleanup_person_history(
            person_history,
            frame_count,
            MIN_TRACK_FRAMES,
            STALE_TRACK_FRAMES,
        )

        on_person_detected_count = trigger_on_person_detected(
            person_history, cap, on_person_detected_count, backend_host
        )
    else:
        active_ids = sorted({
            int(p["track_id"])
            for p in tracked_people
            if p["track_id"] is not None
        })

    if not config.CONSOLE_OUTPUT:
        for person in tracked_people:
            x1, y1, x2, y2 = map(int, person["box"])
            pid = person["track_id"]
            person_confidence = person["conf"]

            hist = person_history.get(int(pid)) if pid is not None else None
            color = compliance_color(hist) if hist else (0, 255, 255)
            compliant = is_compliant(hist) if hist else None

            if compliant is True:
                status = "OK"
            elif compliant is False:
                status = "NO PPE"
            else:
                status = "..."

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

        for box in filtered_boxes:
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

        draw_top_left_overlay(frame, tracked_people, person_history)
    t6 = time.time()

    if config.CONSOLE_OUTPUT and frame_count % 30 == 0:
        fps = (
            processed_frames / (time.time() - start_time)
            if (time.time() - start_time) > 0
            else 0.0
        )
        detected_names = [
            class_names.get(int(box[5]), str(int(box[5])))
            for box in filtered_boxes
        ]
        print(
            f"[Frame {frame_count}] FPS: {fps:.1f} - Active IDs: {active_ids} - Detected: {len(filtered_boxes)} ({', '.join(detected_names)})"
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
print(f"'on_person_detected' called: {on_person_detected_count} times")

if person_history:
    print("\n=== PERSON HISTORY ===")
    print(f"  PPE compliance threshold: {PPE_COMPLIANCE_THRESHOLD:.0%} over {PPE_COMPLIANCE_MIN_FRAMES}+ frames")
    print(f"  Person conf: {person_conf} | Vest conf: {vest_conf} | Helmet conf: {helmet_conf}")
    print("-" * 40)

    for pid, hist in sorted(person_history.items()):
        total_frames = hist["frames"]
        vest_frames = hist["vest_frames"]
        helmet_frames = hist["helmet_frames"]

        vest_ratio = vest_frames / total_frames if total_frames > 0 else 0.0
        helmet_ratio = helmet_frames / total_frames if total_frames > 0 else 0.0

        compliant = is_compliant(hist)
        compliance_str = "COMPLIANT" if compliant else ("UNKNOWN" if compliant is None else "NON-COMPLIANT")

        print(f"Person ID {pid}: [{compliance_str}]")
        print(f"  Frames in view: {total_frames}")
        print(f"  Vest frames: {vest_frames} ({vest_ratio:.1%})")
        print(f"  Helmet frames: {helmet_frames} ({helmet_ratio:.1%})")
        print(f"  Detected flag: {hist.get('detected', False)}")
        print("-" * 40)
else:
    print("No persons tracked.")
