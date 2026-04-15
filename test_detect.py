import threading
import time

import cv2
from ultralytics import YOLO

import config
from box_utils import define_boxes, helmet_boxes, person_boxes, vest_boxes
from zoom_controller import ZoomController


SUPPORTED_CLASSES = ("person", "helmet", "vest")


def _predict(model, frame):
    if config.DEVICE == "cpu":
        return model.predict(
            frame, conf=config.CONFIDENCE, iou=config.IOU, imgsz=640, verbose=False
        )
    if config.DEVICE in ("cuda", "gpu"):
        return model.predict(
            frame,
            conf=config.CONFIDENCE,
            iou=config.IOU,
            imgsz=640,
            verbose=False,
            device=0,
        )
    raise ValueError(f"Unsupported DEVICE setting: {config.DEVICE}")


def _filter_boxes(results, model):
    boxes = results[0].boxes.data.cpu().numpy()
    class_names = model.names if hasattr(model, "names") else {}

    filtered_boxes = []
    for box in boxes:
        class_id = int(box[5])
        conf = float(box[4])
        label = class_names.get(class_id, str(class_id)).lower()

        if label == "person" and conf >= config.PERSON_CONF:
            filtered_boxes.append(box)
        elif label == "vest" and conf >= config.VEST_CONF:
            filtered_boxes.append(box)
        elif label == "helmet" and conf >= config.HELMET_CONF:
            filtered_boxes.append(box)

    return filtered_boxes


def _build_detection_summary():
    counts = {
        "person": len(person_boxes),
        "helmet": len(helmet_boxes),
        "vest": len(vest_boxes),
    }
    detections = [name for name in SUPPORTED_CLASSES if counts[name] > 0]
    return {"detections": detections, "counts": counts}


def run_detector(stop_event, on_detection=None):
    model = YOLO(config.MODEL_PATH, task="detect")
    cap = cv2.VideoCapture(config.VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError("Could not open configured video source")

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
    processed_frames = 0
    start_time = time.time()

    try:
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                break

            frame = zoom_controller.update_zoom(frame, person_boxes)
            results = _predict(model, frame)

            frame_count += 1
            processed_frames += 1
            filtered_boxes = []

            if frame_count % config.FRAME_INTERVAL == 0:
                filtered_boxes = _filter_boxes(results, model)
                define_boxes(filtered_boxes, model)

                if len(person_boxes) > 0 and len(vest_boxes) == 0 and len(helmet_boxes) == 0:
                    zoom_controller.increment_zoom_only_person_frames()
                else:
                    zoom_controller.reset_zoom_only_person_frames()

                if zoom_controller.should_zoom_in():
                    zoom_controller.zoomed_in = True

                if zoom_controller.zoomed_in and len(person_boxes) == 0:
                    zoom_controller.disable_zoom()

                payload = _build_detection_summary()
                if payload["detections"] and on_detection is not None:
                    on_detection(payload)

            if config.CONSOLE_OUTPUT is False:
                for box in filtered_boxes:
                    x1, y1, x2, y2 = map(int, box[:4])
                    conf = float(box[4])
                    class_id = int(box[5])
                    label = (
                        model.names.get(class_id, str(class_id))
                        if hasattr(model, "names")
                        else str(class_id)
                    )
                    color = (0, 255, 0) if label.lower() == "person" else (255, 0, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        frame,
                        f"{label} {conf:.2f}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        2,
                    )

                cv2.imshow("Video", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    stop_event.set()
                    break

            if config.CONSOLE_OUTPUT and frame_count % 30 == 0:
                elapsed = max(time.time() - start_time, 1e-9)
                fps = processed_frames / elapsed
                print(f"[Frame {frame_count}] FPS: {fps:.1f}")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    local_stop = threading.Event()

    def print_detection(payload):
        print(payload)

    try:
        run_detector(stop_event=local_stop, on_detection=print_detection)
    except KeyboardInterrupt:
        local_stop.set()
