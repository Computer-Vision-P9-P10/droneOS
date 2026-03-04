import cv2
import config

def apply_filters(frame):
    frame = cv2.GaussianBlur(frame, (3, 3), 0)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(frame)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l_channel = clahe.apply(l_channel)
    frame = cv2.merge((l_channel, a, b))
    frame = cv2.cvtColor(frame, cv2.COLOR_LAB2BGR)
    return frame

def preprocess_frame(frame):
    target_size = config.DOWNSCALE_SIZE
    h, w = frame.shape[:2]
    if max(h, w) > target_size:
        scale = target_size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return frame
