import cv2

def apply_filters(frame):
    frame = cv2.GaussianBlur(frame, (3, 3), 0)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(frame)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l_channel = clahe.apply(l_channel)
    frame = cv2.merge((l_channel, a, b))
    frame = cv2.cvtColor(frame, cv2.COLOR_LAB2BGR)
    return frame
