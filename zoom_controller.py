import cv2
import time
import numpy as np
from typing import List, Tuple

class ZoomController:
    def __init__(
        self,
        zoom_enabled=True,
        zoom_factor=2,
        zoom_min_duration=0.5,
        zoom_person_frame_threshold=10,
        zoom_size_min_threshold=0.4,
        zoom_size_max_threshold=0.6,
        max_zoom_factor=4.0,
        zoom_step=0.5,
    ):
        self.zoom_enabled = zoom_enabled
        self.zoomed_in = False
        self.zoom_factor = zoom_factor
        self.zoom_min_duration = zoom_min_duration
        self.zoom_only_person_frames = 0
        self.zoom_person_frame_threshold = zoom_person_frame_threshold
        self.zoom_size_min_threshold = zoom_size_min_threshold
        self.zoom_size_max_threshold = zoom_size_max_threshold
        self.max_zoom_factor = max_zoom_factor
        self.zoom_step = zoom_step
        self.last_zoom_change = 0
        self.zoom_center = None

    def update_zoom(self, frame, person_boxes):
        if not self.zoom_enabled or len(person_boxes) == 0:
            self.zoom_only_person_frames = 0
            return frame

        h, w = frame.shape[:2]
        largest_person = max(person_boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
        px = (largest_person[0] + largest_person[2]) // 2
        py = (largest_person[1] + largest_person[3]) // 2

        person_height = largest_person[3] - largest_person[1]
        person_ratio = person_height / h
        should_zoom = person_ratio < self.zoom_size_min_threshold or person_ratio > self.zoom_size_max_threshold

        now = time.time()
        can_change_zoom = (now - self.last_zoom_change) >= self.zoom_min_duration

        if can_change_zoom:
            if should_zoom:
                if person_ratio < self.zoom_size_min_threshold and (not self.zoomed_in or self.zoom_factor < self.max_zoom_factor):
                    self.zoomed_in = True
                    self.zoom_factor = min(self.zoom_factor + self.zoom_step, self.max_zoom_factor)
                    self.last_zoom_change = now
                elif person_ratio > self.zoom_size_max_threshold and (self.zoomed_in and self.zoom_factor > 1.0):
                    self.zoom_factor = max(self.zoom_factor - self.zoom_step, 1.0)
                    if self.zoom_factor == 1.0:
                        self.zoomed_in = False
                    self.last_zoom_change = now
            else:
                self.zoom_only_person_frames = 0

        if self.zoomed_in:
            nh, nw = int(h / self.zoom_factor), int(w / self.zoom_factor)

            if self.zoom_center is None:
                self.zoom_center = [px, py]
            else:
                alpha = 0.15
                self.zoom_center[0] = int(self.zoom_center[0] * (1 - alpha) + px * alpha)
                self.zoom_center[1] = int(self.zoom_center[1] * (1 - alpha) + py * alpha)

            cx, cy = self.zoom_center
            x1 = int(max(0, min(w - nw, cx - nw // 2)))
            y1 = int(max(0, min(h - nh, cy - nh // 2)))
            cropped_frame = frame[y1:y1+nh, x1:x1+nw]
            resized_frame = cv2.resize(cropped_frame, (w, h), interpolation=cv2.INTER_LINEAR)
            return resized_frame
        else:
            self.zoom_only_person_frames = 0
            return frame

    def increment_zoom_only_person_frames(self):
        self.zoom_only_person_frames += 1

    def reset_zoom_only_person_frames(self):
        self.zoom_only_person_frames = 0

    def should_zoom_in(self):
        return self.zoom_only_person_frames >= self.zoom_person_frame_threshold and not self.zoomed_in

    def disable_zoom(self):
        self.zoomed_in = False
        self.zoom_only_person_frames = 0
