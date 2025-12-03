import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment

class SortTracker:
    def __init__(self, max_age=30, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0
        self.next_id = 0

    class KalmanBoxTracker:
        def __init__(self, bbox, track_id):
            self.kf = KalmanFilter(dim_x=7, dim_z=4)
            self.kf.F = np.array([[1,0,0,0,1,0,0],
                                  [0,1,0,0,0,1,0],
                                  [0,0,1,0,0,0,1],
                                  [0,0,0,1,0,0,0],
                                  [0,0,0,0,1,0,0],
                                  [0,0,0,0,0,1,0],
                                  [0,0,0,0,0,0,1]])
            self.kf.H = np.array([[1,0,0,0,0,0,0],
                                  [0,1,0,0,0,0,0],
                                  [0,0,1,0,0,0,0],
                                  [0,0,0,1,0,0,0]])
            self.kf.R[2:,2:] *= 10.
            self.kf.P[4:,4:] *= 1000.
            self.kf.P *= 10.

            self.kf.x[:4] = self.convert_bbox_to_z(bbox)
            self.time_since_update = 0
            self.id = track_id
            self.history = []
            self.hits = 1
            self.hit_streak = 1
            self.age = 0

        def convert_bbox_to_z(self, bbox):
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            x = bbox[0] + w/2.
            y = bbox[1] + h/2.
            s = w * h
            r = w / float(h)
            return np.array([x, y, s, r]).reshape((4,1))

        def convert_x_to_bbox(self, x):
            x_c, y_c, s, r = x[0], x[1], x[2], x[3]
            w = np.sqrt(s * r)
            h = s / w
            return np.array([x_c - w/2., y_c - h/2., x_c + w/2., y_c + h/2.])

        def update(self, bbox):
            self.time_since_update = 0
            self.history = []
            self.hits += 1
            self.hit_streak += 1
            self.kf.update(self.convert_bbox_to_z(bbox))

        def predict(self):
            if self.kf.x[6] + self.kf.x[2] <= 0:
                self.kf.x[6] *= 0.0
            self.kf.predict()
            self.age += 1
            if self.time_since_update > 0:
                self.hit_streak = 0
            self.time_since_update += 1
            self.history.append(self.convert_x_to_bbox(self.kf.x))
            return self.history[-1]

        def get_state(self):
            return self.convert_x_to_bbox(self.kf.x)

    def iou(self, bb_test, bb_gt):
        xx1 = np.maximum(bb_test[0], bb_gt[0])
        yy1 = np.maximum(bb_test[1], bb_gt[1])
        xx2 = np.minimum(bb_test[2], bb_gt[2])
        yy2 = np.minimum(bb_test[3], bb_gt[3])
        w = np.maximum(0., xx2 - xx1)
        h = np.maximum(0., yy2 - yy1)
        wh = w * h
        o = wh / ((bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1])
                  + (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1]) - wh)
        return o

    def associate_detections_to_trackers(self, detections, trackers):
        if len(trackers) == 0:
            return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0,), dtype=int)
        iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)
        for d, det in enumerate(detections):
            for t, trk in enumerate(trackers):
                iou_matrix[d, t] = self.iou(det, trk)
        matched_indices = linear_sum_assignment(-iou_matrix)
        matched_indices = np.array(matched_indices).T

        unmatched_detections = []
        for d, det in enumerate(detections):
            if d not in matched_indices[:, 0]:
                unmatched_detections.append(d)
        unmatched_trackers = []
        for t, trk in enumerate(trackers):
            if t not in matched_indices[:, 1]:
                unmatched_trackers.append(t)

        matches = []
        for m in matched_indices:
            if iou_matrix[m[0], m[1]] < self.iou_threshold:
                unmatched_detections.append(m[0])
                unmatched_trackers.append(m[1])
            else:
                matches.append(m.reshape(1, 2))
        if len(matches) == 0:
            matches = np.empty((0, 2), dtype=int)
        else:
            matches = np.concatenate(matches, axis=0)
        return matches, np.array(unmatched_detections), np.array(unmatched_trackers)

    def update(self, dets):
        self.frame_count += 1
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []

        for t, trk in enumerate(self.trackers):
            pos = trk.predict().flatten()
            trks[t, :4] = pos
            trks[t, 4] = 0
            if np.any(np.isnan(pos)):
                to_del.append(t)
        for t in reversed(to_del):
            self.trackers.pop(t)

        matched, unmatched_dets, unmatched_trks = self.associate_detections_to_trackers(dets[:, :4], trks[:, :4])

        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :4])

        for i in unmatched_dets:
            trk = self.KalmanBoxTracker(dets[i,:4], self.next_id)
            self.next_id += 1
            self.trackers.append(trk)

        i = len(self.trackers)
        for trk in reversed(self.trackers):
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i-1)
            i -=1

        for trk in self.trackers:
            if (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits) and trk.time_since_update <= self.max_age:
                d = trk.get_state()
                ret.append(np.concatenate((d.flatten(), [trk.id])).reshape(1, -1))
        if len(ret)>0:
            return np.concatenate(ret)
        return np.empty((0,5))

def get_centroid(box):
    x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
    return ((x1 + x2) // 2, (y1 + y2) // 2)

def boxes_overlap(box1, box2):
    """Check if two bounding boxes overlap"""
    x1_min, y1_min, x1_max, y1_max = box1[0], box1[1], box1[2], box1[3]
    x2_min, y2_min, x2_max, y2_max = box2[0], box2[1], box2[2], box2[3]
    return not (x1_max < x2_min or x2_max < x1_min or y1_max < y2_min or y2_max < y1_min)

def sort_track_and_update_persons(person_boxes, vest_boxes, helmet_boxes, boots_boxes, gloves_boxes, person_history, sort_tracker):
    """SORT wrapper for your existing tracking interface"""
    if not person_boxes:
        return sort_tracker, person_history
    
    # Extract person detections for SORT [x1,y1,x2,y2,conf]
    person_dets = np.array([[box[0], box[1], box[2], box[3], box[4]] for box in person_boxes])
    
    # Update SORT tracker
    tracked_dets = sort_tracker.update(person_dets)
    
    # Match tracked persons to person_history
    ppe_counted_ids = set()
    for tracked_det in tracked_dets:
        track_id = int(tracked_det[4])
        bbox = tracked_det[:4]
        
        if track_id not in person_history:
            person_history[track_id] = {
                "frames": 0, "vest_frames": 0, "helmet_frames": 0,
                "boots_frames": 0, "gloves_frames": 0, "detected": False,
                "last_centroid": None, "last_area": 0
            }
        
        hist = person_history[track_id]
        hist["frames"] += 1
        
        if track_id not in ppe_counted_ids:
            if any(boxes_overlap(bbox, v_box) for v_box in vest_boxes):
                hist["vest_frames"] += 1
            if any(boxes_overlap(bbox, h_box) for h_box in helmet_boxes):
                hist["helmet_frames"] += 1
            if any(boxes_overlap(bbox, b_box) for b_box in boots_boxes):
                hist["boots_frames"] += 1
            if any(boxes_overlap(bbox, g_box) for g_box in gloves_boxes):
                hist["gloves_frames"] += 1
            ppe_counted_ids.add(track_id)
    
    return sort_tracker, person_history
