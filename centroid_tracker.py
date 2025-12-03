import numpy as np

def get_centroid(box):
    x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
    return ((x1 + x2) // 2, (y1 + y2) // 2)

def boxes_overlap(box1, box2):
    x1_min, y1_min, x1_max, y1_max = box1[0], box1[1], box1[2], box1[3]
    x2_min, y2_min, x2_max, y2_max = box2[0], box2[1], box2[2], box2[3]
    return not (x1_max < x2_min or x2_max < x1_min or y1_max < y2_min or y2_max < y1_min)

def track_and_update_persons(person_boxes, vest_boxes, helmet_boxes, boots_boxes, gloves_boxes, person_history, next_person_id):
    if not person_boxes:
        return next_person_id

    current_centroids = [(p_box, get_centroid(p_box)) for p_box in person_boxes]

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
