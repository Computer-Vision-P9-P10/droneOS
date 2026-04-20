import cv2
import config

STATE_CHANGE_MIN_SECONDS = getattr(config, "STATE_CHANGE_MIN_SECONDS", 10.0)


def collect_state_change_events(
    person_history,
    cap,
    event_count,
):
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps > 0 else 0.0
    emitted_events = []

    for pid, hist in person_history.items():
        current_state = hist.get("state", "unknown")
        if current_state is None:
            continue

        last_sent_state = hist.get("last_sent_state", None)
        if last_sent_state == current_state:
            continue

        state_changed_at_frame = hist.get(
            "state_changed_at_frame",
            hist.get("last_seen_frame", 0),
        )
        current_frame = hist.get("last_seen_frame", 0)

        recent = hist.get("recent")
        if recent is not None:
            frames = len(recent)
            vest_frames = sum(1 for r in recent if r.get("vest"))
            helmet_frames = sum(1 for r in recent if r.get("helmet"))
            avg_vest_conf = sum(r.get("vest_conf", 0.0) for r in recent) / frames if frames > 0 else 0.0
            avg_helmet_conf = sum(r.get("helmet_conf", 0.0) for r in recent) / frames if frames > 0 else 0.0
            avg_person_conf = sum(r.get("person_conf", 0.0) for r in recent) / frames if frames > 0 else 0.0
        else:
            frames = int(hist.get("frames", 0))
            vest_frames = int(hist.get("vest_frames", 0))
            helmet_frames = int(hist.get("helmet_frames", 0))
            avg_person_conf = float(hist.get("last_person_conf", 0.0))
            avg_vest_conf = float(hist.get("last_vest_conf", 0.0))
            avg_helmet_conf = float(hist.get("last_helmet_conf", 0.0))

        if fps > 0:
            stable_time_seconds = max(
                0.0, (current_frame - state_changed_at_frame) / fps
            )
        else:
            stable_time_seconds = 0.0

        if stable_time_seconds < STATE_CHANGE_MIN_SECONDS:
            continue

        event_count += 1
        emitted_events.append(
            {
                "person_id": int(pid),
                "state": current_state,
                "stable_time_seconds": stable_time_seconds,
                "frames": int(frames),
                "vest_frames": int(vest_frames),
                "helmet_frames": int(helmet_frames),
                "person_confidence": float(avg_person_conf),
                "vest_confidence": float(avg_vest_conf),
                "helmet_confidence": float(avg_helmet_conf),
                "last_seen_frame": int(hist.get("last_seen_frame", 0)),
                "last_box": hist.get("last_box"),
            }
        )
        hist["last_sent_state"] = current_state

    return event_count, emitted_events
