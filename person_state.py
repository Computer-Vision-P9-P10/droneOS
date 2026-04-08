import config

PPE_COMPLIANCE_THRESHOLD = getattr(config, "PPE_COMPLIANCE_THRESHOLD", 0.70)
PPE_COMPLIANCE_MIN_FRAMES = getattr(config, "PPE_COMPLIANCE_MIN_FRAMES", 10)

def is_compliant(hist, threshold=PPE_COMPLIANCE_THRESHOLD, min_frames=PPE_COMPLIANCE_MIN_FRAMES):
    frames = hist.get("frames", 0)
    if frames < min_frames:
        return "unknown"

    vest_ratio = hist.get("vest_frames", 0) / frames
    helmet_ratio = hist.get("helmet_frames", 0) / frames

    if vest_ratio >= threshold and helmet_ratio >= threshold:
        return "compliant"

    return "violation"

def get_current_state(hist):
    return hist.get("state", "unknown")

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
