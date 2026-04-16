import config

# Presence threshold (how many of last N frames must show PPE)
PPE_PRESENCE_THRESHOLD = getattr(config, "PPE_PRESENCE_THRESHOLD", 0.70)
PPE_COMPLIANCE_MIN_FRAMES = getattr(config, "PPE_COMPLIANCE_MIN_FRAMES", 10)

# Per-item average-confidence thresholds used to distinguish compliant vs alert
PPE_CONF_VEST_THRESHOLD = getattr(config, "PPE_CONF_VEST_THRESHOLD", 0.30)
PPE_CONF_HELMET_THRESHOLD = getattr(config, "PPE_CONF_HELMET_THRESHOLD", 0.30)

PPE_ALERTS_ENABLED = getattr(config, "PPE_ALERTS_ENABLED", True)


def is_compliant(hist, presence_threshold=PPE_PRESENCE_THRESHOLD, min_frames=PPE_COMPLIANCE_MIN_FRAMES):
    recent = hist.get("recent")
    if recent is not None:
        frames = len(recent)
        if frames < min_frames:
            return "unknown"

        vest_frames = sum(1 for r in recent if r.get("vest"))
        helmet_frames = sum(1 for r in recent if r.get("helmet"))
        vest_presence = vest_frames / frames if frames > 0 else 0.0
        helmet_presence = helmet_frames / frames if frames > 0 else 0.0

        if vest_presence < presence_threshold or helmet_presence < presence_threshold:
            return "violation"

        avg_vest_conf = sum(r.get("vest_conf", 0.0) for r in recent) / frames
        avg_helmet_conf = sum(r.get("helmet_conf", 0.0) for r in recent) / frames

        if (avg_vest_conf >= PPE_CONF_VEST_THRESHOLD
            and avg_helmet_conf >= PPE_CONF_HELMET_THRESHOLD
        ):
            return "compliant"
        if PPE_ALERTS_ENABLED:
            return "alert"
        return "violation"
    else:
        frames = hist.get("frames", 0)
        if frames < min_frames:
            return "unknown"

        vest_ratio = hist.get("vest_frames", 0) / frames if frames > 0 else 0.0
        helmet_ratio = hist.get("helmet_frames", 0) / frames if frames > 0 else 0.0

        if vest_ratio >= presence_threshold and helmet_ratio >= presence_threshold:
            return "compliant"

        return "violation"


def get_current_state(hist):
    return hist.get("state", "unknown")


def cleanup_person_history(person_history, frame_count, min_frames, stale_frames):
    to_delete = []
    for pid, hist in person_history.items():
        recent = hist.get("recent")
        frames = len(recent) if recent is not None else hist.get("frames", 0)
        last_seen = hist.get("last_seen_frame", 0)
        is_stale = (frame_count - last_seen) > stale_frames
        is_short = frames < min_frames and is_stale
        if is_short:
            to_delete.append(pid)

    for pid in to_delete:
        del person_history[pid]
