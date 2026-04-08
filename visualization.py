import cv2

def compliance_color_from_state(state):
    if state == "unknown":
        return (0, 255, 255)
    if state == "compliant":
        return (0, 200, 0)
    return (0, 0, 255)


def draw_top_left_overlay(frame, current_frame_people, person_history):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.58
    thickness = 2
    line_height = 24
    padding = 10
    margin = 12

    current_frame_track_ids = sorted(
        int(person["track_id"])
        for person in current_frame_people
        if person["track_id"] is not None
    )

    lines = []
    lines.append(
        "Active IDs: "
        + (", ".join(map(str, current_frame_track_ids)) if current_frame_track_ids else "none")
    )

    for person in current_frame_people:
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

        compliance_state = hist.get("state", "unknown")
        if compliance_state == "compliant":
            status = "OK"
        elif compliance_state == "violation":
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
