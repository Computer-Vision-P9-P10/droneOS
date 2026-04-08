import asyncio
import httpx
import cv2
import config

backend_host = config.BACKEND_HOST
STATE_CHANGE_MIN_SECONDS = getattr(config, "STATE_CHANGE_MIN_SECONDS", 10.0)


async def violation_detected(violation: str):
    message = f"Missing {violation} detected!"
    url = f"{backend_host}/violation"
    data = {"message": message, "timestamp": ""}

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(url, json=data)
            if response.status_code == 200:
                print("API request successful")
            else:
                print(f"API returned status code {response.status_code}")
        except Exception as e:
            print(f"Failed to send API request: {e}")


async def on_person_detected(pid: int, hist: dict, state: str):
    print(f"Person {pid} state changed to '{state}'. Executing action...")


def trigger_event(
    person_history,
    cap,
    event_count,
):
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps > 0 else 0.0

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

        if fps > 0:
            stable_time_seconds = max(
                0.0, (current_frame - state_changed_at_frame) / fps
            )
        else:
            stable_time_seconds = 0.0

        if stable_time_seconds < STATE_CHANGE_MIN_SECONDS:
            continue

        event_count += 1
        asyncio.run(on_person_detected(pid, hist, current_state))
        hist["last_sent_state"] = current_state

    return event_count
