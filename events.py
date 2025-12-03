import asyncio
import httpx
import cv2

async def violation_detected(backend_host: str, violation: str):
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

async def on_person_detected():
    print("Person detected! Executing action...")

def trigger_on_person_detected(person_history, cap, on_person_detected_count, backend_host):
    for pid, hist in person_history.items():
        if cap.get(cv2.CAP_PROP_FPS) > 0:
            time_in_frame = hist["frames"] / cap.get(cv2.CAP_PROP_FPS)
        else:
            time_in_frame = 0
        overlap_frames = hist["vest_frames"] + hist["helmet_frames"]
        if (
            hist["frames"] > 0
            and overlap_frames / hist["frames"] > 0.6
            and time_in_frame >= 10
            and not hist.get("detected", False)
        ):
            on_person_detected_count += 1
            asyncio.run(on_person_detected())
            hist["frames"] = 0
            hist["vest_frames"] = 0
            hist["helmet_frames"] = 0
            hist["boots_frames"] = 0
            hist["gloves_frames"] = 0
            hist["detected"] = True
            break
    return on_person_detected_count
