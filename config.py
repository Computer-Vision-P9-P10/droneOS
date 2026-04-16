# MODEL_PATH = "yolo12s.pt"
MODEL_PATH = "PPE-Drone-Models/3x/yolo26s/best.pt"
# VIDEO_PATH = 0
VIDEO_PATH = "./videos/new_videos/DJI_0022.mp4" # Compliance example
# VIDEO_PATH = "./videos/DJI_0017.MP4" # Compliance example
# VIDEO_PATH = "./videos/DJI_0006.MP4" # Violation example (only in beginning of video)
# VIDEO_PATH = "./videos/DJI_0008.MP4" # Violation example (Full video)
# VIDEO_PATH = "./videos/DJI_0019.MP4" # Hardware zoom example
# VIDEO_PATH = "./videos/DJI_0020.MP4" # Dark hardware zoom example
# VIDEO_PATH = "./videos/amar/DJI_0018.MP4"
BACKEND_HOST = "0.0.0.0"

FILTERS_ON = False
CONSOLE_OUTPUT = False 
SHOW_LIVE_FEED = True
DEVICE = "cpu"  # "cuda" or "cpu"

# Detection config
CONFIDENCE = 0.1
IOU = 0.6
FRAME_INTERVAL = 1

PERSON_CONF = 0.6
VEST_CONF = 0.1
HELMET_CONF = 0.1

# Events config
STATE_CHANGE_MIN_SECONDS = 2.0

# Tracker config
PPE_ALERTS_ENABLED = True
TRACKER_YAML = "custom_bytetrack.yaml"
PERSON_HISTORY_MAX_FRAMES = 50
MIN_TRACK_FRAMES = 20
STALE_TRACK_FRAMES = 120

PPE_PRESENCE_THRESHOLD = 0.70
PPE_COMPLIANCE_MIN_FRAMES = 20
PPE_CONF_VEST_THRESHOLD = 0.30
PPE_CONF_HELMET_THRESHOLD = 0.30

# Zoom config
ZOOM_ENABLED = False
ZOOM_FACTOR = 2
ZOOM_MIN_DURATION = 2
ZOOM_PERSON_FRAME_THRESHOLD = 10
ZOOM_SIZE_MIN_THRESHOLD = 0.4
ZOOM_SIZE_MAX_THRESHOLD = 0.6
MAX_ZOOM_FACTOR = 4.0
ZOOM_STEP = 0.5
