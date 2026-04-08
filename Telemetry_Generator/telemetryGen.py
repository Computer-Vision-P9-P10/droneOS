import time, math
from datetime import datetime, timezone

class FlightPathSimulator:
    def __init__(self, start_lat, start_lon, start_alt=30, speed_mps=5):
        self.lat = start_lat
        self.lon = start_lon
        self.altitude = start_alt
        self.speed = speed_mps
        self.battery = 100
        self.last_update = time.time()

        # We define waypoints for the simulation. Potentially explore using video file data.
        offset = 0.001  # aprox 100m in degrees. Lat long version.
        self.waypoints = [
            (start_lat + offset, start_lon),
            (start_lat + offset, start_lon + offset),
            (start_lat,          start_lon + offset),
            (start_lat,          start_lon),
        ]
        self.waypoint_index = 0

    def update(self):
        now = time.time()
        dt = now - self.last_update
        self.last_update = now

        # Move toward current waypoint
        target_lat, target_lon = self.waypoints[self.waypoint_index]
        dlat = target_lat - self.lat
        dlon = target_lon - self.lon
        dist = math.sqrt(dlat**2 + dlon**2)

        step = (self.speed / 111_000) * dt  # convert m/s to degrees
        if dist < step:
            self.lat, self.lon = target_lat, target_lon
            self.waypoint_index = (self.waypoint_index + 1) % len(self.waypoints)
        else:
            ratio = step / dist
            self.lat += dlat * ratio
            self.lon += dlon * ratio

        self.battery = max(0, self.battery - 0.01 * dt)

    def get_telemetry(self):
        return {
            "lat": round(self.lat, 6),
            "lon": round(self.lon, 6),
            "altitude": self.altitude,
            "speed": self.speed,
            "battery": round(self.battery, 1),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }