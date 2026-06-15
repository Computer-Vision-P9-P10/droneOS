import time, math
from datetime import datetime, timezone

class FlightPathSimulator:
    def __init__(self, start_lat, start_lon, start_alt=30, speed_mps=30):
        self.start_lat = start_lat
        self.start_lon = start_lon
        self.lat = start_lat
        self.lon = start_lon
        self.altitude = start_alt
        self.speed = speed_mps
        self.battery = 100
        self.last_update = time.time()

        # Fixed scan pattern: 100m forward, 20m left, 100m backward, 20m left.
        meters_to_lat = 1 / 111_000
        meters_to_lon = 1 / (111_000 * max(0.1, abs(math.cos(math.radians(start_lat)))))
        forward_100m = 100 * meters_to_lat
        left_20m = 20 * meters_to_lon

        # save commonly used offsets for later two-point path defaults
        self._forward_100m = forward_100m
        self._left_20m = left_20m

        # Define default "pattern" waypoints
        self._pattern_waypoints = [
            (start_lat + forward_100m, start_lon),
            (start_lat + forward_100m, start_lon - left_20m),
            (start_lat,                start_lon - left_20m),
            (start_lat,                start_lon - 2 * left_20m),
        ]

        # Start using the pattern by default
        self.waypoints = list(self._pattern_waypoints)
        self.waypoint_index = 0

        self.boundary = None

    def set_two_point_path(self, end_lat=None, end_lon=None):
        if end_lat is None or end_lon is None:
            end_lat = self.start_lat + self._forward_100m
            end_lon = self.start_lon
        self.waypoints = [(self.start_lat, self.start_lon), (float(end_lat), float(end_lon))]
        self.waypoint_index = 0
        # intentionally do not touch self.boundary

    def set_square_boundary(self, meters=2, end_lat=None, end_lon=None, set_waypoints=True):
        # Use the original start as one endpoint
        start_lat = float(self.start_lat)
        start_lon = float(self.start_lon)

        # Determine the end point to base the boundary on. Prefer provided end_lat/end_lon,
        # otherwise use the default forward offset from the original start.
        if end_lat is None or end_lon is None:
            end_lat = start_lat + self._forward_100m
            end_lon = start_lon
        else:
            end_lat = float(end_lat)
            end_lon = float(end_lon)

        # Compute min/max over the two endpoints and then pad by `meters`
        min_lat = min(start_lat, end_lat)
        max_lat = max(start_lat, end_lat)
        min_lon = min(start_lon, end_lon)
        max_lon = max(start_lon, end_lon)

        # Use center latitude for longitude degree conversion
        center_lat = (min_lat + max_lat) / 2.0
        meters_to_lat = 1 / 111_000
        meters_to_lon = 1 / (111_000 * max(0.1, abs(math.cos(math.radians(center_lat)))))

        pad_lat = meters * meters_to_lat
        pad_lon = meters * meters_to_lon

        min_lat -= pad_lat
        max_lat += pad_lat
        min_lon -= pad_lon
        max_lon += pad_lon

        # set a fixed boundary (independent of future waypoint changes)
        self.boundary = (min_lat, max_lat, min_lon, max_lon)

        if set_waypoints:
            # build square corners in consistent clockwise order (top-right first)
            top_right = (max_lat, max_lon)
            top_left = (max_lat, min_lon)
            bottom_left = (min_lat, min_lon)
            bottom_right = (min_lat, max_lon)

            # set the simulator's patrol waypoints to this square
            self.waypoints = [top_right, top_left, bottom_left, bottom_right]
            self.waypoint_index = 0
    def reset_to_pattern(self):
        """Restore the original multi-waypoint pattern."""
        self.waypoints = list(self._pattern_waypoints)
        self.waypoint_index = 0
        # Do NOT clear self.boundary here. Boundary is only updated by
        # set_square_boundary(...) when explicitly requested.

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
        if dist < 1e-12:
            # already exactly at waypoint, advance
            self.waypoint_index = (self.waypoint_index + 1) % len(self.waypoints)
        elif dist < step:
            self.lat, self.lon = target_lat, target_lon
            self.waypoint_index = (self.waypoint_index + 1) % len(self.waypoints)
        else:
            ratio = step / dist
            self.lat += dlat * ratio
            self.lon += dlon * ratio

        self.battery = max(0, self.battery - 0.01 * dt)

    def point_outside_boundary(self, lat, lon):
        """
        Return True if the given (lat, lon) is outside the configured square boundary.
        If no boundary is configured, return False.
        """
        if self.boundary is None:
            return False
        min_lat, max_lat, min_lon, max_lon = self.boundary
        return not (min_lat <= lat <= max_lat and min_lon <= lon <= max_lon)

    def get_telemetry(self):
        return {
            "lat": round(self.lat, 6),
            "lon": round(self.lon, 6),
            "altitude": self.altitude,
            "speed": self.speed,
            "battery": round(self.battery, 1),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
