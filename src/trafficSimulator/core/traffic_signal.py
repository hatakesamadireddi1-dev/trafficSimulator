from enum import Enum
from .vehicle import Vehicle


class SignalState(Enum):
    """GREEN, YELLOW, RED phases of a traffic signal."""

    GREEN = "green"
    YELLOW = "yellow"
    RED = "red"


class TrafficSignal:
    """Controls vehicle flow on one segment via a GREEN/YELLOW/RED cycle.

    Uses a phantom vehicle at the stop line to leverage the existing IDM
    braking logic.  During YELLOW a hysteresis flag prevents the phantom
    from chattering on/off while vehicles oscillate around the braking-
    distance boundary.
    """

    def __init__(self, config=None):
        # Set default configuration
        self._set_default_config()

        # Update configuration
        if config:
            for attr, val in config.items():
                setattr(self, attr, val)

        # Calculate properties
        self._init_properties()

    def _set_default_config(self):
        self.segment_index = 0

        self.green_duration = 10.0  # seconds
        self.yellow_duration = 3.0  # seconds
        self.red_duration = 10.0  # seconds

        # None means "end of segment"; resolved on first update
        self.stop_position = None

    def _init_properties(self):
        self._cycle_time = 0.0
        self._state = SignalState.GREEN
        self.phantom = None
        self._position_resolved = False
        # Keeps the phantom locked during YELLOW until vehicles actually stop
        self._yellow_hold = False

    @property
    def state(self):
        """Current signal phase"""
        return self._state

    @property
    def cycle_duration(self):
        """Total duration of one GREEN→YELLOW→RED cycle"""
        return self.green_duration + self.yellow_duration + self.red_duration

    def update(self, simulation, dt):
        """Advance the signal clock and update the phantom"""
        # Resolve stop_position against the real segment length once
        if not self._position_resolved:
            seg_length = simulation.segments[self.segment_index].get_length()
            if self.stop_position is None:
                self.stop_position = seg_length
            self._position_resolved = True

        # Advance cycle clock, wrapping at the end of a full cycle
        self._cycle_time += dt
        if self._cycle_time >= self.cycle_duration:
            self._cycle_time -= self.cycle_duration

        # Determine which phase we are in
        if self._cycle_time < self.green_duration:
            self._state = SignalState.GREEN
        elif self._cycle_time < self.green_duration + self.yellow_duration:
            self._state = SignalState.YELLOW
        else:
            self._state = SignalState.RED

        # Update the phantom based on state and vehicle positions
        self._update_phantom(simulation)

    def _update_phantom(self, simulation):
        """Create, keep, or remove the phantom based on the current phase"""
        if self._state == SignalState.GREEN:
            self.phantom = None
            self._yellow_hold = False

        elif self._state == SignalState.RED:
            self.phantom = self._make_phantom()
            self._yellow_hold = False

        else:
            # YELLOW: hold phantom once triggered until vehicles stop
            if self._yellow_hold:
                if self._all_blocked_vehicles_stopped(simulation):
                    self._yellow_hold = False
                    self.phantom = None
                else:
                    self.phantom = self._make_phantom()
            else:
                if self._any_vehicle_must_stop(simulation):
                    self._yellow_hold = True
                    self.phantom = self._make_phantom()
                else:
                    self.phantom = None

    def _make_phantom(self):
        """Return a zero-length, stationary phantom at the stop line"""
        return Vehicle(
            {
                "x": self.stop_position,
                "v": 0,
                "a": 0,
                "l": 0,
                "s0": 0,
                "path": [],
                "v_max": 0,
                "a_max": 0,
                "b_max": 0,
            }
        )

    def _any_vehicle_must_stop(self, simulation):
        """Check whether any vehicle cannot brake to a stop before the line.
        Uses braking distance estimate d = v² / (2 · b_max)."""
        segment = simulation.segments[self.segment_index]
        for veh_id in segment.vehicles:
            veh = simulation.vehicles[veh_id]
            if veh.x >= self.stop_position:
                continue
            distance_to_line = self.stop_position - veh.x
            if veh.b_max > 0:
                braking_distance = (veh.v**2) / (2.0 * veh.b_max)
            else:
                braking_distance = 0.0
            if braking_distance > distance_to_line:
                return True
        return False

    def _all_blocked_vehicles_stopped(self, simulation):
        """Return True when every vehicle before the stop line has v ≈ 0"""
        _STOPPED_THRESHOLD = 0.05  # m/s
        segment = simulation.segments[self.segment_index]
        for veh_id in segment.vehicles:
            veh = simulation.vehicles[veh_id]
            if veh.x >= self.stop_position:
                continue
            if veh.v > _STOPPED_THRESHOLD:
                return False
        return True
