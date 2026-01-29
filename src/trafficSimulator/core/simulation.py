from .vehicle_generator import VehicleGenerator
from .geometry.quadratic_curve import QuadraticCurve
from .geometry.cubic_curve import CubicCurve
from .geometry.segment import Segment
from .vehicle import Vehicle
from .traffic_signal import TrafficSignal

class Simulation:
    def __init__(self):
        self.segments = []
        self.vehicles = {}
        self.vehicle_generator = []
        self.traffic_signals = {}  # segment_index -> TrafficSignal

        self.t = 0.0
        self.frame_count = 0
        self.dt = 1/60  

    def add_vehicle(self, veh):
        self.vehicles[veh.id] = veh
        if len(veh.path) > 0:
            self.segments[veh.path[0]].add_vehicle(veh)

    def add_segment(self, seg):
        self.segments.append(seg)

    def add_vehicle_generator(self, gen):
        self.vehicle_generator.append(gen)

    def create_vehicle(self, **kwargs):
        veh = Vehicle(kwargs)
        self.add_vehicle(veh)

    def create_segment(self, *args):
        seg = Segment(args)
        self.add_segment(seg)

    def create_quadratic_bezier_curve(self, start, control, end):
        cur = QuadraticCurve(start, control, end)
        self.add_segment(cur)

    def create_cubic_bezier_curve(self, start, control_1, control_2, end):
        cur = CubicCurve(start, control_1, control_2, end)
        self.add_segment(cur)

    def create_vehicle_generator(self, **kwargs):
        gen = VehicleGenerator(kwargs)
        self.add_vehicle_generator(gen)

    def create_traffic_signal(self, segment_index, config=None):
        """Create a traffic signal on segment_index. Raises ValueError if index is out of bounds."""
        if segment_index < 0 or segment_index >= len(self.segments):
            raise ValueError(
                f"segment_index {segment_index} is out of range "
                f"(0..{len(self.segments) - 1})"
            )
        if config is None:
            config = {}
        config["segment_index"] = segment_index
        signal = TrafficSignal(config)
        self.traffic_signals[segment_index] = signal
        return signal

    def run(self, steps):
        for _ in range(steps):
            self.update()

    def update(self):
        # Advance traffic signals first so phantoms are in place before
        # vehicle updates read them.
        for signal in self.traffic_signals.values():
            signal.update(self, self.dt)

        # Update vehicles, injecting each signal's phantom as the leader
        # of the first vehicle on the guarded segment when active.
        for seg_idx, segment in enumerate(self.segments):
            if len(segment.vehicles) == 0:
                continue

            # Determine the effective leader for the front vehicle.
            # If a signal guards this segment and its phantom is active,
            # the phantom becomes the leader; otherwise None (no leader).
            signal = self.traffic_signals.get(seg_idx)
            front_leader = signal.phantom if signal else None

            self.vehicles[segment.vehicles[0]].update(front_leader, self.dt)
            for i in range(1, len(segment.vehicles)):
                self.vehicles[segment.vehicles[i]].update(
                    self.vehicles[segment.vehicles[i - 1]], self.dt
                )

        # Check roads for out of bounds vehicle
        for segment in self.segments:
            # If road has no vehicles, continue
            if len(segment.vehicles) == 0:
                continue
            # If not
            vehicle_id = segment.vehicles[0]
            vehicle = self.vehicles[vehicle_id]
            # If first vehicle is out of road bounds
            if vehicle.x >= segment.get_length():
                # If vehicle has a next road
                if vehicle.current_road_index + 1 < len(vehicle.path):
                    # Update current road to next road
                    vehicle.current_road_index += 1
                    # Add it to the next road
                    next_road_index = vehicle.path[vehicle.current_road_index]
                    self.segments[next_road_index].vehicles.append(vehicle_id)
                # Reset vehicle properties
                vehicle.x = 0
                # In all cases, remove it from its road
                segment.vehicles.popleft()

        # Update vehicle generators
        for gen in self.vehicle_generator:
            gen.update(self)
        # Increment time
        self.t += self.dt
        self.frame_count += 1