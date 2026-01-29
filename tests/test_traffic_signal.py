import unittest
from collections import deque
from trafficSimulator.core.traffic_signal import TrafficSignal, SignalState
from trafficSimulator.core.vehicle import Vehicle
from trafficSimulator.core.geometry.segment import Segment
from trafficSimulator.core.simulation import Simulation


class TestSignalState(unittest.TestCase):
    def test_enum_values_exist(self):
        self.assertIsNotNone(SignalState.GREEN)
        self.assertIsNotNone(SignalState.YELLOW)
        self.assertIsNotNone(SignalState.RED)

    def test_enum_values_are_distinct(self):
        states = [SignalState.GREEN, SignalState.YELLOW, SignalState.RED]
        self.assertEqual(len(states), len(set(states)))

    def test_enum_string_values(self):
        self.assertEqual(SignalState.GREEN.value, "green")
        self.assertEqual(SignalState.YELLOW.value, "yellow")
        self.assertEqual(SignalState.RED.value, "red")


class TestTrafficSignalDefaults(unittest.TestCase):
    def setUp(self):
        self.signal = TrafficSignal({"segment_index": 0})

    def test_segment_index_stored(self):
        self.assertEqual(self.signal.segment_index, 0)

    def test_default_green_duration(self):
        self.assertEqual(self.signal.green_duration, 10.0)

    def test_default_yellow_duration(self):
        self.assertEqual(self.signal.yellow_duration, 3.0)

    def test_default_red_duration(self):
        self.assertEqual(self.signal.red_duration, 10.0)

    def test_default_stop_position_is_none(self):
        self.assertIsNone(self.signal.stop_position)

    def test_initial_state_is_green(self):
        self.assertEqual(self.signal.state, SignalState.GREEN)

    def test_initial_phantom_is_none(self):
        self.assertIsNone(self.signal.phantom)

    def test_cycle_duration(self):
        expected = 10.0 + 3.0 + 10.0  # 23.0
        self.assertEqual(self.signal.cycle_duration, expected)


class TestTrafficSignalCycling(unittest.TestCase):
    def setUp(self):
        self.signal = TrafficSignal(
            {
                "segment_index": 0,
                "green_duration": 10.0,
                "yellow_duration": 3.0,
                "red_duration": 10.0,
                "stop_position": 100.0,
            }
        )
        self.sim = Simulation()
        self.sim.segments.append(Segment(((0, 0), (100, 0))))

    def test_starts_green(self):
        self.assertEqual(self.signal.state, SignalState.GREEN)

    def test_transitions_to_yellow(self):
        # 10 seconds at 60fps = 600 frames
        for _ in range(600):
            self.signal.update(self.sim, 1 / 60)
        self.assertEqual(self.signal.state, SignalState.YELLOW)

    def test_transitions_to_red(self):
        # 13 seconds (green + yellow) at 60fps = 780 frames
        for _ in range(780):
            self.signal.update(self.sim, 1 / 60)
        self.assertEqual(self.signal.state, SignalState.RED)

    def test_wraps_to_green(self):
        # 23 seconds (full cycle) at 60fps = 1380 frames + a bit more
        for _ in range(1400):
            self.signal.update(self.sim, 1 / 60)
        self.assertEqual(self.signal.state, SignalState.GREEN)


class TestPhantomLifecycle(unittest.TestCase):
    def setUp(self):
        self.signal = TrafficSignal(
            {
                "segment_index": 0,
                "green_duration": 10.0,
                "yellow_duration": 3.0,
                "red_duration": 10.0,
                "stop_position": 100.0,
            }
        )
        self.sim = Simulation()
        self.sim.segments.append(Segment(((0, 0), (100, 0))))

    def test_no_phantom_during_green(self):
        self.signal.update(self.sim, 1 / 60)
        self.assertIsNone(self.signal.phantom)

    def test_phantom_exists_during_red(self):
        for _ in range(800):
            self.signal.update(self.sim, 1 / 60)
        self.assertEqual(self.signal.state, SignalState.RED)
        self.assertIsNotNone(self.signal.phantom)

    def test_phantom_velocity_is_zero(self):
        for _ in range(800):
            self.signal.update(self.sim, 1 / 60)
        self.assertEqual(self.signal.phantom.v, 0)

    def test_phantom_position_at_stop_line(self):
        for _ in range(800):
            self.signal.update(self.sim, 1 / 60)
        self.assertEqual(self.signal.phantom.x, 100.0)

    def test_stop_position_resolved_from_segment(self):
        signal = TrafficSignal({"segment_index": 0})
        sim = Simulation()
        sim.segments.append(Segment(((0, 0), (50, 0))))
        for _ in range(800):
            signal.update(sim, 1 / 60)
        self.assertEqual(signal.stop_position, 50.0)


class TestVehicleStopsAtRed(unittest.TestCase):
    def setUp(self):
        self.sim = Simulation()
        self.sim.segments.append(Segment(((0, 0), (200, 0))))
        self.signal = TrafficSignal(
            {
                "segment_index": 0,
                "green_duration": 0.1,
                "yellow_duration": 0.1,
                "red_duration": 30.0,
                "stop_position": 100.0,
            }
        )
        self.sim.traffic_signals[0] = self.signal
        self.vehicle = Vehicle({"x": 10, "v": 15, "path": [0]})
        self.sim.add_vehicle(self.vehicle)

    def test_vehicle_stops_before_stop_line(self):
        for _ in range(600):
            self.sim.update()
        self.assertLess(self.vehicle.x, 100.0)

    def test_vehicle_velocity_reaches_zero(self):
        for _ in range(600):
            self.sim.update()
        self.assertAlmostEqual(self.vehicle.v, 0, delta=0.1)


class TestVehicleResumesOnGreen(unittest.TestCase):
    def test_vehicle_accelerates_after_green(self):
        sim = Simulation()
        sim.segments.append(Segment(((0, 0), (200, 0))))
        # Short red (5s), then back to green
        signal = TrafficSignal(
            {
                "segment_index": 0,
                "green_duration": 5.0,
                "yellow_duration": 0.1,
                "red_duration": 5.0,
                "stop_position": 100.0,
            }
        )
        sim.traffic_signals[0] = signal
        vehicle = Vehicle({"x": 90, "v": 0, "path": [0]})
        sim.add_vehicle(vehicle)

        # Advance past green (5s) and yellow (0.1s) into red
        for _ in range(320):  # ~5.3 seconds
            sim.update()
        self.assertEqual(signal.state, SignalState.RED)

        # Advance through rest of red (~4.7s) back to green
        for _ in range(300):  # ~5 more seconds
            sim.update()
        self.assertEqual(signal.state, SignalState.GREEN)

        # Let vehicle accelerate
        for _ in range(120):
            sim.update()
        self.assertGreater(vehicle.v, 1.0)


class TestYellowPhaseLogic(unittest.TestCase):
    def setUp(self):
        self.sim = Simulation()
        self.sim.segments.append(Segment(((0, 0), (200, 0))))
        self.signal = TrafficSignal(
            {
                "segment_index": 0,
                "green_duration": 2.0,
                "yellow_duration": 3.0,
                "red_duration": 10.0,
                "stop_position": 100.0,
            }
        )
        self.sim.traffic_signals[0] = self.signal

    def test_vehicle_past_stop_line_not_blocked(self):
        vehicle = Vehicle({"x": 110, "v": 10, "path": [0]})
        self.sim.add_vehicle(vehicle)
        for _ in range(150):
            self.sim.update()
        self.assertEqual(self.signal.state, SignalState.YELLOW)
        self.assertIsNone(self.signal.phantom)

    def test_fast_vehicle_blocked_during_yellow(self):
        vehicle = Vehicle({"x": 50, "v": 15, "path": [0]})
        self.sim.add_vehicle(vehicle)
        for _ in range(150):
            self.sim.update()
        self.assertEqual(self.signal.state, SignalState.YELLOW)
        self.assertIsNotNone(self.signal.phantom)

    def test_slow_vehicle_not_blocked_during_yellow(self):
        vehicle = Vehicle({"x": 50, "v": 2, "path": [0]})
        self.sim.add_vehicle(vehicle)
        for _ in range(150):
            self.sim.update()
        self.assertEqual(self.signal.state, SignalState.YELLOW)
        self.assertIsNone(self.signal.phantom)


class TestMultipleVehicleQueue(unittest.TestCase):
    def setUp(self):
        self.sim = Simulation()
        self.sim.segments.append(Segment(((0, 0), (300, 0))))
        self.signal = TrafficSignal(
            {
                "segment_index": 0,
                "green_duration": 0.1,
                "yellow_duration": 0.1,
                "red_duration": 10.0,
                "stop_position": 200.0,
            }
        )
        self.sim.traffic_signals[0] = self.signal
        self.vehicles = [
            Vehicle({"x": 50, "v": 15, "path": [0]}),
            Vehicle({"x": 30, "v": 15, "path": [0]}),
            Vehicle({"x": 10, "v": 15, "path": [0]}),
        ]
        for v in self.vehicles:
            self.sim.add_vehicle(v)

    def test_all_vehicles_stop_before_line(self):
        for _ in range(900):
            self.sim.update()
        for v in self.vehicles:
            self.assertLess(v.x, 200.0)

    def test_all_vehicles_resume_after_green(self):
        # Run until we're in GREEN phase (robust timing)
        for _ in range(2000):
            self.sim.update()
            if self.signal.state == SignalState.GREEN:
                break
        self.assertEqual(self.signal.state, SignalState.GREEN)
        # Let vehicles accelerate
        for _ in range(300):
            self.sim.update()
        for v in self.vehicles:
            self.assertGreater(v.v, 0.5)


class TestSignalWithVehicleGenerator(unittest.TestCase):
    def test_generated_vehicles_stop_at_signal(self):
        sim = Simulation()
        sim.segments.append(Segment(((0, 0), (300, 0))))
        signal = TrafficSignal(
            {
                "segment_index": 0,
                "green_duration": 0.1,
                "yellow_duration": 0.1,
                "red_duration": 60.0,
                "stop_position": 200.0,
            }
        )
        sim.traffic_signals[0] = signal
        # Pass path inside vehicle config, not as generator attribute
        sim.create_vehicle_generator(vehicle_rate=60, vehicles=[(1, {"path": [0]})])
        for _ in range(1800):
            sim.update()
        # Check all vehicles on segment stopped before line
        for vid in sim.segments[0].vehicles:
            self.assertLess(sim.vehicles[vid].x, 200.0)


class TestCreateTrafficSignalHelper(unittest.TestCase):
    def setUp(self):
        self.sim = Simulation()
        self.sim.segments.append(Segment(((0, 0), (100, 0))))
        self.sim.segments.append(Segment(((100, 0), (200, 0))))

    def test_returns_traffic_signal_instance(self):
        signal = self.sim.create_traffic_signal(0)
        self.assertIsInstance(signal, TrafficSignal)

    def test_signal_registered_in_dict(self):
        self.sim.create_traffic_signal(0)
        self.assertIn(0, self.sim.traffic_signals)

    def test_segment_index_set_correctly(self):
        signal = self.sim.create_traffic_signal(1)
        self.assertEqual(signal.segment_index, 1)

    def test_config_applied(self):
        signal = self.sim.create_traffic_signal(0, {"green_duration": 20.0})
        self.assertEqual(signal.green_duration, 20.0)

    def test_replaces_existing_signal(self):
        self.sim.create_traffic_signal(0, {"green_duration": 10.0})
        signal2 = self.sim.create_traffic_signal(0, {"green_duration": 25.0})
        self.assertEqual(self.sim.traffic_signals[0].green_duration, 25.0)
        self.assertIs(self.sim.traffic_signals[0], signal2)

    def test_default_config_when_none(self):
        signal = self.sim.create_traffic_signal(0)
        self.assertEqual(signal.green_duration, 10.0)  # Your default

    def test_negative_index_raises_value_error(self):
        with self.assertRaises(ValueError):
            self.sim.create_traffic_signal(-1)

    def test_index_equal_to_length_raises_value_error(self):
        with self.assertRaises(ValueError):
            self.sim.create_traffic_signal(2)

    def test_index_greater_than_length_raises_value_error(self):
        with self.assertRaises(ValueError):
            self.sim.create_traffic_signal(10)

    def test_empty_segments_raises_value_error(self):
        empty_sim = Simulation()
        with self.assertRaises(ValueError):
            empty_sim.create_traffic_signal(0)


class TestCustomStopPosition(unittest.TestCase):
    def test_vehicle_stops_at_custom_position(self):
        sim = Simulation()
        sim.segments.append(Segment(((0, 0), (200, 0))))
        signal = TrafficSignal(
            {
                "segment_index": 0,
                "green_duration": 0.1,
                "yellow_duration": 0.1,
                "red_duration": 30.0,
                "stop_position": 50.0,
            }
        )
        sim.traffic_signals[0] = signal
        vehicle = Vehicle({"x": 10, "v": 10, "path": [0]})
        sim.add_vehicle(vehicle)
        for _ in range(600):
            sim.update()
        self.assertLess(vehicle.x, 50.0)

    def test_phantom_at_custom_position(self):
        sim = Simulation()
        sim.segments.append(Segment(((0, 0), (200, 0))))
        signal = TrafficSignal(
            {
                "segment_index": 0,
                "green_duration": 0.1,
                "yellow_duration": 0.1,
                "red_duration": 30.0,
                "stop_position": 75.0,
            }
        )
        sim.traffic_signals[0] = signal
        for _ in range(60):
            signal.update(sim, 1 / 60)
        self.assertEqual(signal.phantom.x, 75.0)


class TestCycleWrapAround(unittest.TestCase):
    def test_state_after_multiple_cycles(self):
        signal = TrafficSignal(
            {
                "segment_index": 0,
                "green_duration": 10.0,
                "yellow_duration": 3.0,
                "red_duration": 10.0,
                "stop_position": 100.0,
            }
        )
        sim = Simulation()
        sim.segments.append(Segment(((0, 0), (100, 0))))
        cycle = 23.0
        # 3 full cycles + 5 seconds into green
        steps = int(3 * cycle * 60) + 300
        for _ in range(steps):
            signal.update(sim, 1 / 60)
        self.assertEqual(signal.state, SignalState.GREEN)

    def test_state_mid_red_after_cycles(self):
        signal = TrafficSignal(
            {
                "segment_index": 0,
                "green_duration": 10.0,
                "yellow_duration": 3.0,
                "red_duration": 10.0,
                "stop_position": 100.0,
            }
        )
        sim = Simulation()
        sim.segments.append(Segment(((0, 0), (100, 0))))
        cycle = 23.0
        # 2 full cycles + 13s (green) + 3s (yellow) + 5s (mid-red) = 2*23 + 18 = 64s
        mid_red_time = 2 * cycle + 10.0 + 3.0 + 5.0
        steps = int(mid_red_time * 60)
        for _ in range(steps):
            signal.update(sim, 1 / 60)
        self.assertEqual(signal.state, SignalState.RED)


if __name__ == "__main__":
    unittest.main()
