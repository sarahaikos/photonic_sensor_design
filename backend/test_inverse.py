"""DAG system graphs, parameter scaling, and the dual-level tuner."""

from __future__ import annotations

import unittest

import numpy as np

from inverse.graph import GraphError, PhotonicGraph, from_devices, sensor_graph
from inverse.parameters import ParameterSpec, bind, system_specs
from inverse.pipeline import Stack, SystemObjective, forward, inner_tune, run_inverse


class DagTest(unittest.TestCase):
    def test_all_pass_order_and_devices(self):
        graph = sensor_graph("all-pass")
        self.assertEqual(graph.topological_order(), ["src", "bus", "cpl", "ring", "det"])
        kinds = [device["type"] for device in graph.to_devices()]
        self.assertEqual(kinds, ["waveguide", "coupler", "ring", "detector"])
        self.assertEqual(graph.bus_coupler_id(), "cpl")
        self.assertNotIn("src", kinds)

    def test_add_drop_is_acyclic_and_has_two_couplers(self):
        graph = sensor_graph("add-drop", heater=True)
        order = graph.topological_order()
        self.assertEqual(order[0], "src")
        self.assertIn("cpl_drop", order)
        self.assertLess(order.index("ring"), order.index("cpl_drop"))
        self.assertLess(order.index("heater"), order.index("ring"))
        couplers = [device for device in graph.to_devices() if device["type"] == "coupler"]
        self.assertEqual(len(couplers), 2)
        self.assertEqual(graph.nodes["ring"].params["config"], "add-drop")

    def test_cycle_is_rejected_and_not_kept(self):
        graph = PhotonicGraph()
        graph.add("a", "waveguide", {"length_um": 10.0})
        graph.add("b", "waveguide", {"length_um": 10.0})
        graph.connect("a", "out", "b", "in")
        with self.assertRaises(GraphError):
            graph.connect("b", "out", "a", "in")
        self.assertEqual(len(graph.edges), 1)
        self.assertEqual(graph.topological_order(), ["a", "b"])

    def test_port_rules(self):
        graph = sensor_graph("all-pass")
        with self.assertRaises(GraphError):
            graph.connect("cpl", "through", "det", "in")
        with self.assertRaises(GraphError):
            graph.connect("bus", "out", "ring", "heat")
        with self.assertRaises(GraphError):
            graph.connect("det", "in", "bus", "in")

    def test_from_devices_round_trip_kinds(self):
        devices = [
            {"type": "waveguide", "length_um": 80.0},
            {"type": "coupler", "gap_nm": 180.0, "length_um": 10.0},
            {"type": "ring", "radius_um": 8.0, "config": "all-pass"},
            {
                "type": "detector",
                "responsivity_a_per_w": 0.8,
                "dark_current_na": 1.0,
                "bandwidth_mhz": 50.0,
                "load_ohm": 50.0,
            },
        ]
        graph = from_devices(devices)
        rebuilt = graph.to_devices()
        self.assertEqual([device["type"] for device in rebuilt], ["waveguide", "coupler", "ring", "detector"])
        self.assertEqual(rebuilt[1]["gap_nm"], 180.0)
        self.assertEqual(rebuilt[2]["radius_um"], 8.0)
        self.assertEqual(graph.bus_coupler_id(), "cpl")


class ParameterTest(unittest.TestCase):
    def test_linear_and_log_round_trip_at_bounds(self):
        linear = ParameterSpec("gap_nm", "device", 200.0, 50.0, 500.0, node_id="cpl", key="gap_nm")
        self.assertEqual(linear.to_scaled(50.0), 0.0)
        self.assertEqual(linear.to_scaled(500.0), 1.0)
        self.assertAlmostEqual(linear.from_scaled(0.0), 50.0)
        self.assertAlmostEqual(linear.from_scaled(1.0), 500.0)
        self.assertAlmostEqual(linear.from_scaled(linear.to_scaled(200.0)), 200.0)
        # Out-of-range values sit on the bound in scaled space.
        self.assertEqual(linear.to_scaled(10.0), 0.0)

        log_spec = ParameterSpec("q_target", "system", 10.0, 1.0, 100.0, scale="log", frozen=True)
        self.assertAlmostEqual(log_spec.to_scaled(10.0), 0.5)
        self.assertAlmostEqual(log_spec.from_scaled(0.5), 10.0)

    def test_free_set_and_locks(self):
        graph = sensor_graph("all-pass")
        graph.lock("ring", "radius_um")
        bound = bind(graph, system_specs(SystemObjective()), free={("cpl", "gap_nm"), ("ring", "radius_um")})
        free = {(spec.node_id, spec.key) for spec in bound.free_device}
        self.assertEqual(free, {("cpl", "gap_nm")})
        packed = bound.pack()
        self.assertEqual(packed.shape, (1,))
        self.assertGreater(len(bound.system), 0)
        self.assertTrue(all(spec.frozen and spec.level == "system" for spec in bound.system))

    def test_apply_writes_only_free_parameters(self):
        graph = sensor_graph("all-pass")
        bound = bind(graph, [], free={("cpl", "gap_nm")})
        tuned = bound.apply(graph, np.array([1.0]))
        self.assertAlmostEqual(tuned.nodes["cpl"].params["gap_nm"], 500.0)
        self.assertEqual(tuned.nodes["cpl"].params["length_um"], 12.0)
        self.assertEqual(graph.nodes["cpl"].params["gap_nm"], 200.0)


class PipelineTest(unittest.TestCase):
    def test_inner_tune_moves_kappa_toward_target(self):
        graph = sensor_graph("all-pass")
        objective = SystemObjective(kappa_weight=1.0, kappa_target=0.1)
        before = forward(graph, Stack())
        result = inner_tune(
            graph,
            objective=objective,
            steps=12,
            step_size=0.03,
            free={("cpl", "gap_nm"), ("cpl", "length_um")},
        )
        after = result["metrics"]["kappa"]
        self.assertLess(abs(after - 0.1), abs(before["kappa"] - 0.1))
        self.assertLess(result["loss"], result["initial_loss"])
        self.assertEqual(result["history"][-1], result["loss"])
        radius = next(row for row in result["parameters"]["device"] if row["key"] == "radius_um")
        self.assertTrue(radius["frozen"])
        self.assertEqual(radius["value"], radius["initial"])

    def test_outer_level_rejects_the_locked_coupler(self):
        objective = SystemObjective(kappa_weight=1.0, kappa_target=0.1)
        locked = sensor_graph("all-pass")
        locked.name = "locked coupler"
        locked.nodes["cpl"].params["gap_nm"] = 450.0
        locked.lock("cpl", "gap_nm")
        locked.lock("cpl", "length_um")
        open_gap = sensor_graph("all-pass")
        open_gap.name = "open coupler"
        open_gap.nodes["cpl"].params["gap_nm"] = 450.0
        result = run_inverse(
            graphs=[locked, open_gap],
            objective=objective,
            steps=10,
            step_size=0.04,
            free={("cpl", "gap_nm"), ("cpl", "length_um")},
        )
        self.assertEqual(result["selected"], "open coupler")
        by_name = {candidate["graph"]["name"]: candidate for candidate in result["candidates"]}
        self.assertLess(by_name["open coupler"]["loss"], by_name["locked coupler"]["loss"])
        self.assertEqual(result["levels"]["outer"], "discrete DAG architecture")
        self.assertEqual(result["levels"]["inner"], "continuous device parameters")

    def test_both_architectures_return_two_candidates(self):
        result = run_inverse(config="both", steps=1, step_size=0.01)
        self.assertEqual(len(result["candidates"]), 2)
        self.assertEqual(sum(1 for candidate in result["candidates"] if candidate["selected"]), 1)
        self.assertIn(result["selected"], {"all-pass sensor", "add-drop sensor"})


if __name__ == "__main__":
    unittest.main()
