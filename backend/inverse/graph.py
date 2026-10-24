"""Photonic systems as directed acyclic graphs of components.

Edges are dependencies between ports (optical power or heat), not the
electromagnetic field. A ring's round trip stays inside the ring node, so a
resonator circuit is still a DAG of components. Cycles are rejected.
"""

from __future__ import annotations

from dataclasses import dataclass


class GraphError(ValueError):
    """The component graph is not a valid DAG."""


@dataclass(frozen=True)
class PortSpec:
    name: str
    flow: str  # "in" or "out"
    kind: str  # "optical" or "thermal"


_PORTS: dict[str, tuple[PortSpec, ...]] = {
    "source": (PortSpec("out", "out", "optical"),),
    "waveguide": (
        PortSpec("in", "in", "optical"),
        PortSpec("out", "out", "optical"),
    ),
    "coupler": (
        PortSpec("in", "in", "optical"),
        PortSpec("through", "out", "optical"),
        PortSpec("cross", "out", "optical"),
        PortSpec("heat", "in", "thermal"),
    ),
    "ring": (
        PortSpec("bus", "in", "optical"),
        PortSpec("drop", "out", "optical"),
        PortSpec("heat", "in", "thermal"),
    ),
    "detector": (PortSpec("in", "in", "optical"),),
    "heater": (PortSpec("out", "out", "thermal"),),
}


def ports_of(kind: str) -> dict[str, PortSpec]:
    try:
        specs = _PORTS[kind]
    except KeyError as exc:
        known = ", ".join(sorted(_PORTS))
        raise GraphError(f"unknown component kind {kind!r}; expected one of {known}") from exc
    return {spec.name: spec for spec in specs}


@dataclass(frozen=True)
class Edge:
    src: str
    src_port: str
    dst: str
    dst_port: str
    kind: str


class Node:
    def __init__(self, node_id: str, kind: str, params: dict):
        ports_of(kind)
        self.id = node_id
        self.kind = kind
        self.params = dict(params)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "kind": self.kind,
            "params": dict(self.params),
            "ports": [
                {"name": spec.name, "flow": spec.flow, "kind": spec.kind}
                for spec in _PORTS[self.kind]
            ],
        }


class PhotonicGraph:
    """Insertion-ordered component DAG."""

    def __init__(self, name: str = "system"):
        self.name = name
        self.nodes: dict[str, Node] = {}
        self.edges: list[Edge] = []
        # (node id, param key) held fixed by the inner tuner.
        self.locks: set[tuple[str, str]] = set()

    def add(self, node_id: str, kind: str, params: dict | None = None) -> Node:
        if not isinstance(node_id, str) or not node_id:
            raise GraphError("node id must be a non-empty string")
        if node_id in self.nodes:
            raise GraphError(f"duplicate node id {node_id!r}")
        node = Node(node_id, kind, params or {})
        self.nodes[node_id] = node
        return node

    def connect(self, src: str, src_port: str, dst: str, dst_port: str) -> PhotonicGraph:
        if src not in self.nodes or dst not in self.nodes:
            raise GraphError(f"edge {src}.{src_port} → {dst}.{dst_port} names a missing node")
        if src == dst:
            raise GraphError(f"edge {src}.{src_port} → {dst}.{dst_port} would cycle")
        src_spec = self._require_port(self.nodes[src], src_port)
        dst_spec = self._require_port(self.nodes[dst], dst_port)
        if src_spec.flow != "out":
            raise GraphError(f"{src}.{src_port} is not an output")
        if dst_spec.flow != "in":
            raise GraphError(f"{dst}.{dst_port} is not an input")
        if src_spec.kind != dst_spec.kind:
            raise GraphError(
                f"{src}.{src_port} ({src_spec.kind}) cannot drive "
                f"{dst}.{dst_port} ({dst_spec.kind})"
            )
        for edge in self.edges:
            if edge.dst == dst and edge.dst_port == dst_port:
                raise GraphError(f"{dst}.{dst_port} already has a driver")
            if (
                edge.src == src
                and edge.src_port == src_port
                and edge.dst == dst
                and edge.dst_port == dst_port
            ):
                raise GraphError(f"duplicate edge {src}.{src_port} → {dst}.{dst_port}")
        self.edges.append(Edge(src, src_port, dst, dst_port, src_spec.kind))
        try:
            self.topological_order()
        except GraphError:
            self.edges.pop()
            raise
        return self

    def lock(self, node_id: str, key: str) -> PhotonicGraph:
        if node_id not in self.nodes:
            raise GraphError(f"cannot lock unknown node {node_id!r}")
        self.locks.add((node_id, key))
        return self

    def topological_order(self) -> list[str]:
        indeg = {node_id: 0 for node_id in self.nodes}
        for edge in self.edges:
            indeg[edge.dst] += 1
        ready = [node_id for node_id in self.nodes if indeg[node_id] == 0]
        order: list[str] = []
        while ready:
            node_id = ready.pop(0)
            order.append(node_id)
            for edge in self.edges:
                if edge.src != node_id:
                    continue
                indeg[edge.dst] -= 1
                if indeg[edge.dst] == 0:
                    ready.append(edge.dst)
        if len(order) != len(self.nodes):
            stuck = [node_id for node_id in self.nodes if node_id not in order]
            raise GraphError(f"cycle through {', '.join(stuck)}")
        return order

    def to_devices(self) -> list[dict]:
        """Flat placement list consumed by the compact circuit model. Sources are omitted."""
        devices = []
        for node_id in self.topological_order():
            node = self.nodes[node_id]
            if node.kind == "source":
                continue
            devices.append({"id": node.id, "type": node.kind, **node.params})
        return devices

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "order": self.topological_order(),
            "nodes": [self.nodes[node_id].to_dict() for node_id in self.nodes],
            "edges": [
                {
                    "src": edge.src,
                    "src_port": edge.src_port,
                    "dst": edge.dst,
                    "dst_port": edge.dst_port,
                    "kind": edge.kind,
                }
                for edge in self.edges
            ],
            "devices": self.to_devices(),
            "locks": [{"node_id": node_id, "key": key} for node_id, key in sorted(self.locks)],
        }

    def copy(self) -> PhotonicGraph:
        other = PhotonicGraph(self.name)
        for node in self.nodes.values():
            other.add(node.id, node.kind, node.params)
        for edge in self.edges:
            other.edges.append(edge)
        other.locks = set(self.locks)
        return other

    def bus_coupler_id(self) -> str | None:
        """Coupler whose cross port drives the ring bus."""
        for edge in self.edges:
            if (
                edge.dst_port == "bus"
                and self.nodes[edge.dst].kind == "ring"
                and self.nodes[edge.src].kind == "coupler"
                and edge.src_port == "cross"
            ):
                return edge.src
        return None

    def _require_port(self, node: Node, port: str) -> PortSpec:
        specs = ports_of(node.kind)
        if port not in specs:
            known = ", ".join(specs)
            raise GraphError(f"{node.id} ({node.kind}) has no port {port!r}; ports: {known}")
        return specs[port]


def sensor_graph(config: str = "all-pass", *, heater: bool = False) -> PhotonicGraph:
    """All-pass or add-drop ring sensor used by the rest of this app.

    source → bus → coupler —cross→ ring
                         —through→ detector
    Add-drop continues ring —drop→ drop coupler —cross→ drop bus → drop detector.
    """
    if config not in ("all-pass", "add-drop"):
        raise GraphError("config must be 'all-pass' or 'add-drop'")
    graph = PhotonicGraph(f"{config} sensor")
    graph.add("src", "source")
    graph.add("bus", "waveguide", {"length_um": 100.0})
    graph.add("cpl", "coupler", {"gap_nm": 200.0, "length_um": 12.0})
    graph.add("ring", "ring", {"radius_um": 10.0, "config": config})
    graph.add(
        "det",
        "detector",
        {
            "responsivity_a_per_w": 0.9,
            "dark_current_na": 5.0,
            "bandwidth_mhz": 100.0,
            "load_ohm": 50.0,
        },
    )
    graph.connect("src", "out", "bus", "in")
    graph.connect("bus", "out", "cpl", "in")
    graph.connect("cpl", "cross", "ring", "bus")
    graph.connect("cpl", "through", "det", "in")
    if config == "add-drop":
        graph.add("cpl_drop", "coupler", {"gap_nm": 200.0, "length_um": 12.0})
        graph.add("drop_bus", "waveguide", {"length_um": 100.0})
        graph.add(
            "det_drop",
            "detector",
            {
                "responsivity_a_per_w": 0.9,
                "dark_current_na": 5.0,
                "bandwidth_mhz": 100.0,
                "load_ohm": 50.0,
            },
        )
        graph.connect("ring", "drop", "cpl_drop", "in")
        graph.connect("cpl_drop", "cross", "drop_bus", "in")
        graph.connect("drop_bus", "out", "det_drop", "in")
    if heater:
        graph.add("heater", "heater", {"power_mw": 10.0, "width_um": 2.0, "length_um": 10.0})
        graph.connect("heater", "out", "ring", "heat")
    return graph


def from_devices(devices: list[dict], name: str = "placement") -> PhotonicGraph:
    """Canonical sensor DAG for a flat placement list.

    The first coupler drives the ring. A second coupler is the drop coupler when
    the ring is add-drop. The first waveguide is the bus; further waveguides are
    the drop bus in an add-drop circuit and extra bus sections otherwise.
    """
    rings = [device for device in devices if device.get("type") == "ring"]
    if len(rings) != 1:
        raise GraphError("sensor DAG expects exactly one ring")
    config = rings[0].get("config", "all-pass")
    if config not in ("all-pass", "add-drop"):
        raise GraphError("ring config must be 'all-pass' or 'add-drop'")

    graph = PhotonicGraph(name)
    graph.add("src", "source")
    graph.add("ring", "ring", _params(rings[0]))

    waveguides = [device for device in devices if device.get("type") == "waveguide"]
    couplers = [device for device in devices if device.get("type") == "coupler"]
    detectors = [device for device in devices if device.get("type") == "detector"]
    heaters = [device for device in devices if device.get("type") == "heater"]

    if config == "add-drop" and len(waveguides) >= 2:
        bus_wgs = waveguides[:1]
        drop_wgs = waveguides[1:]
    else:
        bus_wgs = waveguides
        drop_wgs = []

    previous = "src"
    previous_port = "out"
    for index, device in enumerate(bus_wgs):
        node_id = _node_id(device, "bus" if index == 0 else f"bus_{index}")
        graph.add(node_id, "waveguide", _params(device))
        graph.connect(previous, previous_port, node_id, "in")
        previous, previous_port = node_id, "out"

    bus_coupler = couplers[0] if couplers else None
    if bus_coupler is not None:
        coupler_id = _node_id(bus_coupler, "cpl")
        graph.add(coupler_id, "coupler", _params(bus_coupler))
        graph.connect(previous, previous_port, coupler_id, "in")
        graph.connect(coupler_id, "cross", "ring", "bus")
        if detectors:
            det_id = _node_id(detectors[0], "det")
            graph.add(det_id, "detector", _params(detectors[0]))
            graph.connect(coupler_id, "through", det_id, "in")
    elif detectors:
        det_id = _node_id(detectors[0], "det")
        graph.add(det_id, "detector", _params(detectors[0]))
        graph.connect(previous, previous_port, det_id, "in")

    if config == "add-drop" and len(couplers) >= 2:
        drop_id = _node_id(couplers[1], "cpl_drop")
        graph.add(drop_id, "coupler", _params(couplers[1]))
        graph.connect("ring", "drop", drop_id, "in")
        tail, tail_port = drop_id, "cross"
        for index, device in enumerate(drop_wgs):
            node_id = _node_id(device, "drop_bus" if index == 0 else f"drop_bus_{index}")
            graph.add(node_id, "waveguide", _params(device))
            graph.connect(tail, tail_port, node_id, "in")
            tail, tail_port = node_id, "out"
        if len(detectors) >= 2:
            det_id = _node_id(detectors[1], "det_drop")
            graph.add(det_id, "detector", _params(detectors[1]))
            graph.connect(tail, tail_port, det_id, "in")

    if len(heaters) > 1:
        raise GraphError("ring heat port accepts one heater")
    if heaters:
        node_id = _node_id(heaters[0], "heater")
        graph.add(node_id, "heater", _params(heaters[0]))
        graph.connect(node_id, "out", "ring", "heat")
    return graph


def _node_id(device: dict, fallback: str) -> str:
    node_id = device.get("id")
    if isinstance(node_id, str) and node_id:
        return node_id
    return fallback


def _params(device: dict) -> dict:
    skip = {"id", "type", "x", "y"}
    return {key: value for key, value in device.items() if key not in skip}
