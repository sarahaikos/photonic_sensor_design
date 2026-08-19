"""Geometric primitives for the FDTD domain (rectangles, cylinders, polygons).

GDSII layouts map onto these primitives. Binary GDS parsing is a loader in
front of Polygon; the Yee grid only ever sees these shapes.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Rect:
    x0: float
    y0: float
    width: float
    height: float
    material: str
    role: str = ""

    def contains(self, x: float, y: float) -> bool:
        return self.x0 <= x <= self.x0 + self.width and self.y0 <= y <= self.y0 + self.height

    def to_dict(self) -> dict:
        return {
            "kind": "rectangle",
            "x0": self.x0,
            "y0": self.y0,
            "width": self.width,
            "height": self.height,
            "material": self.material,
            "role": self.role,
        }


@dataclass
class Cylinder:
    x: float
    y: float
    radius: float
    material: str
    role: str = ""
    inner_radius: float = 0.0

    def contains(self, x: float, y: float) -> bool:
        r2 = (x - self.x) ** 2 + (y - self.y) ** 2
        if r2 > self.radius**2:
            return False
        if self.inner_radius > 0 and r2 < self.inner_radius**2:
            return False
        return True

    def to_dict(self) -> dict:
        return {
            "kind": "cylinder",
            "x": self.x,
            "y": self.y,
            "radius": self.radius,
            "inner_radius": self.inner_radius,
            "material": self.material,
            "role": self.role,
        }


@dataclass
class Polygon:
    vertices: list[tuple[float, float]]
    material: str
    role: str = ""
    layer: int = 0

    def contains(self, x: float, y: float) -> bool:
        inside = False
        n = len(self.vertices)
        for i in range(n):
            x1, y1 = self.vertices[i]
            x2, y2 = self.vertices[(i + 1) % n]
            if (y1 > y) != (y2 > y):
                xin = (x2 - x1) * (y - y1) / (y2 - y1 + 1e-30) + x1
                if x < xin:
                    inside = not inside
        return inside

    def to_dict(self) -> dict:
        return {
            "kind": "polygon",
            "vertices": [list(v) for v in self.vertices],
            "material": self.material,
            "role": self.role,
            "layer": self.layer,
        }


Shape = Rect | Cylinder | Polygon


@dataclass
class Scene:
    shapes: list[Shape] = field(default_factory=list)
    background: str = "clad"
    units: str = "um"

    def add(self, shape: Shape) -> None:
        self.shapes.append(shape)

    def material_at(self, x: float, y: float) -> str:
        for shape in reversed(self.shapes):
            if shape.contains(x, y):
                return shape.material
        return self.background

    def bounds(self) -> tuple[float, float, float, float]:
        xs: list[float] = []
        ys: list[float] = []
        for s in self.shapes:
            if isinstance(s, Rect):
                xs += [s.x0, s.x0 + s.width]
                ys += [s.y0, s.y0 + s.height]
            elif isinstance(s, Cylinder):
                xs += [s.x - s.radius, s.x + s.radius]
                ys += [s.y - s.radius, s.y + s.radius]
            else:
                xs += [v[0] for v in s.vertices]
                ys += [v[1] for v in s.vertices]
        if not xs:
            return 0.0, 1.0, 0.0, 1.0
        pad = 0.6
        return min(xs) - pad, max(xs) + pad, min(ys) - pad, max(ys) + pad

    def to_list(self) -> list[dict]:
        return [s.to_dict() for s in self.shapes]


def polygons_from_layout(records: list[dict]) -> list[Polygon]:
    """Import a GDS-like layout: [{layer, material, vertices:[[x,y],...]}]."""
    out = []
    for rec in records:
        verts = [(float(x), float(y)) for x, y in rec["vertices"]]
        out.append(
            Polygon(
                vertices=verts,
                material=rec.get("material", "si"),
                role=rec.get("role", "gds"),
                layer=int(rec.get("layer", 0)),
            )
        )
    return out
