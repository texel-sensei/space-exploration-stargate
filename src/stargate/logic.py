import math
from itertools import pairwise
from dataclasses import dataclass

import pygame
from pygame import draw
import numpy as np

upper_vertex_direction = np.array([0.8506508337159, 0, -0.52573107107952])
left_vertex_direction = np.array([0.93417235727245, 0.35682209419822, 0])
right_vertex_direction = np.array([0.57735041155694, 0.57735018458227, -0.57735021142964])
triangle_center_direction = np.array([0.85296865578697, 0.3373248250886, -0.39831700267993])

goal_direction = -np.array([-0.93775912744763, -0.30188301187802, 0.17168129291553])

points_on_070 = [
    np.array([0.86880679636461, 0.053744337006915, -0.49222585957094]),  # 205
    np.array([0.86242823714895, 0.057787140581026, -0.50286994556408]),  # 304
    np.array([0.83546825365882, 0.073846273919835, -0.54455442791004]),  # 700
]

points_on_222 = [
    np.array([0.84478490154347, 0.37966205311477, -0.37708778228532]),  # 007
    np.array([0.87942553666472, 0.30063274350054, -0.36909440390951]),  # 700
    np.array([0.85284918844395, 0.31802037944722, -0.41413922782894]),  # 304
    np.array([0.83121704043107, 0.33042723271945, -0.44709738936207]),  # 070
    np.array([0.84478490154347, 0.37966205311477, -0.37708778228532]),  # 007
]

points_on_205 = [
    np.array([0.85288311315261, 0.46836080771291, -0.2307131316107]),  # 700
    np.array([0.88845067774003, 0.43315420837694, -0.15176569107879]),  # 007
    np.array([0.89188942385196, 0.39260214388534, -0.22449234338315]),  # 070
    np.array([0.85288311315261, 0.46836080771291, -0.2307131316107]),  # 700
]

points_on_105 = [
    np.array([0.92293041850093, 0.35671470838645, -0.14475517063813]),  # 700
    np.array([0.89188951459678, 0.39260200117596, -0.22449223243791]),  # 007
]




def main(mouse: "Point | None"):
    center = pt(500, 500)
    show_point(center)
    vis = Triangle.from_center(center, 500)

    triangle_plane = Plane(triangle_center_direction, 1)

    tri_on_plane = Triangle(
        [
            triangle_plane.project_point(upper_vertex_direction),
            triangle_plane.project_point(left_vertex_direction),
            triangle_plane.project_point(right_vertex_direction),
        ]
    )
    target_point = triangle_plane.project_point(goal_direction)

    def draw_tested(points: list[np.ndarray]):
        for i in range(len(points) - 1):
            start = vis.from_barycentric(tri_on_plane.barycentric(triangle_plane.project_point(points[i])))
            end = vis.from_barycentric(tri_on_plane.barycentric(triangle_plane.project_point(points[i + 1])))
            draw.line(screen, "green", start, end)

    draw_tested(points_on_070)
    draw_tested(points_on_105)
    draw_tested(points_on_205)
    draw_tested(points_on_222)

    bar = tri_on_plane.barycentric(target_point)

    projected = vis.from_barycentric(bar)
    show_point(projected)

    t = vis
    scale = 8
    for _ in range(7):
        c = t.coords(projected, scale)
        t.draw("white")
        log(f"Coord: {c}")
        t = t.inner_tri(c, scale)

    log(f"Center: {center}")
    log(f"Mouse: {mouse}")
    vis.draw()
    # vis.draw_normal("green")
    h = vis.height()

    for i in range(1, 9):
        edge = (vis[0], vis[2])
        n = normalized(edge[0] - edge[1])
        n[0], n[1] = n[1], -n[0]

        delta = n * i / 8 * vis.height()

        draw.line(screen, "teal", edge[0] + delta, edge[1] + delta)

    for i in range(1, 9):
        edge = (vis[2], vis[1])
        n = normalized(edge[0] - edge[1])
        n[0], n[1] = n[1], -n[0]

        delta = n * i / 8 * vis.height()

        draw.line(screen, "teal", edge[0] + delta, edge[1] + delta)

    for i in range(1, 9):
        edge = (vis[1], vis[0])
        n = normalized(edge[0] - edge[1])
        n[0], n[1] = n[1], -n[0]

        delta = n * i / 8 * vis.height()

        draw.line(screen, "teal", edge[0] + delta, edge[1] + delta)

    if mouse is not None and vis.is_inside(mouse):
        scale = 8
        t = vis
        for _ in range(7):
            c = t.coords(mouse, scale)
            log(f"Coord: {c}")
            t = t.inner_tri(c, scale)
            t.draw("white")


# ================================================================================
# Helper functions
# ================================================================================

type Point = np.ndarray


def pt(x, y) -> Point:
    return np.array([x, y])


logs = []


@dataclass
class Triangle:
    vertices: list[Point]

    @classmethod
    def from_center(cls, c: Point, d: float):
        delta = np.array([d, 0])

        return cls(
            [
                c + delta @ rot(90),  # top
                c + delta @ rot(90 + 120),  # left
                c + delta @ rot(90 + 240),
            ]
        )

    def edge(self, i) -> "Line":
        match i:
            case 0:
                return Line(self[0], self[1])
            case 1:
                return Line(self[1], self[2])
            case 2:
                return Line(self[2], self[0])

    def coords(self, p: Point, scale: int) -> np.ndarray:
        h = self.height() / scale
        res = []
        for i in range(3):
            edge = self.edge(i)
            d = edge.distance(p)
            c = int(d / h)
            res.append(c)

        return np.array(res)

    def draw(self, color="red"):
        for start, end in pairwise(self.vertices + [self.vertices[0]]):
            draw.line(screen, color, start, end)

    def height(self):
        a = np.linalg.norm(self.vertices[0] - self.vertices[1])
        return math.sqrt(3) / 2 * a

    def inner_tri(self, coords: np.ndarray, scale: int) -> "Triangle":
        f = self.height() / scale
        is_upper = sum(coords) == (scale - 1)

        if is_upper:
            l0 = self.edge(0).shift(coords[0] * f)
            l1 = self.edge(1).shift(coords[1] * f)
            l2 = self.edge(2).shift(coords[2] * f)
        else:
            l0 = self.edge(0).shift((coords[0] + 1) * f)
            l1 = self.edge(1).shift((coords[1] + 1) * f)
            l2 = self.edge(2).shift((coords[2] + 1) * f)

        return Triangle([l0.intersect(l1), l1.intersect(l2), l2.intersect(l0)])

    def is_inside(self, p: Point) -> bool:
        for i in range(3):
            e = self.edge(i)
            d = p - e.start
            if e.normal().dot(d) < 0:
                return False

        return True

    def draw_normal(self, color):
        for i in range(3):
            self.edge(i).draw_normal(color)

    def barycentric(self, p: Point) -> np.ndarray:
        a, b, c = self.vertices
        v0 = b - a
        v1 = c - a
        v2 = p - a
        d00 = v0.dot(v0)
        d01 = v0.dot(v1)
        d11 = v1.dot(v1)
        d20 = v2.dot(v0)
        d21 = v2.dot(v1)
        denom = d00 * d11 - d01 * d01
        v = (d11 * d20 - d01 * d21) / denom
        w = (d00 * d21 - d01 * d20) / denom
        u = 1 - v - w
        return np.array([u, v, w])

    def from_barycentric(self, bar: np.ndarray) -> Point:
        a, b, c = self.vertices
        return a * bar[0] + b * bar[1] + c * bar[2]

    def __mul__(self, f):
        center = sum(self.vertices) / len(self.vertices)

        new_vertices = []
        for v in self.vertices:
            d = v - center
            new_vertices.append(center + f * d)

        return Triangle(new_vertices)

    def __getitem__(self, i):
        return self.vertices[i]


@dataclass
class Plane:
    normal: np.ndarray
    dist: float

    def __post_init__(self):
        self.normal = self.normal / np.linalg.norm(self.normal)

    def project_point(self, dir: np.ndarray) -> Point:
        ray_origin = np.zeros(3)

        denom = self.normal.dot(dir)
        assert abs(denom) > 0.0001

        plane_center = self.normal * self.dist
        diff = plane_center - ray_origin
        t = diff.dot(self.normal) / denom

        return ray_origin + dir * t


@dataclass
class Line:
    start: Point
    end: Point

    def normal(self) -> np.ndarray:
        n = normalized(self.end - self.start)
        n[0], n[1] = n[1], -n[0]

        return n

    def distance(self, p: Point) -> float:
        delta = p - self.start
        return self.normal().dot(delta)

    def draw(self, color):
        draw.line(screen, color, self.start, self.end)

    def draw_normal(self, color):
        start = (self.start + self.end) / 2
        end = start + self.normal() * 50
        draw.line(screen, color, start, end)

    def intersect(self, other: "Line") -> Point:
        x1, y1 = self.start
        x2, y2 = self.end
        x3, y3 = other.start
        x4, y4 = other.end

        x = ((x2 - x1) * (x3 * y4 - y3 * x4) - (x4 - x3) * (x1 * y2 - y1 * x2)) / (
                (x2 - x1) * (y4 - y3) - (y2 - y1) * (x4 - x3)
        )
        y = ((y2 - y1) * (x3 * y4 - y3 * x4) - (y4 - y3) * (x1 * y2 - y1 * x2)) / (
                (x2 - x1) * (y4 - y3) - (y2 - y1) * (x4 - x3)
        )

        return np.array([x, y])

    def shift(self, distance: float) -> "Line":
        delta = self.normal() * distance
        return Line(self.start + delta, self.end + delta)


font = pygame.font.SysFont(None, 48)


def update(surf: pygame.Surface):
    global screen
    screen = surf
    logs.clear()

    p = pygame.mouse.get_pos()
    main(pt(*p) if pygame.mouse.get_focused() else None)

    y = 0
    for text in logs:
        img = font.render(text, True, "white")
        screen.blit(img, (10, y))
        y += 48


def log(text: str):
    logs.append(text)


def show_point(p: Point, color="yellow"):
    draw.circle(screen, color, p, 3)


def normalized(x):
    return x / np.linalg.norm(x)


def rot(angle):
    angle = math.radians(angle)
    c = math.cos(angle)
    s = math.sin(angle)
    return np.array([[c, -s], [s, c]])
