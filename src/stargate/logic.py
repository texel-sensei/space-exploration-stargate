import math
from itertools import pairwise
from dataclasses import dataclass

import pygame
from pygame import draw
import numpy as np


upper_vertex_direction = np.array([0.8506508337159, 0, -0.52573107107952])
left_vertex_direction = np.array([0.93417235727245, 0.35682209419822, 0])
right_vertex_direction = np.array(
    [0.57735041155694, 0.57735018458227, -0.57735021142964]
)
triangle_center_direction = np.array(
    [0.85296865578697, 0.3373248250886, -0.39831700267993]
)

goal_direction = -np.array([-0.93775912744763, -0.30188301187802, -0.17168129291553])


def main(mouse: "Point | None"):
    center = pt(500, 500)
    show_point(center)
    vis = Triangle.from_center(center, 500)

    # triangle_plane = Plane(triangle_center_direction, 1)

    log(f"Center: {center}")
    log(f"Mouse: {mouse}")
    vis.draw()
    h = vis.height()

    if mouse is not None:
        log(f"Coord: {vis.coords(mouse, 8)}")

    for i in range(1, 9):
        edge = (vis[0], vis[2])
        n = normalized(edge[0] - edge[1])
        n[0], n[1] = n[1], -n[0]

        delta = n * i / 8 * vis.height()

        draw.line(screen, "green", edge[0] + delta, edge[1] + delta)

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

        draw.line(screen, "lavender", edge[0] + delta, edge[1] + delta)


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

    def draw(self):
        for start, end in pairwise(self.vertices + [self.vertices[0]]):
            draw.line(screen, "red", start, end)

    def height(self):
        a = np.linalg.norm(self.vertices[0] - self.vertices[1])
        return math.sqrt(3) / 2 * a

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

    def intersect(self, other: Line) -> Point:
        pass


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
