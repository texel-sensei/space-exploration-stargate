import math
from dataclasses import dataclass

import pygame
from pygame import draw
import numpy as np

SHOW_GRID = True

#upper_vertex_direction = np.array([0.8506508337159, 0, -0.52573107107952], dtype="float64")
upper_vertex_direction = np.array([0.93860686319311, 0.296438286539, -0.17646954026295], dtype="float64")
#left_vertex_direction = np.array([0.93417235727245, 0.35682209419822, 0], dtype="float64")
left_vertex_direction = np.array([0.93851870026722, 0.3018409243365, -0.16755508241893], dtype="float64")
#right_vertex_direction = np.array( [0.57735041155694, 0.57735018458227, -0.57735021142964], dtype="float64")
right_vertex_direction = np.array( [0.93523107507538, 0.30630887945282, -0.1775322691279], dtype="float64")
triangle_center_direction = np.array(
    [0.85296865578697, 0.3373248250886, -0.39831700267993], dtype="float64"
)

goal_direction = -np.array([-0.93775912744763, -0.30188301187802, 0.17168129291553], dtype="float64")

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
    #show_point(center)
    vis = Triangle.from_center(center, 500)
    #vis[0] += pt(600,300)

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
            start = vis.barycentric_to_cartesian(
                tri_on_plane.cartesian_to_barycentric(triangle_plane.project_point(points[i]))
            )
            end = vis.barycentric_to_cartesian(
                tri_on_plane.cartesian_to_barycentric(triangle_plane.project_point(points[i + 1]))
            )
            draw.line(screen, "green", start, end)

    #draw_tested(points_on_070)
    #draw_tested(points_on_105)
    #draw_tested(points_on_205)
    #draw_tested(points_on_222)

    if False:
        vis = tri_on_plane
        projected = target_point
    else:
        bar = tri_on_plane.cartesian_to_barycentric(target_point)
        projected = vis.barycentric_to_cartesian(bar)
        show_point(projected)

    if SHOW_GRID:
        n = vis.normal()
        for e in range(3):
            edge = vis.edge(e)
            delta = vis.height(e)/8
            for i in range(1,9):
                edge.shift(i*delta, n).draw('teal')

    def to_2D(dir) -> Point:
        point = triangle_plane.project_point(dir)
        bar = tri_on_plane.cartesian_to_barycentric(point)
        return vis.barycentric_to_cartesian(bar)

    def vis_stargate_dir(dir, color):
        show_point(to_2D(dir), color)

    actual = np.array([0.92675364243318, 0.33209708435429, -0.17561097004518])
    actual2 = np.array([0.93534662471347, 0.32239837989494, -0.14555403216068])
    actual3 = np.array([0.93784352415677, 0.30058757678801, -0.17348381157307])
    actual4 = np.array([0.93775916592287, 0.30188269207433, -0.17168164419532])
    #         np.array([0.93775912744763, 0.30188301187802, -0.17168129291553])
    vis_stargate_dir(actual, 'red')
    vis_stargate_dir(actual2, 'teal')
    vis_stargate_dir(actual3, 'white')

    # vis_stargate_dir(upper_vertex_direction2, 'red')
    # vis_stargate_dir(left_vertex_direction2, 'red')
    # vis_stargate_dir(right_vertex_direction2, 'red')

    #projected = to_2D(actual4)

    t = vis
    scale = 8
    log("Coord: [0 1 5]")
    log("Coord: [3 0 3]")
    for _ in range(5):
        c = t.coords(projected, scale)
        t.draw("white")
        log(f"Coord: {c}")
        t = t.inner_tri(c, scale)

    log(f"Calculated: {t.center()}")

    log(f"Center: {center}")
    log(f"Mouse: {mouse}")
    vis.draw()
    # vis.draw_normal("green")

    if mouse is not None and vis.is_inside(mouse):
        scale = 8
        t = vis
        a,b,c = vis.cartesian_to_barycentric(mouse)*8
        a=int(a)
        b=int(b)
        c=int(c)
        log(f"Mouse bar: {c} {a} {b}")
        for _ in range(1):
            c = t.coords(mouse, scale)
            log(f"Coord: {c}")
            t = t.inner_tri(c, scale)
            #t.draw("white")


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
        delta = np.array([d, 0], dtype="float64")

        return cls(
            [
                c + delta @ rot(90),  # top
                c + delta @ rot(90 + 120),  # left
                c + delta @ rot(90 + 240),
            ]
        )

    def center(self) -> Point:
        return sum(self.vertices)/3

    def edge(self, i) -> "Line":
        match i:
            case 0:
                return Line(self[0], self[1])
            case 1:
                return Line(self[1], self[2])
            case 2:
                return Line(self[2], self[0])

    def coords(self, p: Point, scale: int) -> np.ndarray:
        res = []
        n = self.normal()
        for i in range(3):
            h = self.height(i) / scale
            edge = self.edge(i)
            d = edge.distance(p,n)
            c = int(d / h)
            res.append(c)

        return np.array(res)

    def draw(self, color="red"):
        colors = ['red', 'green', 'blue']
        for e in range(3):
            self.edge(e).draw(colors[e])

    def area(self) -> float:
        # Herons formula
        a = self.edge(0).length()
        b = self.edge(1).length()
        c = self.edge(2).length()
        return math.sqrt(4*a*a*b*b - (a*a+b*b-c*c)**2)/4

    def height(self, edge: int):
        # area = 1/2 * base * height
        # height = 2*area/base
        base = self.edge(edge).length()
        return 2*self.area()/base

    def normal(self):
        a = self[1] - self[0]
        b = self[2] - self[0]
        return normalized(np.cross(a,b))

    def inner_tri(self, coords: np.ndarray, scale: int) -> "Triangle":
        is_upper = sum(coords) == (scale - 1)
        n = self.normal()

        if is_upper:
            l0 = self.edge(0).shift(coords[0] * self.height(0) / scale, n)
            l1 = self.edge(1).shift(coords[1] * self.height(1) / scale, n)
            l2 = self.edge(2).shift(coords[2] * self.height(2) / scale, n)
        else:
            l0 = self.edge(0).shift((coords[0] + 1) * self.height(0) / scale, n)
            l1 = self.edge(1).shift((coords[1] + 1) * self.height(1) / scale, n)
            l2 = self.edge(2).shift((coords[2] + 1) * self.height(2) / scale, n)

        a = l0.intersect(l1)
        b = l1.intersect(l2)
        c = l2.intersect(l0)

        return Triangle([c,a,b])
        #return Triangle([a,b,c])

    def is_inside(self, p: Point) -> bool:
        for i in range(3):
            e = self.edge(i)
            d = p - e.start
            if e.normal(self.normal()).dot(d) < 0:
                return False

        return True

    def draw_normal(self, color):
        for i in range(3):
            self.edge(i).draw_normal(color)

    def cartesian_to_barycentric(self, p: Point) -> np.ndarray:
        # a, b, c = self.vertices
        # v0 = b - a
        # v1 = c - a
        # v2 = p - a
        # d00 = v0.dot(v0)
        # d01 = v0.dot(v1)
        # d11 = v1.dot(v1)
        # d20 = v2.dot(v0)
        # d21 = v2.dot(v1)
        # denom = d00 * d11 - d01 * d01
        # v = (d11 * d20 - d01 * d21) / denom
        # w = (d00 * d21 - d01 * d20) / denom
        # u = 1 - v - w
        # return np.array([u, v, w])

        p0,p1,p2 = self.vertices
        area = 0.5*np.linalg.norm(np.cross((p1-p0),(p2-p0)))
        u = (np.linalg.norm(np.cross(p1-p,p2-p))*0.5)/area
        v = (np.linalg.norm(np.cross(p0-p,p2-p))*0.5)/area
        w = (np.linalg.norm(np.cross(p0-p,p1-p))*0.5)/area

        return np.array([u,v,w])

    def barycentric_to_cartesian(self, bar: np.ndarray) -> Point:
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

    def __setitem__(self, i, value):
        self.vertices[i] = value


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

    def normal(self, up) -> np.ndarray:
        if self.start.shape == (2,):
            n = normalized(self.end - self.start)
            n[0], n[1] = n[1], -n[0]
        else:
            diff = self.end-self.start
            n = normalized(np.cross(diff, up))

        return n

    def length(self) -> float:
        return np.linalg.norm(self.end - self.start)

    def distance(self, p: Point, up) -> float:
        delta = p - self.start
        return self.normal(up).dot(delta)

    def draw(self, color):
        draw.line(screen, color, self.start, self.end)

    def draw_normal(self, color):
        start = (self.start + self.end) / 2
        end = start + self.normal(None) * 50
        draw.line(screen, color, start, end)

    def intersect(self, other: "Line") -> Point:
        # a1 + lambda1*dir1 = a2 + lambda2*dir2
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

    def shift(self, distance: float, up) -> "Line":
        delta = self.normal(up) * distance
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
