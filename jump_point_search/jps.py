from manim import *

import sys, os
import numpy as np

# Get the absolute path to the directory containing the target of the symlink
# os.path.realpath(__file__) gets the path to the resolved target file
symlink_dir = os.path.dirname(os.path.realpath(__file__))
PATHFINDING_DIR = os.path.join(symlink_dir, "pathfinding_link")
sys.path.append(PATHFINDING_DIR)

from search.nodes import Node
from domains.astar import Astar
from domains.gridmap import Gridmap
from domains.jps import JumpPointSearch
from search.tie_breakers import StandardTieBreaker
from search.heuristics import OctileDistanceHeuristic
from utils.directions import Direction
from utils.scenario_parser import ScenarioParser


SCENARIO_FILE = "maps/lak303d.map.scen"
SCENARIO_INDEX = 385

CELL = 1.0
FREE_RGB = (226, 230, 238)
WALL_RGB = (23, 28, 43)

SCAN_HIT = "#F08C00"   # a scan that terminated on a jump point
SCAN_MISS = "#8E99AB"  # a scan that ran into an obstacle
CURRENT = "#7048E8"
PATH = "#1C7ED6"
START = "#2F9E44"
GOAL = "#E03131"

FULL_VIEW_MARGIN = 8 * CELL
ZOOM_CELLS = 46
DETAILED_EXPANSIONS = 6
BATCH_SIZE = 6

# Radii are per-view: mobjects are rescaled on each camera change so that they keep a
# sensible on-screen size at both zoom levels.
JUMP_RADIUS_ZOOM = 0.55 * CELL
JUMP_RADIUS_FULL = 1.1 * CELL
MARKER_RADIUS_ZOOM = 0.9 * CELL
MARKER_RADIUS_FULL = 3.0 * CELL


def load_map_array(path):
    """Parse a MovingAI .map into a padded (height+2, width+2) bool array of traversable cells.

    Padding matches Gridmap's, so a padded search state indexes straight into this array.
    """
    with open(path, "rb") as map_file:
        map_file.readline()  # type
        height = int(map_file.readline().split()[1])
        width = int(map_file.readline().split()[1])
        map_file.readline()  # "map"
        body = map_file.read()

    cells = np.frombuffer(body.translate(None, b"\r\n"), dtype=np.uint8)
    cells = cells[: width * height].reshape(height, width)

    free = np.zeros((height + 2, width + 2), dtype=bool)
    free[1:-1, 1:-1] = cells == ord(".")  # '.' only, matching Gridmap.load
    return free


def stroke_at(pixels, frame_width):
    """Stroke width that renders `pixels` thick while the camera frame is `frame_width` wide."""
    return pixels * frame_width / config.frame_width


def step_direction(origin, target):
    return (int(np.sign(target[0] - origin[0])), int(np.sign(target[1] - origin[1])))


def scan_end(free, state, direction):
    """Last cell a JPS scan from `state` along `direction` can reach before an obstacle."""
    x, y = state
    dx, dy = direction
    diagonal = dx != 0 and dy != 0
    while free[y + dy][x + dx] and (not diagonal or (free[y + dy][x] and free[y][x + dx])):
        x, y = x + dx, y + dy
    return x, y


class GridMapImage(Group):
    """A whole gridmap as one ImageMobject, with cell <-> scene coordinate conversion.

    One mobject regardless of map size: a 194x194 map is 37k cells, which is far past
    what a VGroup of Squares can render in reasonable time.
    """

    def __init__(self, free, cell=CELL):
        super().__init__()
        self.free = free
        self.cell = cell
        self.rows, self.cols = free.shape

        rgb = np.where(free[:, :, None], np.array(FREE_RGB, np.uint8), np.array(WALL_RGB, np.uint8))
        image = ImageMobject(rgb)
        image.set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])
        image.height = self.rows * cell
        image.move_to(ORIGIN)
        image.set_z_index(0)

        self.image = image
        self.add(image)

    def cell_point(self, state):
        x, y = state
        return np.array([
            (x + 0.5 - self.cols / 2) * self.cell,
            (self.rows / 2 - y - 0.5) * self.cell,
            0.0,
        ])

    def cell_points(self, states):
        arr = np.asarray(states, dtype=float)
        out = np.zeros((len(arr), 3))
        out[:, 0] = (arr[:, 0] + 0.5 - self.cols / 2) * self.cell
        out[:, 1] = (self.rows / 2 - arr[:, 1] - 0.5) * self.cell
        return out


DIR_BY_INDEX = [d for d, _ in sorted(Direction.dir_map.items(), key=lambda item: item[1])]


class Expansion:
    def __init__(self, state, rays, jump_points):
        self.state = state
        self.rays = rays  # (end_state, hit_jump_point) per direction scanned
        self.jump_points = jump_points


class RecordingJPS(JumpPointSearch):
    """JPS expander that records the cells scanned and jump points found per expansion."""

    def __init__(self, gridmap, heuristic, goal, free):
        super().__init__(gridmap, heuristic, goal)
        self.free = free
        self.events = []

    def expand(self, node):
        successors = super().expand(node)
        found = {step_direction(node.state, s.state): s.state for s in successors}

        rays = []
        direction = self.get_dir_from_jp(node.parent, node)
        for index, should_scan in enumerate(self.compute_successors(node, direction)):
            if not should_scan:
                continue
            scanned = DIR_BY_INDEX[index]
            if scanned in found:
                rays.append((found[scanned], True))
            else:
                rays.append((scan_end(self.free, node.state, scanned), False))

        self.events.append(Expansion(node.state, rays, list(found.values())))
        return successors


def run_search(scenario_file=SCENARIO_FILE, scenario_index=SCENARIO_INDEX):
    scenarios = ScenarioParser().parse(os.path.join(PATHFINDING_DIR, scenario_file), direct_mapnames=True)
    scenario = scenarios[scenario_index]
    map_path = os.path.join(PATHFINDING_DIR, scenario.map_file)

    gridmap = Gridmap()
    gridmap.load(map_path)
    free = load_map_array(map_path)

    heuristic = OctileDistanceHeuristic()
    goal_state = gridmap.cnvt_to_padded(scenario.goal)
    expander = RecordingJPS(gridmap, heuristic, Node(state=goal_state), free)
    astar = Astar(gridmap, heuristic, StandardTieBreaker(), expander)

    path = astar.search(gridmap.cnvt_to_padded(scenario.start), goal_state)
    print(f"scenario {scenario_index}: {len(expander.events)} expansions, {len(path)} jump points")
    return free, path, expander.events


class JumpPointSearchScene(MovingCameraScene):
    def build_rays(self, grid, event, frame_width):
        origin = grid.cell_point(event.state)
        return VGroup(*[
            Line(
                origin,
                grid.cell_point(end),
                color=SCAN_HIT if hit else SCAN_MISS,
                stroke_width=stroke_at(6 if hit else 3, frame_width),
                z_index=1,
            )
            for end, hit in event.rays
        ])

    def build_jump_points(self, grid, event, radius):
        return VGroup(*[
            Dot(grid.cell_point(state), radius=radius, color=SCAN_HIT, z_index=2)
            for state in event.jump_points
        ])

    def construct(self):
        free, path, events = run_search()
        grid = GridMapImage(free)

        frame = self.camera.frame
        full_width = FULL_VIEW_MARGIN + max(
            grid.image.width,
            grid.image.height * config.frame_width / config.frame_height,
        )
        zoom_width = ZOOM_CELLS * CELL

        start_state, goal_state = path[0], path[-1]
        start_dot = Dot(grid.cell_point(start_state), radius=MARKER_RADIUS_FULL, color=START, z_index=5)
        goal_dot = Dot(grid.cell_point(goal_state), radius=MARKER_RADIUS_FULL, color=GOAL, z_index=5)

        frame.set(width=full_width).move_to(grid.image)
        self.play(FadeIn(grid), run_time=1.5)
        self.play(GrowFromCenter(start_dot), GrowFromCenter(goal_dot))
        self.wait(0.5)

        # Follow the first few expansions closely: this is where the scan/jump mechanic reads.
        marker_shrink = MARKER_RADIUS_ZOOM / MARKER_RADIUS_FULL
        cursor = Circle(
            radius=1.7 * CELL,
            color=CURRENT,
            stroke_width=stroke_at(3, zoom_width),
            z_index=4,
        ).move_to(grid.cell_point(events[0].state))

        self.play(
            frame.animate.set(width=zoom_width).move_to(grid.cell_point(start_state)),
            *[dot.animate.scale(marker_shrink) for dot in (start_dot, goal_dot)],
            run_time=1.5,
        )
        self.play(Create(cursor), run_time=0.4)

        explored = []
        for event in events[:DETAILED_EXPANSIONS]:
            rays = self.build_rays(grid, event, zoom_width)
            jump_points = self.build_jump_points(grid, event, JUMP_RADIUS_ZOOM)
            focus = grid.cell_point(event.state)

            self.play(frame.animate.move_to(focus), cursor.animate.move_to(focus), run_time=0.7)
            self.play(LaggedStart(*[Create(ray) for ray in rays], lag_ratio=0.2), run_time=1.0)
            self.play(LaggedStart(*[GrowFromCenter(d) for d in jump_points], lag_ratio=0.2), run_time=0.6)
            self.play(FadeOut(rays), run_time=0.4)
            explored.extend(jump_points)

        # Zoom back out and burn through the rest of the search in batches.
        jump_grow = JUMP_RADIUS_FULL / JUMP_RADIUS_ZOOM
        self.play(
            frame.animate.set(width=full_width).move_to(grid.image),
            *[dot.animate.scale(1 / marker_shrink) for dot in (start_dot, goal_dot)],
            *[dot.animate.scale(jump_grow) for dot in explored],
            FadeOut(cursor),
            run_time=1.5,
        )

        remaining = events[DETAILED_EXPANSIONS:]
        for i in range(0, len(remaining), BATCH_SIZE):
            batch = remaining[i : i + BATCH_SIZE]
            rays = VGroup(*[ray for e in batch for ray in self.build_rays(grid, e, full_width)])
            jump_points = VGroup(*[
                dot for e in batch for dot in self.build_jump_points(grid, e, JUMP_RADIUS_FULL)
            ])

            self.play(Create(rays, lag_ratio=0), run_time=0.5)
            self.play(FadeOut(rays), FadeIn(jump_points), run_time=0.4)
            explored.extend(jump_points)

        self.wait(0.5)

        path_line = VMobject(z_index=3)
        path_line.set_points_as_corners(grid.cell_points(path))
        path_line.set_stroke(color=PATH, width=stroke_at(5, full_width))

        self.play(*[dot.animate.set_opacity(0.2) for dot in explored], run_time=0.8)
        self.play(Create(path_line), run_time=3, rate_func=linear)
        self.wait(2)
