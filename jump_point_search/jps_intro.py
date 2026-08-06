"""JPS up close, then JPS at full size: the scan procedure, and the search that follows.

Two framings of one search. The first sits on a handful of cells around the start: the
map is drawn with gridlines, every cell a scan reads is filled in and the scan itself is
drawn as a ray out of the cell that fired it, and nothing is ever cleared, so the shape
of an expansion stays on the map. The camera then pulls back to the whole map and plays
every remaining expansion, one per beat, through to the scan that lands on the goal.

orz304d is 75x155, which is the wrong way round for video, so the picture is turned a
quarter turn anticlockwise (see GridMapImage). Cell coordinates stay in map space
throughout; only the drawing is turned, which is why compass names in the narration go
through direction_name() before they reach the screen.

The scan rules below mirror domains/jps.py. They are re-derived here rather than reused
because the expander only reports where a jump landed, and this scene needs every cell
it read on the way there.
"""

from manim import *

import sys, os
import numpy as np
from itertools import zip_longest

symlink_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, symlink_dir)
PATHFINDING_DIR = os.path.join(symlink_dir, "pathfinding_link")
sys.path.append(PATHFINDING_DIR)

from search.nodes import Node
from domains.astar import Astar
from domains.gridmap import Gridmap
from domains.jps import JumpPointSearch
from search.tie_breakers import StandardTieBreaker
from search.heuristics import OctileDistanceHeuristic

from jps import CELL, GridMapImage, load_map_array, step_direction, stroke_at


MAP_FILE = "maps/orz304d.map"
# The free cells nearest the map's top-right and bottom-left corners. Turned, that is a run
# from the top left of the frame to the bottom right, across the long axis of the map.
START_CELL = (55, 20)
GOAL_CELL = (6, 148)
ROTATE = True  # quarter turn anticlockwise, so 75x155 is drawn 155x75

DETAIL_EXPANSIONS = 3    # expansions drawn cell by cell; the rest play out at map scale
STRIDE = 8               # cells a straight scan reads per beat
DETAILED_DIAG_STEPS = 3  # diagonal steps drawn at full pace before the rest speed up
VIEW_MARGIN = 3.0        # cells of empty space kept around the detailed expansions
MAP_MARGIN = 4.0         # cells of empty space kept around the map itself
MAIN_CELL = 0.86         # fill size for a cell read by a scan, leaving the gridlines visible
PROBE_CELL = 0.5         # fill size for the scans a diagonal step fires off

SWEEP_BEAT = 0.34        # seconds an expansion gets once the camera is back at map scale
DETAIL_MARKER = 0.4 * CELL   # start/goal dot radius while zoomed in
MAP_MARKER = 1.5 * CELL      # start/goal dot radius at map scale
MAP_JUMP = 0.7 * CELL        # jump-point dot radius at map scale
DETAIL_JUMP = 0.32 * CELL    # jump-point dot radius while zoomed in
LABELLED_JUMP = 0.46 * CELL  # ... widened to hold an f-value, when those are printed

# f is what A* sorts the open list by: the cost of reaching the jump point plus the octile
# estimate from there to the goal. Printing it only makes sense while the camera is in
# close: at map scale a jump-point dot is a few pixels across and there are hundreds of
# them, so the sweep carries the value in the dot's colour alone.
SHOW_F_VALUES = True     # print each jump point's f-value inside its dot
COLOUR_JUMP_BY_F = True  # and colour every jump point by f, most promising to least
F_TOP_PERCENTILE = 95    # f at the far end of the colour ramp (see f_range)

STRAIGHT = "#F59F00"  # a straight scan, while it is live
DIAGONAL = "#7048E8"  # a diagonal step
PROBE = "#4DABF7"     # the straight scans a diagonal step fires before continuing
MISS = "#7A8595"      # a scan that died on an obstacle
JUMP = "#F08C00"      # jump point
FORCED = "#FFD43B"    # the neighbour that is forced
BLOCKER = "#FF6B6B"   # the obstacle doing the forcing
START = "#2F9E44"
GOAL = "#E03131"
PATH = "#1C7ED6"
GRID_LINE = "#6C7A89"
PANEL = "#0D1117"
# The f-value ramp, lowest f first: viridis, sampled at six even stops. Perceptually uniform,
# so equal steps in f look like equal steps in colour, and monotone in lightness, so the ramp
# still reads as an ordering in greyscale or to a colourblind viewer. It also runs dark at the
# promising end, which puts the strongest mark on the dots that matter most against the map.
# The top stop is pulled back from viridis's pure yellow, which all but vanishes on light cells.
F_STOPS = ("#440154", "#414487", "#2C728E", "#22A884", "#7AD151", "#DCE319")
INK = "#E9ECEF"       # only ever used on a dark plate: it is the colour of a free cell
MAP_INK = "#495867"   # annotation drawn straight onto the map, which is mostly light

NORTH, EAST, SOUTH, WEST = (0, -1), (1, 0), (0, 1), (-1, 0)
STRAIGHTS = (NORTH, EAST, SOUTH, WEST)
DIAGONALS = ((1, -1), (-1, -1), (1, 1), (-1, 1))
COMPONENTS = {
    (1, -1): (NORTH, EAST),
    (-1, -1): (NORTH, WEST),
    (1, 1): (SOUTH, EAST),
    (-1, 1): (SOUTH, WEST),
}
DIR_NAMES = {
    NORTH: "north", EAST: "east", SOUTH: "south", WEST: "west",
    (1, -1): "north-east", (-1, -1): "north-west", (1, 1): "south-east", (-1, 1): "south-west",
}


def screen_direction(direction):
    """Where a map direction ends up pointing once the picture has been turned.

    Anticlockwise, the map's x axis runs up the screen and its y axis runs right, so map
    east reads as north. The narration names what the viewer can see, not what the array
    indices say.
    """
    dx, dy = direction
    return (dy, -dx) if ROTATE else (dx, dy)


def direction_name(direction):
    return DIR_NAMES[screen_direction(direction)]


# ---------------------------------------------------------------------------
# f-values
# ---------------------------------------------------------------------------

def octile(a, b):
    """Cost of the cheapest obstacle-free run between two cells: Gridmap.distance, inlined.

    Used for both halves of f. Between an expanded node and one of its jump points the
    run is a straight line or a pure diagonal, so this is the exact edge cost; from a
    jump point to the goal it is the heuristic, the same one the search is using.
    """
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])
    return max(dx, dy) + (2 ** 0.5 - 1) * min(dx, dy)


def gradient_at(t):
    """Colour at t in [0, 1] along F_STOPS."""
    t = float(np.clip(t, 0.0, 1.0)) * (len(F_STOPS) - 1)
    lower = min(int(t), len(F_STOPS) - 2)
    return interpolate_color(
        ManimColor(F_STOPS[lower]), ManimColor(F_STOPS[lower + 1]), t - lower
    )


def promise_colour(f, low, high):
    """Where an f-value sits on the ramp, against the lowest and highest f in the search."""
    span = high - low
    return gradient_at(0.0 if span <= 0 else (f - low) / span)


def readable_ink(colour):
    """Panel dark or paper light, whichever holds up against the colour behind it."""
    r, g, b = ManimColor(colour).to_rgb()
    return PANEL if 0.299 * r + 0.587 * g + 0.114 * b > 0.55 else INK


# ---------------------------------------------------------------------------
# recording the scans
# ---------------------------------------------------------------------------

class StraightScan:
    """One straight scan: the cells it read, and what stopped it."""

    def __init__(self, origin, direction, cells, jump_point, forced, wall):
        self.origin = origin
        self.direction = direction
        self.cells = cells            # free cells read, in order, excluding the origin
        self.jump_point = jump_point  # cell the scan stopped on, or None if it died
        self.forced = forced          # (blocker, forced neighbour) pairs at the jump point
        self.wall = wall              # obstacle that stopped the scan, or None
        self.f = None                 # f of the jump point, filled in by record_f_values


class DiagonalStep:
    """One diagonal step, and the pair of straight scans fired from it."""

    def __init__(self, cell, probes, is_jump_point):
        self.cell = cell
        self.probes = probes
        self.is_jump_point = is_jump_point
        self.f = None


class DiagonalScan:
    def __init__(self, origin, direction, steps, jump_point):
        self.origin = origin
        self.direction = direction
        self.steps = steps
        self.jump_point = jump_point
        self.f = None


class Expansion:
    def __init__(self, state, arrival, scans, g=0.0):
        self.state = state
        self.arrival = arrival  # direction this node was reached from, None at the start
        self.scans = scans
        self.g = g              # cost of reaching this node: the g every f below starts from


def forced_neighbours(free, cell, direction):
    """(blocker, forced neighbour) pairs that make `cell` a jump point, entered along `direction`.

    Travelling north with an obstacle to the south-east, the cell to the east can only be
    reached optimally through here: that is the forced neighbour, and it is the whole
    reason this cell is worth putting on the open list.
    """
    x, y = cell
    dx, dy = direction
    pairs = []
    if dx == 0:
        for side in (1, -1):
            if not free[y - dy][x + side] and free[y][x + side]:
                pairs.append(((x + side, y - dy), (x + side, y)))
    else:
        for side in (1, -1):
            if not free[y + side][x - dx] and free[y + side][x]:
                pairs.append(((x - dx, y + side), (x, y + side)))
    return pairs


def diagonal_allowed(free, cell, direction):
    """A diagonal step may not cut a corner: both orthogonals and the target must be free."""
    x, y = cell
    dx, dy = direction
    return free[y + dy][x] and free[y][x + dx] and free[y + dy][x + dx]


def straight_scan(free, origin, direction, goal):
    x, y = origin
    dx, dy = direction
    cells = []
    while True:
        x, y = x + dx, y + dy
        if not free[y][x]:
            return StraightScan(origin, direction, cells, None, (), (x, y))
        cells.append((x, y))
        forced = forced_neighbours(free, (x, y), direction)
        if forced or (x, y) == goal:
            return StraightScan(origin, direction, cells, (x, y), tuple(forced), None)


def diagonal_scan(free, origin, direction, goal):
    steps = []
    cell = origin
    while diagonal_allowed(free, cell, direction):
        cell = (cell[0] + direction[0], cell[1] + direction[1])
        # Both component scans are recorded even though the expander stops at the first one
        # that hits: the pair of tests is the point of the diagonal step.
        probes = [straight_scan(free, cell, component, goal) for component in COMPONENTS[direction]]
        hit = cell == goal or any(probe.jump_point for probe in probes)
        steps.append(DiagonalStep(cell, probes, hit))
        if hit:
            return DiagonalScan(origin, direction, steps, cell)
    return DiagonalScan(origin, direction, steps, None)


def record_f_values(scans, origin, g, goal):
    """Give every jump point the f the search will sort it by, once it is on the open list.

    Only the cells that actually become nodes get one: a diagonal step's probes report a
    jump point too, but the node pushed is the step the probe fired from, never the cell
    the probe landed on.
    """
    for scan in scans:
        if isinstance(scan, DiagonalScan):
            for step in scan.steps:
                if step.is_jump_point:
                    step.f = g + octile(origin, step.cell) + octile(step.cell, goal)
            scan.f = scan.steps[-1].f if scan.jump_point else None
        elif scan.jump_point:
            scan.f = g + octile(origin, scan.jump_point) + octile(scan.jump_point, goal)


def jump_points_found(scan):
    """(cell, f) for every jump point a scan reported — at most one, but see the diagonals."""
    if isinstance(scan, DiagonalScan):
        return [(step.cell, step.f) for step in scan.steps if step.is_jump_point]
    return [(scan.jump_point, scan.f)] if scan.jump_point else []


def f_range(events, top=F_TOP_PERCENTILE):
    """The two ends of the colour ramp: lowest f in the search, and near enough the highest.

    Taken across every expansion rather than per beat, so a dot's colour means the same
    thing in the third expansion as it does in the eightieth. The top end is a percentile
    and not the maximum because a handful of dead-end jump points sit far above the rest
    (208 against a median of 154 here) and would push every other dot to the same end of
    the ramp. Anything above it is drawn at F_STOPS[-1], which is what it deserves.
    """
    values = [f for event in events for scan in event.scans for _, f in jump_points_found(scan)]
    if not values:
        return 0.0, 1.0
    low = min(values)
    return low, max(float(np.percentile(values, top)), low + 1e-9)


def successor_directions(free, state, arrival):
    """Natural + forced directions to scan from `state`, having arrived along `arrival`.

    The same set as JumpPointSearch.compute_successors, in the same order, minus any
    direction whose first step is an obstacle: the expander scans those and fails at once.
    """
    x, y = state
    dx, dy = arrival if arrival else (0, 0)
    if arrival is None:
        directions = list(STRAIGHTS) + list(DIAGONALS)
    elif dx and dy:
        directions = [(0, dy), (dx, 0), (dx, dy)]
    else:
        directions = [arrival]
        for side in (1, -1):
            if dx == 0 and not free[y - dy][x + side]:
                directions += [(side, 0), (side, dy)]
            elif dy == 0 and not free[y + side][x - dx]:
                directions += [(0, side), (dx, side)]
    return [d for d in directions if free[y + d[1]][x + d[0]]]


class TracingJPS(JumpPointSearch):
    """The library expander, plus a cell-by-cell record of every scan it implies."""

    def __init__(self, gridmap, heuristic, goal, free):
        super().__init__(gridmap, heuristic, goal)
        self.free = free
        self.events = []

    def expand(self, node):
        successors = super().expand(node)

        arrival = None if node.parent is None else step_direction(node.parent.state, node.state)
        scans = [
            (diagonal_scan if all(direction) else straight_scan)(
                self.free, node.state, direction, self.goal_.state
            )
            for direction in successor_directions(self.free, node.state, arrival)
        ]
        record_f_values(scans, node.state, node.g, self.goal_.state)
        self.events.append(Expansion(node.state, arrival, scans, node.g))
        return successors


def run_search(map_file=MAP_FILE, start_cell=START_CELL, goal_cell=GOAL_CELL):
    """The whole search, start to goal, with every scan the expander implied recorded."""
    map_path = os.path.join(PATHFINDING_DIR, map_file)

    gridmap = Gridmap()
    gridmap.load(map_path)
    free = load_map_array(map_path)

    heuristic = OctileDistanceHeuristic()
    goal_state = gridmap.cnvt_to_padded(goal_cell)
    expander = TracingJPS(gridmap, heuristic, Node(state=goal_state), free)
    astar = Astar(gridmap, heuristic, StandardTieBreaker(), expander)

    path = astar.search(gridmap.cnvt_to_padded(start_cell), goal_state)
    print(f"{start_cell} -> {goal_cell}: {len(expander.events)} expansions, {len(path)} jump points")
    return free, path, expander.events


def cells_read(scan):
    """Only the cells a scan actually stepped onto: what the expander paid for."""
    if isinstance(scan, DiagonalScan):
        return [
            cell
            for step in scan.steps
            for cell in [step.cell] + [c for probe in step.probes for c in cells_read(probe)]
        ]
    return list(scan.cells)


def scan_cells(scan):
    """Every cell a scan touches, including the obstacles it stopped against."""
    if isinstance(scan, DiagonalScan):
        cells = []
        for step in scan.steps:
            cells.append(step.cell)
            for probe in step.probes:
                cells.extend(scan_cells(probe))
        return cells

    cells = list(scan.cells)
    if scan.wall:
        cells.append(scan.wall)
    for blocker, neighbour in scan.forced:
        cells.extend((blocker, neighbour))
    return cells


def plural(count, word):
    return f"{count} {word}" if count == 1 else f"{count} {word}s"


def viewport(events, extra):
    """Cell bounds of everything the scene draws, padded by VIEW_MARGIN."""
    cells = list(extra)
    for event in events:
        cells.append(event.state)
        for scan in event.scans:
            cells.extend(scan_cells(scan))

    xs = [c[0] for c in cells]
    ys = [c[1] for c in cells]
    return (
        (min(xs) - VIEW_MARGIN, max(xs) + 1 + VIEW_MARGIN),
        (min(ys) - VIEW_MARGIN, max(ys) + 1 + VIEW_MARGIN),
    )


# ---------------------------------------------------------------------------
# the scene
# ---------------------------------------------------------------------------

class JumpPointSearchExample(MovingCameraScene):
    """Three expansions cell by cell, then the rest of the search across the whole map."""

    # -- small helpers -----------------------------------------------------

    def font(self, size):
        """Font size that renders like `size` would at the default camera width."""
        return size * self.view_width / config.frame_width

    def cell_square(self, cell, colour, opacity, scale=MAIN_CELL, z_index=3):
        return Square(
            side_length=CELL * scale,
            stroke_width=0,
            fill_color=colour,
            fill_opacity=opacity,
            z_index=z_index,
        ).move_to(self.grid.cell_point(cell))

    def cell_outline(self, cell, colour, opacity=0.0, z_index=7):
        return Square(
            side_length=CELL,
            stroke_color=colour,
            stroke_width=stroke_at(2.6, self.view_width),
            fill_color=colour,
            fill_opacity=opacity,
            z_index=z_index,
        ).move_to(self.grid.cell_point(cell))

    def jump_colour(self, f):
        """A jump point's colour: its place on the f ramp, or plain amber if that is off."""
        if not COLOUR_JUMP_BY_F or f is None:
            return JUMP
        return promise_colour(f, self.f_low, self.f_high)

    def jump_dot(self, cell, scale=1.0, hollow=False, f=None):
        """A jump point. Ringed in the panel colour so it reads on top of an amber scan.

        Hollow ones are the cells a diagonal step's probes found: those never become nodes,
        so they have no f and stay the plain jump colour.
        """
        radius = (LABELLED_JUMP if SHOW_F_VALUES and not hollow else DETAIL_JUMP) * scale
        if hollow:
            return Circle(
                radius=radius, color=JUMP,
                stroke_width=stroke_at(2.6, self.view_width), z_index=8,
            ).move_to(self.grid.cell_point(cell))
        dot = Dot(self.grid.cell_point(cell), radius=radius, color=self.jump_colour(f), z_index=8)
        return dot.set_stroke(PANEL, width=stroke_at(1.6, self.view_width), opacity=1)

    def jump_beat(self, cell, f, scale=1.0):
        """A jump point arriving: the dot to keep, and the animations that bring it in.

        The f-value is a mobject of its own rather than a child of the dot. The dots are
        scaled up on the way out to map scale and the labels are dropped there, and a
        child would have to be dragged through both.
        """
        dot = self.jump_dot(cell, scale=scale, f=f)
        animations = [GrowFromCenter(dot)]

        if SHOW_F_VALUES and f is not None:
            self.say_once("f-value", self.f_caption)
            label = Text(f"{f:.0f}", font_size=self.font(11),
                         color=readable_ink(dot.get_fill_color()))
            # Across the family, as in panelled(): the glyphs of a Text keep a z_index of
            # their own, and at 0 they would be drawn underneath the dot they sit in.
            label.set_z_index(9, family=True)
            if label.width > 0.72 * dot.width:
                label.scale_to_fit_width(0.72 * dot.width)
            label.move_to(self.grid.cell_point(cell))
            self.f_labels.add(label)
            animations.append(FadeIn(label))

        return dot, animations

    def panelled(self, content):
        """Content on a dark rounded plate, so it stays readable over map and scans alike.

        z-indices are set across the whole family: a plain Text inside a VGroup keeps its
        own z_index of 0 and would otherwise render underneath the plate.
        """
        pad = 0.01 * self.view_width
        content.set_z_index(12, family=True)
        backing = RoundedRectangle(
            corner_radius=0.7 * pad,
            width=content.width + 2 * pad,
            height=content.height + 1.7 * pad,
            stroke_width=0, fill_color=PANEL, fill_opacity=0.82,
        ).move_to(content)
        backing.set_z_index(11)
        return VGroup(backing, content)

    # -- framing -----------------------------------------------------------

    def framing(self, x_range, y_range):
        """Camera width and centre that hold a block of cells, whichever way the map is turned."""
        (x0, x1), (y0, y1) = x_range, y_range
        corners = self.grid.cell_points([
            (x0 - 0.5, y0 - 0.5), (x1 - 0.5, y0 - 0.5),
            (x0 - 0.5, y1 - 0.5), (x1 - 0.5, y1 - 0.5),
        ])
        low, high = corners.min(axis=0), corners.max(axis=0)
        span = high - low
        width = max(span[0], span[1] * config.frame_width / config.frame_height)
        return width, (low + high) / 2

    def reframe(self, width, centre, run_time=1.5, extra=()):
        """Move the camera, rescaling the HUD so it keeps the size it had on screen.

        The caption goes with it: at either scale the next thing said writes a fresh one,
        and animating a panel of text through a zoom is never worth the arithmetic.
        """
        ratio = width / self.view_width
        height = width * config.frame_height / config.frame_width
        half = np.array([width / 2 - 0.03 * width, height / 2 - 0.03 * width, 0.0])

        animations = [self.camera.frame.animate.set(width=width).move_to(centre), *extra]
        for panel, corner in self.hud:
            target = centre + corner * half
            if panel in self.mobjects:
                animations.append(panel.animate.scale(ratio).move_to(target, aligned_edge=corner))
            else:
                # A panel that is not on screen yet is moved outright, so that whenever it
                # is faded in it is already the right size for the frame it appears in.
                panel.scale(ratio).move_to(target, aligned_edge=corner)
        if len(self.caption):
            animations.append(FadeOut(self.caption))
            self.caption = VGroup()

        self.play(*animations, run_time=run_time)
        self.view_width = width
        self.caption_anchor = self.title.get_corner(DL) + DOWN * 0.3 * self.title.height
        self.caption_width = 0.58 * width

    # -- static furniture --------------------------------------------------

    def visible_cells(self, centre, width):
        """Cell-index bounds of the camera frame, padded by a cell on every side."""
        height = width * config.frame_height / config.frame_width
        cells = [
            self.grid.point_cell((centre[0] + sx * width / 2, centre[1] + sy * height / 2))
            for sx in (-1, 1) for sy in (-1, 1)
        ]
        xs = [c[0] for c in cells]
        ys = [c[1] for c in cells]
        return (min(xs) - 1, max(xs) + 1), (min(ys) - 1, max(ys) + 1)

    def gridlines(self, x_range, y_range):
        """Gridlines over the window, drawn above the fills so the steps stay countable."""
        (x0, x1), (y0, y1) = x_range, y_range
        x0, x1 = int(np.floor(x0)), int(np.ceil(x1))
        y0, y1 = int(np.floor(y0)), int(np.ceil(y1))
        width = stroke_at(1.0, self.view_width)

        lines = VGroup(z_index=6)
        for x in range(x0, x1 + 1):
            lines.add(Line(self.grid.cell_point((x - 0.5, y0 - 0.5)),
                           self.grid.cell_point((x - 0.5, y1 - 0.5))))
        for y in range(y0, y1 + 1):
            lines.add(Line(self.grid.cell_point((x0 - 0.5, y - 0.5)),
                           self.grid.cell_point((x1 - 0.5, y - 0.5))))
        lines.set_stroke(color=GRID_LINE, width=width, opacity=0.5)
        return lines

    def jump_swatch(self, size, steps=24):
        """The legend's jump-point mark: a dot, or the f ramp itself when dots are coloured by f.

        Built from a row of solid slices because a gradient across a single filled shape
        is a stroke-level trick in Manim and does not survive being dropped into a VGroup.
        """
        if not COLOUR_JUMP_BY_F:
            return Dot(radius=0.6 * size, color=JUMP)
        bar = VGroup(*[
            Rectangle(width=3.4 * size / steps, height=1.2 * size, stroke_width=0,
                      fill_color=gradient_at(index / (steps - 1)), fill_opacity=0.95)
            for index in range(steps)
        ])
        return bar.arrange(RIGHT, buff=0)

    def build_hud(self, frame):
        """Title and legend, pinned to the corners of the frame as it stands now.

        Both are kept in self.hud so reframe() can carry them through a zoom, and both are
        sized off self.view_width, so whichever scale they are built at is the one they
        keep on screen.
        """
        margin = 0.03 * self.view_width

        self.title = self.panelled(Text("Jump Point Search, one cell at a time",
                                        font_size=self.font(21), color=INK, weight=BOLD))
        self.title.move_to(frame.get_corner(UL) + RIGHT * margin + DOWN * margin, aligned_edge=UL)

        self.caption_anchor = self.title.get_corner(DL) + DOWN * 0.3 * self.title.height
        self.caption_width = 0.58 * self.view_width
        self.caption = VGroup()

        rows = VGroup()
        for colour, label in (
            (STRAIGHT, "straight scan"),
            (DIAGONAL, "diagonal step"),
            (PROBE, "scan fired from a diagonal step"),
            (MISS, "scan stopped by an obstacle"),
            (JUMP, "jump point, lowest f to highest" if COLOUR_JUMP_BY_F else "jump point"),
            (FORCED, "forced neighbour"),
            (BLOCKER, "the obstacle that forces it"),
        ):
            text = Text(label, font_size=self.font(14), color=INK)
            if colour == JUMP:
                swatch = self.jump_swatch(text.height)
            else:
                swatch = Square(side_length=text.height * 1.2, stroke_width=0,
                                fill_color=colour, fill_opacity=0.9)
            rows.add(VGroup(swatch, text.next_to(swatch, RIGHT, buff=0.5 * text.height)))
        rows.arrange(DOWN, aligned_edge=LEFT, buff=0.6 * rows[0].height)

        self.legend = self.panelled(rows)
        self.legend.move_to(frame.get_corner(UR) + LEFT * margin + DOWN * margin, aligned_edge=UR)
        self.hud = [(self.title, UL), (self.legend, UR)]

    # -- narration ---------------------------------------------------------

    def say(self, message, wait=0.0):
        text = Text(message, font_size=self.font(17), color=INK)
        if text.width > self.caption_width:
            text.scale_to_fit_width(self.caption_width)
        new = self.panelled(text)
        new.move_to(self.caption_anchor, aligned_edge=UL)

        if len(self.caption):
            self.play(FadeOut(self.caption), FadeIn(new), run_time=0.25)
        else:
            self.play(FadeIn(new), run_time=0.3)
        self.caption = new
        if wait:
            self.wait(wait)

    def say_once(self, key, message, wait=0.0):
        """Explain a mechanic the first time it happens, and never again."""
        if key in self.pending:
            self.pending.discard(key)
            self.say(message, wait)
            return True
        return False

    # -- scan drawing ------------------------------------------------------

    def scan_ray(self, start, end, colour, width, z_index):
        """The scan itself: a line out of the cell that fired it, over the cells it read."""
        line = Line(self.grid.cell_point(start), self.grid.cell_point(end), z_index=z_index)
        line.set_stroke(color=colour, width=stroke_at(width, self.view_width))
        return line

    def play_beats(self, *sequences):
        """Run several beat sequences in lockstep: beat n of each plays alongside the others.

        A beat is (animations, run_time). Sequences may be different lengths, which is the
        normal case for the pair of scans a diagonal step fires: one usually outruns the other.
        """
        for group in zip_longest(*sequences, fillvalue=None):
            live = [beat for beat in group if beat]
            animations = [animation for beat in live for animation in beat[0]]
            if animations:
                self.play(*animations, run_time=max(beat[1] for beat in live))

    def straight_beats(self, scan, colour, beat, scale=MAIN_CELL, opacity=0.5,
                       z_index=3, width=3.0):
        """A straight scan as beats: STRIDE cells and the length of ray covering them.

        The ray is one segment per beat rather than one line grown in stages, so it arrives
        exactly in step with the fills. Returns (fills, ray, beats) — the caller plays the
        beats, alone or against another scan's.
        """
        squares = VGroup(*[
            self.cell_square(cell, colour, opacity, scale, z_index) for cell in scan.cells
        ])
        ray = VGroup()
        beats = []
        anchor = scan.origin

        for start in range(0, len(squares), STRIDE):
            block = scan.cells[start : start + STRIDE]
            segment = self.scan_ray(anchor, block[-1], colour, width, z_index + 2)
            ray.add(segment)
            anchor = block[-1]
            beats.append((
                [LaggedStart(*[FadeIn(square, scale=0.55) for square in squares[start : start + STRIDE]],
                             lag_ratio=0.14),
                 Create(segment)],
                beat,
            ))
        return squares, ray, beats

    def kill_beat(self, scan, squares, ray, beat):
        """Grey out a scan that ran into an obstacle, and flag the obstacle."""
        wall = self.cell_outline(scan.wall, MISS, opacity=0.4)
        animations = [FadeIn(wall, scale=1.3)]
        if len(squares):
            animations.append(squares.animate.set_fill(MISS, opacity=0.4))
        if len(ray):
            animations.append(ray.animate.set_stroke(MISS, opacity=0.55))
        return VGroup(wall), (animations, beat)

    def forced_beat(self, scan, beat):
        """Outline the obstacle and the neighbour it forces. No beat if the scan found the goal."""
        marks = VGroup()
        for blocker, neighbour in scan.forced:
            marks.add(self.cell_outline(blocker, BLOCKER, opacity=0.3))
            marks.add(self.cell_outline(neighbour, FORCED, opacity=0.35))
        if not len(marks):
            return marks, None
        return marks, ([LaggedStart(*[FadeIn(m, scale=1.3) for m in marks], lag_ratio=0.15)], beat)

    def probe_beats(self, probe, pace, slow):
        """One of the two straight scans a diagonal step fires, start to finish, as beats."""
        drawn = VGroup()
        squares, ray, beats = self.straight_beats(
            probe, PROBE, beat=(0.2 if slow else 0.11) * pace,
            scale=PROBE_CELL, opacity=0.55, z_index=2, width=2.0,
        )
        drawn.add(squares, ray)

        if probe.jump_point is None:
            wall, beat = self.kill_beat(probe, squares, ray, beat=(0.2 if slow else 0.1) * pace)
            drawn.add(wall)
            beats.append(beat)
        else:
            marks, beat = self.forced_beat(probe, beat=0.3 * pace)
            drawn.add(marks)
            if beat:
                beats.append(beat)
            marker = self.jump_dot(probe.jump_point, scale=0.65, hollow=True)
            drawn.add(marker)
            beats.append(([GrowFromCenter(marker)], 0.25 * pace))

        return drawn, beats

    def play_diagonal(self, scan, pace, detail):
        """Walk a diagonal one cell at a time, scanning both straight directions at each step."""
        drawn = VGroup()
        found = VGroup()
        anchor = scan.origin

        for index, step in enumerate(scan.steps):
            slow = detail and index < DETAILED_DIAG_STEPS
            square = self.cell_square(step.cell, DIAGONAL, 0.5)
            leg = self.scan_ray(anchor, step.cell, DIAGONAL, 3.0, 5)
            anchor = step.cell
            drawn.add(square, leg)
            self.play(FadeIn(square, scale=0.55), Create(leg),
                      run_time=(0.3 if slow else 0.12) * pace)

            if index == 0:
                self.say_once("diagonal-probes",
                              "each diagonal step halts and scans both straight directions "
                              "before it moves on")

            # Both scans go out at once: the step waits on the pair, not on one and then the other.
            sequences = []
            for probe in step.probes:
                probe_drawn, beats = self.probe_beats(probe, pace, slow)
                drawn.add(probe_drawn)
                sequences.append(beats)
            self.play_beats(*sequences)

            if step.is_jump_point:
                self.say_once("diagonal-jump",
                              "a straight scan hit something, so the diagonal stops here: "
                              "this cell goes on the open list, not the cell the scan found")
                dot, appear = self.jump_beat(step.cell, step.f)
                self.play(square.animate.set_fill(JUMP, opacity=0.45),
                          *appear, run_time=0.3 * pace)
                found.add(dot)

        return drawn, found

    def play_expansion(self, event, pace=1.0, detail=False):
        """Draw every scan of one expansion.

        Returns the scan mobjects and the jump-point dots separately: the scans get dimmed
        at the end, the jump points are what survives.
        """
        drawn = VGroup()
        found = VGroup()

        for scan in event.scans:
            name = direction_name(scan.direction)

            if isinstance(scan, DiagonalScan):
                if detail:
                    self.say(f"scanning {name}")
                diagonal_drawn, diagonal_found = self.play_diagonal(scan, pace, detail)
                drawn.add(diagonal_drawn)
                found.add(*diagonal_found)
                continue

            stride_demo = detail and "stride" in self.pending and len(scan.cells) >= STRIDE
            if stride_demo:
                self.pending.discard("stride")
                self.say(f"scanning {name}: a straight scan reads a whole block of "
                         f"{STRIDE} cells at a time")
            elif detail:
                self.say(f"scanning {name}")

            squares, ray, beats = self.straight_beats(scan, STRAIGHT, beat=0.2 * pace)
            drawn.add(squares, ray)
            self.play_beats(beats)
            if stride_demo:
                self.show_stride(squares, scan.direction)

            if scan.jump_point is None:
                self.say_once("miss", "nothing forced along the way, so the scan dies "
                                      "at the obstacle and returns nothing")
                wall, beat = self.kill_beat(scan, squares, ray, beat=0.25 * pace)
                drawn.add(wall)
                self.play_beats([beat])
            else:
                self.say_once("forced", "an obstacle beside the scan forces a neighbour: "
                                        "the only optimal way in is through this cell")
                marks, beat = self.forced_beat(scan, beat=0.35 * pace)
                drawn.add(marks)
                if beat:
                    self.play_beats([beat])
                dot, appear = self.jump_beat(scan.jump_point, scan.f)
                self.play(*appear, run_time=0.3 * pace)
                found.add(dot)

        return drawn, found

    def show_stride(self, squares, direction):
        """Brace the first block of a scan: the run of cells a real implementation reads at once."""
        side = DOWN if screen_direction(direction)[1] == 0 else RIGHT
        brace = Brace(squares[:STRIDE], direction=side, color=MAP_INK, buff=0.2 * CELL)
        brace.set_z_index(9)
        label = self.panelled(Text(f"{STRIDE} cells", font_size=self.font(15), color=INK))
        label.next_to(brace, side, buff=0.15 * CELL)
        group = VGroup(brace, label)

        self.play(FadeIn(group), run_time=0.3)
        self.wait(1.1)
        self.play(FadeOut(group), run_time=0.3)

    # -- the rest of the search, at map scale ------------------------------

    def sweep_rays(self, event):
        """One expansion drawn small: a ray per scan, and a dot per jump point it found.

        No cell fills and no probes. A whole map's worth of expansions has to stay on
        screen, and at this scale a cell is a couple of pixels: the ray is the only part
        of an expansion that still reads.
        """
        origin = self.grid.cell_point(event.state)
        rays = VGroup()
        dots = VGroup()

        for scan in event.scans:
            diagonal = isinstance(scan, DiagonalScan)
            steps = scan.steps if diagonal else scan.cells
            if not steps:
                continue  # a diagonal that could not take its first step: nothing was read
            reached = steps[-1].cell if diagonal else steps[-1]
            hit = scan.jump_point is not None

            ray = Line(origin, self.grid.cell_point(reached), z_index=4)
            ray.set_stroke(
                color=(DIAGONAL if diagonal else STRAIGHT) if hit else MISS,
                width=stroke_at(3.2 if hit else 2.0, self.view_width),
                opacity=0.95 if hit else 0.45,
            )
            rays.add(ray)

            # No f-values printed here: a dot is a couple of pixels across at this scale,
            # and the ramp is doing that job instead.
            for cell, f in jump_points_found(scan):
                dot = Dot(self.grid.cell_point(cell), radius=MAP_JUMP,
                          color=self.jump_colour(f), z_index=8)
                dots.add(dot.set_stroke(PANEL, width=stroke_at(1.2, self.view_width), opacity=1))

        return rays, dots

    def play_sweep(self, events, goal_state, jump_points):
        """Every remaining expansion, one per beat, up to the scan that lands on the goal.

        Each expansion's rays are held on screen until the next one draws, which keeps the
        map legible over eighty-odd beats while still showing where the search just was.
        Captions are the only thing that interrupt, at a handful of fixed points.
        """
        milestones = {
            int(0.22 * len(events)): "every dot is a jump point waiting on the open list"
                                     + (", and the greener it is the lower its f-value"
                                        if COLOUR_JUMP_BY_F else ""),
            int(0.48 * len(events)): "a long straight run costs one scan, "
                                     "however many cells it crosses",
            int(0.76 * len(events)): "grey scans found nothing: that whole direction "
                                     "was read and thrown away",
        }
        live = VGroup()

        for index, event in enumerate(events):
            if index in milestones:
                self.say(milestones[index])

            rays, dots = self.sweep_rays(event)
            if not len(rays):
                continue

            appear = [AnimationGroup(*[Create(ray) for ray in rays], lag_ratio=0.12)]
            if len(dots):
                appear.append(LaggedStart(*[GrowFromCenter(dot) for dot in dots], lag_ratio=0.25))
            self.play(LaggedStart(*appear, lag_ratio=0.5),
                      *([FadeOut(live)] if len(live) else []), run_time=SWEEP_BEAT)

            live = rays
            jump_points.add(*dots)

            if any(scan.jump_point == goal_state for scan in event.scans):
                return live

        return live

    # -- construction ------------------------------------------------------

    def construct(self):
        self.pending = {"stride", "forced", "miss", "diagonal-probes", "diagonal-jump", "pruning"}

        free, path, events = run_search()
        detailed, rest = events[:DETAIL_EXPANSIONS], events[DETAIL_EXPANSIONS:]
        start_state, goal_state = path[0], path[-1]

        self.f_low, self.f_high = f_range(events)
        self.f_labels = VGroup()
        self.f_caption = ("the number in a jump point is its f-value: cost to reach it, "
                          "plus the estimate on to the goal"
                          + (", so the greener the better" if COLOUR_JUMP_BY_F else ""))
        if SHOW_F_VALUES:
            self.pending.add("f-value")

        self.grid = GridMapImage(free, rotate_ccw=ROTATE)
        rows, cols = free.shape
        map_width, map_centre = self.framing((-MAP_MARGIN, cols + MAP_MARGIN),
                                             (-MAP_MARGIN, rows + MAP_MARGIN))
        # The goal is a map away and deliberately left out: this framing is the start's
        # neighbourhood, which is all the detailed expansions ever touch.
        detail_width, detail_centre = self.framing(*viewport(detailed, [start_state]))

        frame = self.camera.frame
        self.view_width = map_width
        frame.set(width=map_width).move_to(map_centre)

        start_dot = Dot(self.grid.cell_point(start_state), radius=MAP_MARKER, color=START, z_index=11)
        goal_dot = Dot(self.grid.cell_point(goal_state), radius=MAP_MARKER, color=GOAL, z_index=11)
        self.build_hud(frame)

        # -- the whole map, then in to the start ---------------------------

        self.play(FadeIn(self.grid), run_time=1.2)
        self.play(FadeIn(self.title), GrowFromCenter(start_dot), GrowFromCenter(goal_dot),
                  run_time=0.9)
        self.say(f"{int(free.sum()):,} free cells between the start and the goal", wait=1.0)

        shrink = DETAIL_MARKER / MAP_MARKER
        self.reframe(detail_width, detail_centre, run_time=2.0,
                     extra=[start_dot.animate.scale(shrink), goal_dot.animate.scale(shrink)])

        lines = self.gridlines(*self.visible_cells(detail_centre, detail_width))
        self.play(Create(lines, lag_ratio=0.01), FadeIn(self.legend), run_time=1.2)
        self.say("from the start, JPS scans outwards in all eight directions", wait=0.7)

        # -- the first few expansions, cell by cell ------------------------

        scans = VGroup()
        jump_points = VGroup()
        for index, event in enumerate(detailed):
            if index:
                arrival = direction_name(event.arrival)
                left = plural(len(event.scans), "direction")
                said = self.say_once(
                    "pruning",
                    f"arriving {arrival}, everything behind is already covered: "
                    f"only {left} left to scan",
                    wait=0.5,
                )
                if not said:
                    self.say(f"arriving {arrival}: {left} left to scan")
                cursor = Circle(radius=0.7 * CELL, color=DIAGONAL,
                                stroke_width=stroke_at(3.0, self.view_width),
                                z_index=9).move_to(self.grid.cell_point(event.state))
                self.play(Create(cursor), run_time=0.3)
                scans.add(cursor)

            drawn, found = self.play_expansion(
                event, pace=1.0 if index == 0 else 0.55, detail=index == 0
            )
            scans.add(drawn)
            jump_points.add(*found)

        read = {cell for event in detailed for scan in event.scans for cell in cells_read(scan)}
        self.say(f"{plural(len(read), 'cell')} read, and only "
                 f"{plural(len(jump_points), 'jump point')} on the open list")
        self.play(scans.animate.set_opacity(0.25), run_time=0.9)
        # Each dot pulses in its own colour: with the ramp on, flashing them all one colour
        # would undo the only thing telling them apart.
        self.play(LaggedStart(*[Indicate(dot, color=dot.get_fill_color(), scale_factor=1.6)
                                for dot in jump_points], lag_ratio=0.12), run_time=1.2)
        self.wait(0.6)

        # -- back out, and run the search to the goal ----------------------

        # Cleared before the zoom rather than during it: reframe only rescales the panels
        # still on screen, and a panel cannot be faded and rescaled in the same beat. The
        # f-values go with them: three digits inside a map-scale dot is a smudge.
        self.play(FadeOut(lines), FadeOut(self.legend), FadeOut(self.f_labels), run_time=0.6)
        detail_jump = LABELLED_JUMP if SHOW_F_VALUES else DETAIL_JUMP
        self.reframe(map_width, map_centre, run_time=2.0, extra=[
            start_dot.animate.scale(1 / shrink), goal_dot.animate.scale(1 / shrink),
            # each dot about its own centre: scaling the group would move them apart
            *[dot.animate.scale(MAP_JUMP / detail_jump) for dot in jump_points],
        ])
        self.say(f"{plural(len(rest), 'expansion')} left, same rules, one beat each",
                 wait=0.6)

        live = self.play_sweep(rest, goal_state, jump_points)

        self.say("a scan runs the length of the corridor and lands on the goal")
        self.play(Flash(goal_dot, color=GOAL, flash_radius=3.5 * CELL, line_length=2.0 * CELL,
                        line_stroke_width=stroke_at(3.0, self.view_width)),
                  Indicate(goal_dot, color=GOAL, scale_factor=1.5), run_time=1.0)

        # -- the path -------------------------------------------------------

        path_line = VMobject(z_index=10)
        path_line.set_points_as_corners(self.grid.cell_points(path))
        path_line.set_stroke(color=PATH, width=stroke_at(4.0, self.view_width))

        self.play(FadeOut(live), jump_points.animate.set_opacity(0.2),
                  scans.animate.set_opacity(0.12), run_time=0.9)
        self.say(f"{plural(len(events), 'expansion')} in all, and the path through them "
                 f"turns {plural(len(path) - 2, 'time')}")
        self.play(Create(path_line), run_time=3.0, rate_func=linear)
        self.wait(2)
