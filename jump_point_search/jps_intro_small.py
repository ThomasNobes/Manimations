"""JPS on a grid small enough to follow by eye: every scan, every cell it reads, one camera.

The map is nineteen cells by thirteen and hand-authored below, so the whole search fits on
screen at a size where a cell is worth drawing. Nothing is ever cleared: every cell a scan
reads is filled in and the scan itself is drawn as a ray out of the cell that fired it, so
the shape of an expansion stays on the map and the nine expansions accumulate into the
picture of what the search actually touched.

The search itself is the library's — domains/jps.py expands, Astar drives. The scan rules
are re-derived here rather than reused because the expander only reports where a jump
landed, and this scene needs every cell it read on the way there.

Gridmap pads a loaded map with a ring of blocked cells, and every coordinate below is in
that padded space so a search state indexes straight into the free array. The ring is a
wall as far as the search is concerned but it is not part of the picture, which is why the
marks that land on an obstacle check in_map() first.
"""

from manim import *

import sys, os
import textwrap
import tempfile
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

from jps import CELL, load_map_array, step_direction, stroke_at


# The map. '@' blocks, 'S' starts, 'G' is the goal, and both of those are free cells.
# The two features that make the search worth watching: the staircase across the middle,
# which no diagonal can cut through, and the wall on the right, which reaches the bottom
# edge so the only way to the goal is over the top of it.
GRID = [
    "...................",
    ".@.................",
    ".@.................",
    ".@...@@............",
    ".......@...........",
    "........@..........",
    "........@.......@..",
    "................@..",
    "................@..",
    "....S...........@..",
    "................@..",
    "............@...@G.",
    "............@...@..",
]
WALL_CHAR, START_CHAR, GOAL_CHAR = "@", "S", "G"

STRIDE = 1               # cells a straight scan reads per beat: one, at this size
DETAILED_DIAG_STEPS = 2  # diagonal steps drawn at full pace before the rest speed up
CELL_FILL = 0.86         # side of a drawn cell, leaving the gridlines showing around it
PROBE_CELL = 0.5         # fill size for the scans a diagonal step fires off
MARGIN_LEFT, MARGIN_RIGHT = 1.0, 9.7  # empty cells beside the map: the legend lives right
MARGIN_TOP, MARGIN_BOTTOM = 3.4, 0.7  # ... and the title and caption live above it

DETAIL_JUMP = 0.32 * CELL    # jump-point dot radius
LABELLED_JUMP = 0.42 * CELL  # ... widened to hold an f-value, when those are printed
CAPTION_CHARS = 62           # characters a caption line is wrapped at: two lines, at most

# f is what A* sorts the open list by: the cost of reaching the jump point plus the octile
# estimate from there to the goal. On a map this size there is room to print it in the dot.
SHOW_F_VALUES = True     # print each jump point's f-value inside its dot
COLOUR_JUMP_BY_F = True  # and colour every jump point by f, most promising to least
F_TOP_PERCENTILE = 95    # f at the far end of the colour ramp (see f_range)

PAPER = "#FFFFFF"     # the background, and a free cell
GRID_LINE = "#CBD3C9"
WALL = "#6E7175"
START = "#8CE99A"
GOAL = "#FF8787"
STRAIGHT = "#F59F00"  # a straight scan, while it is live
DIAGONAL = "#7048E8"  # a diagonal step
PROBE = "#4DABF7"     # the straight scans a diagonal step fires before continuing
MISS = "#A5AEB8"      # a scan that found nothing
STOPPED = "#343A40"   # ring on the obstacle a scan died against
JUMP = "#F08C00"      # jump point
FORCED = "#FFD43B"    # the neighbour that is forced
BLOCKER = "#C92A2A"   # the obstacle doing the forcing
PATH = "#1C7ED6"
PANEL = "#0D1117"     # the plate every piece of text sits on
# The f-value ramp, lowest f first. Deliberately not START's green or GOAL's red: the two
# markers sit on the same map as the dots and a scale that borrowed their colours would
# read as "this jump point is the goal" rather than "this jump point is promising".
F_STOPS = ("#0CA678", "#F59F00", "#C2255C")
INK = "#E9ECEF"       # only ever used on a dark plate

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


def direction_name(direction):
    return DIR_NAMES[direction]


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
    thing in the second expansion as it does in the ninth. The top end is a percentile and
    not the maximum because a dead-end jump point can sit far above the rest and would push
    every other dot to the same end of the ramp. Anything above it is drawn at F_STOPS[-1],
    which is what it deserves.
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


def marked_cell(grid, mark):
    """Where a marker character sits in the hand-authored grid, in map coordinates."""
    for y, row in enumerate(grid):
        x = row.find(mark)
        if x >= 0:
            return (x, y)
    raise ValueError(f"no {mark!r} in the grid")


def write_map(grid, path):
    """The grid as a MovingAI .map file, which is the only thing Gridmap knows how to load.

    Going through a file rather than filling in Gridmap's arrays by hand keeps the search
    reading the map exactly the way it reads a benchmark one, markers stripped back to the
    free cells they are.
    """
    body = "\n".join(row.replace(START_CHAR, ".").replace(GOAL_CHAR, ".") for row in grid)
    with open(path, "w") as out:
        out.write(f"type octile\nheight {len(grid)}\nwidth {len(grid[0])}\nmap\n{body}\n")


def run_search(grid=GRID):
    """The whole search, start to goal, with every scan the expander implied recorded."""
    if len({len(row) for row in grid}) != 1:
        raise ValueError("grid rows are not all the same width")

    handle, map_path = tempfile.mkstemp(prefix="jps_small_", suffix=".map")
    os.close(handle)
    try:
        write_map(grid, map_path)
        gridmap = Gridmap()
        gridmap.load(map_path)
        free = load_map_array(map_path)
    finally:
        os.unlink(map_path)

    heuristic = OctileDistanceHeuristic()
    goal_state = gridmap.cnvt_to_padded(marked_cell(grid, GOAL_CHAR))
    expander = TracingJPS(gridmap, heuristic, Node(state=goal_state), free)
    astar = Astar(gridmap, heuristic, StandardTieBreaker(), expander)

    path = astar.search(gridmap.cnvt_to_padded(marked_cell(grid, START_CHAR)), goal_state)
    print(f"{len(grid[0])}x{len(grid)}: {len(expander.events)} expansions, {len(path)} jump points")
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


def plural(count, word):
    return f"{count} {word}" if count == 1 else f"{count} {word}s"


# ---------------------------------------------------------------------------
# the picture of the map
# ---------------------------------------------------------------------------

class SmallGrid:
    """The hand-authored map drawn cell by cell, plus cell <-> scene conversion.

    Same coordinate convention as jps.GridMapImage — the array is padded the way Gridmap
    pads it, and the map proper is centred on the origin — but at this size the cells are
    worth drawing individually rather than as one image: an obstacle is inset inside its
    cell, which is what leaves the gridlines readable underneath the scans.

    The pieces are handed out separately instead of as one group so the map can be built up
    on screen in the order a reader would draw it: rule the grid, drop in the obstacles,
    then mark the two ends.
    """

    def __init__(self, free, start, goal, cell=CELL):
        self.free = free
        self.cell = cell
        self.rows, self.cols = free.shape  # padded
        self.map_rows, self.map_cols = self.rows - 2, self.cols - 2

        self.lines = self.build_lines()
        self.walls = VGroup(*[
            self.filled(state, WALL)
            for state in self.map_states() if not free[state[1]][state[0]]
        ])
        self.walls.set_z_index(1)
        self.markers = VGroup(self.filled(start, START), self.filled(goal, GOAL))
        # Above the scans and above the jump-point dots: these two cells are the only ones
        # a viewer needs to be able to find at any point in the eighty seconds that follow.
        self.markers.set_z_index(9)

    def map_states(self):
        """Every cell of the map proper, in reading order, skipping the padding ring."""
        return [
            (x, y)
            for y in range(1, self.map_rows + 1)
            for x in range(1, self.map_cols + 1)
        ]

    def in_map(self, state):
        """Is this a cell of the map proper, rather than the padding ring around it?"""
        x, y = state
        return 1 <= x <= self.map_cols and 1 <= y <= self.map_rows

    def filled(self, state, colour):
        return Square(
            side_length=CELL_FILL * self.cell, stroke_width=0,
            fill_color=colour, fill_opacity=1.0,
        ).move_to(self.cell_point(state))

    def build_lines(self):
        """The gridlines, and with them the border: the outermost lines are the map's edge.

        Left at a nominal stroke width — the scene sets the real one once it knows how wide
        the camera ended up, which it cannot work out until this grid exists to ask.
        """
        lines = VGroup(z_index=6)
        for x in range(1, self.map_cols + 2):
            lines.add(Line(self.cell_point((x - 0.5, 0.5)),
                           self.cell_point((x - 0.5, self.map_rows + 0.5))))
        for y in range(1, self.map_rows + 2):
            lines.add(Line(self.cell_point((0.5, y - 0.5)),
                           self.cell_point((self.map_cols + 0.5, y - 0.5))))
        lines.set_stroke(color=GRID_LINE)
        return lines

    def cell_point(self, state):
        return self.cell_points([state])[0]

    def cell_points(self, states):
        arr = np.asarray(states, dtype=float)
        col, row = arr[:, 0], arr[:, 1]
        out = np.zeros((len(arr), 3))
        out[:, 0] = (col + 0.5 - self.cols / 2) * self.cell
        out[:, 1] = (self.rows / 2 - row - 0.5) * self.cell
        return out


# ---------------------------------------------------------------------------
# the scene
# ---------------------------------------------------------------------------

class JumpPointSearchIntro(MovingCameraScene):
    """Every expansion of one small search, cell by cell, from a single fixed camera."""

    # -- small helpers -----------------------------------------------------

    def font(self, size):
        """Font size that renders like `size` would at the default camera width."""
        return size * self.view_width / config.frame_width

    def cell_square(self, cell, colour, opacity, scale=CELL_FILL, z_index=3):
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

        The f-value is a mobject of its own rather than a child of the dot, so the dots can
        be dimmed at the end without dragging their labels along.
        """
        dot = self.jump_dot(cell, scale=scale, f=f)
        animations = [GrowFromCenter(dot)]

        if SHOW_F_VALUES and f is not None:
            self.say_once("f-value", self.f_caption)
            label = Text(f"{f:.0f}", font_size=self.font(12),
                         color=readable_ink(dot.get_fill_color()))
            # Across the family, as in panelled(): the glyphs of a Text keep a z_index of
            # their own, and at 0 they would be drawn underneath the dot they sit in.
            label.set_z_index(9, family=True)
            if label.width > 0.78 * dot.width:
                label.scale_to_fit_width(0.78 * dot.width)
            label.move_to(self.grid.cell_point(cell))
            self.f_labels.add(label)
            animations.append(FadeIn(label))

        return dot, animations

    def panelled(self, content):
        """Content on a dark rounded plate, so it stays readable over paper and scans alike.

        z-indices are set across the whole family: a plain Text inside a VGroup keeps its
        own z_index of 0 and would otherwise render underneath the plate.
        """
        pad = 0.008 * self.view_width
        content.set_z_index(12, family=True)
        backing = RoundedRectangle(
            corner_radius=0.7 * pad,
            width=content.width + 2 * pad,
            height=content.height + 1.7 * pad,
            stroke_width=0, fill_color=PANEL, fill_opacity=0.86,
        ).move_to(content)
        backing.set_z_index(11)
        return VGroup(backing, content)

    # -- framing -----------------------------------------------------------

    def framing(self, x_range, y_range):
        """Camera width and centre that hold a block of cells, in padded coordinates."""
        (x0, x1), (y0, y1) = x_range, y_range
        corners = self.grid.cell_points([
            (x0 - 0.5, y0 - 0.5), (x1 - 0.5, y0 - 0.5),
            (x0 - 0.5, y1 - 0.5), (x1 - 0.5, y1 - 0.5),
        ])
        low, high = corners.min(axis=0), corners.max(axis=0)
        span = high - low
        width = max(span[0], span[1] * config.frame_width / config.frame_height)
        return width, (low + high) / 2

    # -- static furniture --------------------------------------------------

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
        """Title and caption above the map, legend in the margin beside it.

        Everything is sized off self.view_width, which never changes: one camera, so a
        panel built here keeps the on-screen size it was built at.
        """
        margin = 0.025 * self.view_width

        self.title = self.panelled(Text("Jump Point Search, one cell at a time",
                                        font_size=self.font(20), color=INK, weight=BOLD))
        self.title.move_to(frame.get_corner(UL) + RIGHT * margin + DOWN * margin, aligned_edge=UL)

        self.caption_anchor = self.title.get_corner(DL) + DOWN * 0.35 * self.title.height
        self.caption_width = 0.55 * self.view_width
        self.caption = VGroup()

        rows = VGroup()
        for colour, label in (
            (START, "start"),
            (GOAL, "goal"),
            (WALL, "obstacle"),
            (STRAIGHT, "straight scan"),
            (DIAGONAL, "diagonal step"),
            (PROBE, "scan from a diagonal"),
            (MISS, "scan that found nothing"),
            (JUMP, "jump point: low f to high f" if COLOUR_JUMP_BY_F else "jump point"),
            (FORCED, "forced neighbour"),
            (BLOCKER, "the obstacle forcing it"),
        ):
            text = Text(label, font_size=self.font(13), color=INK)
            if colour == JUMP:
                swatch = self.jump_swatch(text.height)
            else:
                swatch = Square(side_length=text.height * 1.2, stroke_width=0,
                                fill_color=colour, fill_opacity=0.9)
            rows.add(VGroup(swatch, text.next_to(swatch, RIGHT, buff=0.5 * text.height)))
        rows.arrange(DOWN, aligned_edge=LEFT, buff=0.7 * rows[0].height)

        self.legend = self.panelled(rows)
        self.legend.move_to(frame.get_corner(UR) + LEFT * margin + DOWN * margin, aligned_edge=UR)

    # -- narration ---------------------------------------------------------

    def say(self, message, wait=0.0):
        text = Text(textwrap.fill(message, CAPTION_CHARS), font_size=self.font(16),
                    color=INK, line_spacing=0.6)
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

    def straight_beats(self, scan, colour, beat, scale=CELL_FILL, opacity=0.5,
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
        """Fade out a scan that found nothing, and ring the obstacle that stopped it.

        A scan that ran off the map gets no ring: the padding it stopped on is a wall to the
        search, but it is not a cell the picture draws.
        """
        marks = VGroup()
        animations = []
        if self.grid.in_map(scan.wall):
            marks.add(self.cell_outline(scan.wall, STOPPED))
            animations.append(FadeIn(marks[0], scale=1.3))
        if len(squares):
            animations.append(squares.animate.set_fill(MISS, opacity=0.5))
        if len(ray):
            animations.append(ray.animate.set_stroke(MISS, opacity=0.6))
        return marks, (animations, beat)

    def forced_beat(self, scan, beat):
        """Outline the obstacle and the neighbour it forces. No beat if the scan found the goal."""
        marks = VGroup()
        for blocker, neighbour in scan.forced:
            if self.grid.in_map(blocker):
                marks.add(self.cell_outline(blocker, BLOCKER, opacity=0.3))
            marks.add(self.cell_outline(neighbour, FORCED, opacity=0.35))
        if not len(marks):
            return marks, None
        return marks, ([LaggedStart(*[FadeIn(m, scale=1.3) for m in marks], lag_ratio=0.15)], beat)

    def probe_beats(self, probe, pace, slow):
        """One of the two straight scans a diagonal step fires, start to finish, as beats."""
        drawn = VGroup()
        squares, ray, beats = self.straight_beats(
            probe, PROBE, beat=(0.22 if slow else 0.13) * pace,
            scale=PROBE_CELL, opacity=0.55, z_index=2, width=2.0,
        )
        drawn.add(squares, ray)

        if probe.jump_point is None:
            marks, beat = self.kill_beat(probe, squares, ray, beat=(0.2 if slow else 0.12) * pace)
            drawn.add(marks)
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
                      run_time=(0.32 if slow else 0.14) * pace)

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

            if detail:
                self.say(f"scanning {name}")

            squares, ray, beats = self.straight_beats(scan, STRAIGHT, beat=0.2 * pace)
            drawn.add(squares, ray)
            self.play_beats(beats)

            if scan.jump_point is None:
                if self.grid.in_map(scan.wall):
                    self.say_once("miss", "nothing was forced along the way, so the scan "
                                          "dies at the obstacle and returns nothing")
                else:
                    self.say_once("edge", "the edge of the map stops a scan the same way an "
                                          "obstacle does, and returns nothing either")
                marks, beat = self.kill_beat(scan, squares, ray, beat=0.25 * pace)
                drawn.add(marks)
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

    # -- construction ------------------------------------------------------

    def construct(self):
        self.camera.background_color = PAPER
        self.pending = {"forced", "miss", "edge", "diagonal-probes", "diagonal-jump", "pruning"}

        free, path, events = run_search()
        start_state, goal_state = path[0], path[-1]

        self.f_low, self.f_high = f_range(events)
        self.f_labels = VGroup()
        self.f_caption = ("the number in a jump point is its f-value: cost to reach it, "
                          "plus the estimate on to the goal"
                          + (", so the greener the better" if COLOUR_JUMP_BY_F else ""))
        if SHOW_F_VALUES:
            self.pending.add("f-value")

        # The map proper is padded rows 1..rows-2 by padded cols 1..cols-2, and the framing
        # is that block plus the margins the title and legend live in.
        rows, cols = free.shape
        self.grid = SmallGrid(free, start_state, goal_state)
        self.view_width, centre = self.framing((1 - MARGIN_LEFT, cols - 1 + MARGIN_RIGHT),
                                               (1 - MARGIN_TOP, rows - 1 + MARGIN_BOTTOM))
        self.grid.lines.set_stroke(width=stroke_at(1.6, self.view_width))

        frame = self.camera.frame
        frame.set(width=self.view_width).move_to(centre)
        self.build_hud(frame)

        # -- draw the map --------------------------------------------------

        self.play(Create(self.grid.lines, lag_ratio=0.04), run_time=1.6)
        self.play(FadeIn(self.title),
                  LaggedStart(*[FadeIn(wall, scale=0.6) for wall in self.grid.walls],
                              lag_ratio=0.06),
                  run_time=1.4)
        self.play(LaggedStart(*[GrowFromCenter(marker) for marker in self.grid.markers],
                              lag_ratio=0.4), run_time=0.9)
        self.say(f"{cols - 2} by {rows - 2} cells, {plural(len(self.grid.walls), 'obstacle')}, "
                 f"and a goal walled off from the start", wait=1.2)
        self.play(FadeIn(self.legend), run_time=0.8)
        self.say("from the start, JPS scans outwards in all eight directions", wait=0.8)

        # -- every expansion, cell by cell ---------------------------------

        scans = VGroup()
        jump_points = VGroup()
        for index, event in enumerate(events):
            if index:
                arrival = direction_name(event.arrival)
                left = plural(len(event.scans), "direction")
                said = self.say_once(
                    "pruning",
                    f"arriving {arrival}, everything behind is already covered: "
                    f"only {left} left to scan",
                    wait=0.6,
                )
                if not said:
                    self.say(f"arriving {arrival}: {left} left to scan")
                cursor = Circle(radius=0.7 * CELL, color=DIAGONAL,
                                stroke_width=stroke_at(3.0, self.view_width),
                                z_index=9).move_to(self.grid.cell_point(event.state))
                self.play(Create(cursor), run_time=0.3)
                scans.add(cursor)

            drawn, found = self.play_expansion(
                event, pace=1.0 if index == 0 else 0.55, detail=True
            )
            scans.add(drawn)
            jump_points.add(*found)

        read = {cell for event in events for scan in event.scans for cell in cells_read(scan)}
        self.say(f"{plural(len(events), 'expansion')}, {plural(len(read), 'cell')} read of the "
                 f"{int(free.sum())} free, and {plural(len(jump_points), 'jump point')} in all")
        self.play(scans.animate.set_opacity(0.25), FadeOut(self.f_labels), run_time=0.9)
        # Each dot pulses in its own colour: with the ramp on, flashing them all one colour
        # would undo the only thing telling them apart.
        self.play(LaggedStart(*[Indicate(dot, color=dot.get_fill_color(), scale_factor=1.6)
                                for dot in jump_points], lag_ratio=0.1), run_time=1.4)

        self.say("the scan that reaches the goal ends the search")
        flash = Flash(self.grid.markers[1], color=GOAL, flash_radius=1.4 * CELL,
                      line_length=0.7 * CELL,
                      line_stroke_width=stroke_at(3.0, self.view_width))
        flash.lines.set_z_index(10)  # Flash builds its lines at z 0, under everything drawn
        self.play(flash, Indicate(self.grid.markers[1], color=GOAL, scale_factor=1.4),
                  run_time=1.0)

        # -- the path -------------------------------------------------------

        path_line = VMobject(z_index=10)
        path_line.set_points_as_corners(self.grid.cell_points(path))
        path_line.set_stroke(color=PATH, width=stroke_at(4.0, self.view_width))

        self.play(jump_points.animate.set_opacity(0.25),
                  scans.animate.set_opacity(0.12), run_time=0.9)
        self.say(f"{plural(len(path), 'jump point')} on the path, "
                 f"and it turns {plural(len(path) - 2, 'time')}")
        self.play(Create(path_line), run_time=3.0, rate_func=linear)
        self.wait(2)
