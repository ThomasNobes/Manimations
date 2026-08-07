"""JPS on a grid small enough to follow by eye: every scan drawn as an arrow, one camera.

The map is nineteen cells by thirteen and hand-authored below, so the whole search fits on
screen at a size where a cell is worth drawing. Nothing is ever cleared: every scan is an
arrow growing a cell at a time out of the cell that fired it, and once the scan resolves
the arrow settles into what it turned out to be — always its own colour, washed out towards
the paper and dashed if it found nothing, darkened and left solid if it produced a jump
point, so a dead arrow still says which kind of scan drew it. The shape of an
expansion stays on the map, and the nine expansions accumulate into a picture of what the
search actually touched, with every arrow saying whether it paid for itself.

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

from example_maps import *
GRID = CENTRAL_START_GRID  # select which map to use for anim
GRID = TOOTHCOMB_GRID  # select which map to use for anim
GRID = STAIRCASE_GRID  # select which map to use for anim
GRID = CHECKERBOARD_GRID  # select which map to use for anim

START_CHAR, GOAL_CHAR = "S", "G"  # anything else that is not a '.' blocks, per MovingAI
STRIDE = 1               # cells a straight scan reads per beat: one, at this size
DETAILED_DIAG_STEPS = 2  # diagonal steps drawn at full pace before the rest speed up
CELL_FILL = 1.0          # side of a map cell — obstacle, start, goal: the whole cell
MARK_FILL = 0.86         # ... and of a highlight drawn on one, which stays inside the lines
MARKER_LETTER = 0.5      # cap height of the S and the G, as a fraction of their cell
MARK_POP = 1.15          # how far a highlight overshoots as it lands: MARK_FILL * this <= 1
ARROW_TIP = 0.42 * CELL  # length of an arrowhead, at map scale
ARROW_DASH = 0.2 * CELL  # dash length on the arrows that found nothing
# Empty cells kept around the map: the legend lives in the right margin, the title and the
# caption in the one above. Spelled out rather than LEFT/RIGHT/TOP/BOTTOM, which are all
# Manim direction vectors that `from manim import *` has already put in this namespace.
MARGIN_LEFT, MARGIN_RIGHT = 1.0, 9.7
MARGIN_TOP, MARGIN_BOTTOM = 3.4, 0.7

DETAIL_JUMP = 0.32 * CELL    # jump-point dot radius
LABELLED_JUMP = 0.42 * CELL  # ... widened to hold an f-value, when those are printed
RING_GAP = 0.03 * CELL       # air between a jump-point dot and the ring drawn around it
RING_LIMIT = 0.45 * CELL     # ... and how far out that ring is allowed to get: inside the cell
PATH_LABEL_GAP = 0.1 * CELL  # air left between the path line and the S or the G it runs into
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
# What a scan settles into when it found nothing: its own colour washed this far towards the
# paper, and dashed. Fading rather than greying keeps a dead scan legible as the kind of scan
# it was — a cyan probe stays cyan — while the drop in contrast still reads as "this bought
# nothing" beside the darkened solid arrows that did.
MISS_FADE = 0.55
# What a scan settles into once it has paid off: its own colour, darkened, and left solid.
# Keyed by the live colour so a scan can be resolved without being told what kind it was.
HIT = {
    STRAIGHT: "#A85D00",
    DIAGONAL: "#3B23A0",
    PROBE: "#15629B",
}
JUMP = "#F08C00"      # jump point
FORCED = "#FFD43B"    # the neighbour that is forced
BLOCKER = "#C92A2A"   # the obstacle doing the forcing
PATH = "#1C7ED6"
PANEL = "#0D1117"     # the plate every piece of text sits on
# The f-value ramp, lowest f first: viridis, sampled at six even stops. Perceptually uniform,
# so equal steps in f look like equal steps in colour, and monotone in lightness, so the ramp
# still reads as an ordering in greyscale or to a colourblind viewer. It also runs dark at the
# promising end, which puts the strongest mark on the dots that matter most against PAPER.
# The top stop is pulled back from viridis's pure yellow, which all but vanishes on white.
# F_STOPS = ("#440154", "#414487", "#2C728E", "#22A884", "#7AD151", "#DCE319")
F_STOPS = ("#DCE319", "#7AD151", "#22A884", "#2C728E", "#414487", "#440154")
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


def faded(colour, amount=MISS_FADE):
    """A scan's colour washed towards the paper: what it settles into when it found nothing.

    Blended rather than made transparent, so a dead scan looks the same whether it happens
    to lie over bare paper or across an arrow drawn by an earlier expansion, and so the
    whole-group fade at the end of the scene still has its opacity to spend.
    """
    return interpolate_color(ManimColor(colour), ManimColor(PAPER), amount)


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


def evenly_sampled(corners, samples=240):
    """The same polyline, re-cornered at points spaced evenly along its length.

    Create hands each bezier in a mobject the same slice of the run time, so a polyline
    built straight out of its corners is drawn a segment at a time regardless of how long
    the segments are: a single diagonal step between two jump points takes as long to draw
    as a run of six cells, and the line visibly stalls on the turns. Cutting every segment
    into pieces of roughly equal length makes equal-time-per-piece equal-speed-per-pixel.
    The corners themselves stay in the list so nothing is rounded off.
    """
    corners = [np.asarray(c, dtype=float) for c in corners]
    lengths = [np.linalg.norm(b - a) for a, b in zip(corners, corners[1:])]
    total = sum(lengths)
    if total == 0:
        return corners
    step = total / samples
    points = [corners[0]]
    for start, end, length in zip(corners, corners[1:], lengths):
        pieces = max(1, round(length / step))
        for i in range(1, pieces + 1):
            points.append(start + (end - start) * (i / pieces))
    return points


# ---------------------------------------------------------------------------
# the picture of the map
# ---------------------------------------------------------------------------

class SmallGrid:
    """The hand-authored map drawn cell by cell, plus cell <-> scene conversion.

    Same coordinate convention as jps.GridMapImage — the array is padded the way Gridmap
    pads it, and the map proper is centred on the origin — but at this size the cells are
    worth drawing individually rather than as one image: an obstacle fills its cell edge
    to edge, and the gridlines are ruled back over the top so the lattice still reads.

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
        # The two ends, each a filled cell lettered with what it is. The squares are kept in
        # a group of their own as well, since they are the only part of a marker that takes
        # a gridline stroke — see rule().
        self.marker_cells = VGroup(self.filled(start, START), self.filled(goal, GOAL))
        self.markers = VGroup(
            VGroup(self.marker_cells[0], self.marker_label("S", start, START)),
            VGroup(self.marker_cells[1], self.marker_label("G", goal, GOAL)),
        )
        # Above the scans and above the jump-point dots: these two cells are the only ones
        # a viewer needs to be able to find at any point in the eighty seconds that follow.
        self.markers.set_z_index(9)
        # And the letters above the lattice, which is ruled over everything else: a gridline
        # crossing a glyph would only make it harder to read.
        for _, label in self.markers:
            label.set_z_index(11, family=True)

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
            side_length=CELL_FILL * self.cell, stroke_width=0, stroke_color=GRID_LINE,
            fill_color=colour, fill_opacity=1.0,
        ).move_to(self.cell_point(state))

    def marker_label(self, letter, state, colour):
        """The S or the G, sat in the middle of its cell.

        Sized off the cell rather than through the scene's font(): the letter has to hold its
        proportion to the square it names whatever the camera ends up at, and the grid is
        built before the scene knows that width anyway. Inked dark or light against the fill
        underneath, the same test the f-values in the jump-point dots go through.
        """
        label = Text(letter, font_size=48, weight=BOLD, color=readable_ink(colour))
        label.scale_to_fit_height(MARKER_LETTER * self.cell)
        return label.move_to(self.cell_point(state))

    def rule(self, width):
        """Draw the lattice at `width`: the gridlines, and the edge of every filled cell.

        A filled cell carries a border of its own in the gridline colour, exactly under the
        line that is drawn over the top of it. Belt and braces: a hairline is antialiased
        slightly differently depending on whether Manim redrew it for an animation or took
        it from its cached static frame, and doubling it up keeps the lattice the same
        weight over an obstacle as it is over paper. Called from the scene, which is the
        first thing to know the camera width every stroke here is scaled against.
        """
        self.lines.set_stroke(width=width)
        # The marker squares, not the whole markers: a stroke laid on a Text outlines every
        # glyph in it, and at this size that alone is enough to blur the letter.
        for group in (self.walls, self.marker_cells):
            group.set_stroke(color=GRID_LINE, width=width)

    def build_lines(self):
        """The gridlines, and with them the border: the outermost lines are the map's edge.

        Ruled over everything drawn on the map, obstacles and markers included, since those
        now fill their cells completely and would otherwise swallow the lattice wherever
        two of them touch. Left at a nominal stroke width — the scene sets the real one once
        it knows how wide the camera ended up, which it cannot work out until this grid
        exists to ask.
        """
        lines = VGroup()
        for x in range(1, self.map_cols + 2):
            lines.add(Line(self.cell_point((x - 0.5, 0.5)),
                           self.cell_point((x - 0.5, self.map_rows + 0.5))))
        for y in range(1, self.map_rows + 2):
            lines.add(Line(self.cell_point((0.5, y - 0.5)),
                           self.cell_point((self.map_cols + 0.5, y - 0.5))))
        lines.set_stroke(color=GRID_LINE)
        # On the whole family, not the group: a z_index handed to VGroup() sits on the
        # container and leaves every line in it at zero, which is under the obstacles. The
        # renderer only sometimes honours that — a mobject caught up in an animation is
        # painted over the static background whatever its z — and the lattice flickers.
        lines.set_z_index(10, family=True)
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

class JumpPointSearchIntroArrows(MovingCameraScene):
    """Every expansion of one small search, cell by cell, from a single fixed camera."""

    # -- small helpers -----------------------------------------------------

    def font(self, size):
        """Font size that renders like `size` would at the default camera width."""
        return size * self.view_width / config.frame_width

    def cell_outline(self, cell, colour, opacity=0.0, z_index=7):
        """A highlight on one cell: inset, so its stroke never crosses into the neighbours.

        A cell-sized square would carry half its stroke outside the cell, and now that the
        obstacles fill their cells and the gridlines are ruled on top there is nothing to
        hide that overhang.
        """
        return Square(
            side_length=MARK_FILL * CELL,
            stroke_color=colour,
            stroke_width=stroke_at(2.6, self.view_width),
            fill_color=colour,
            fill_opacity=opacity,
            z_index=z_index,
        ).move_to(self.grid.cell_point(cell))

    def node_ring(self, cell, colour, z_index=9):
        """The mark on a node coming off the open list: a ring traced around its dot.

        Every node on the map is drawn as a dot, so the cursor follows that shape rather
        than boxing it — a circle just outside the dot's edge, drawn heavier than the
        outlines used for forced neighbours so it reads as the thing being expanded. The
        radius is capped short of the cell's own half-width, since the gridlines are ruled
        over the top and a ring reaching past them would look like it belonged to the
        neighbouring cell as much as to this one.
        """
        radius = min((LABELLED_JUMP if SHOW_F_VALUES else DETAIL_JUMP) + RING_GAP, RING_LIMIT)
        return Circle(
            radius=radius,
            color=colour,
            stroke_width=stroke_at(4.0, self.view_width),
            z_index=z_index,
        ).move_to(self.grid.cell_point(cell))

    def clear_of_label(self, point, towards, label):
        """`point` pushed out along the line to `towards`, far enough to clear a marker letter.

        The S and the G sit in the middle of the two cells the path ends on, so a line run
        to the cell centre strikes straight through the glyph. The endpoint is moved to
        where the line leaves a box around the letter instead: a straight approach clears
        the glyph's half-width or half-height, a diagonal one clears the side it actually
        crosses. Worked in scene units off the letter's own box rather than in cells, so
        the gap left is the same air whichever way the path comes in — shifting by a fixed
        number of cells would leave a diagonal ending half again as far out as a straight one.
        """
        step = np.asarray(towards, dtype=float) - np.asarray(point, dtype=float)
        length = float(np.linalg.norm(step))
        if length == 0:
            return point
        direction = step / length
        half = (label.width / 2 + PATH_LABEL_GAP, label.height / 2 + PATH_LABEL_GAP)
        reach = min(
            half[axis] / abs(direction[axis]) if abs(direction[axis]) > 1e-9 else np.inf
            for axis in (0, 1)
        )
        # Never past the next corner: on a one-cell first or last leg the line still has to
        # keep the direction it turns in.
        return point + direction * min(reach, 0.8 * length)

    def arrow_point(self, origin, cell):
        """Where an arrow from `origin` aimed at `cell` should end.

        The centre of the cell, everywhere but the goal. The goal is a filled square drawn
        over the scans, so the one arrow that reaches it would vanish head and all inside
        it; that arrow stops on the near edge of the square instead, pulled back along its
        own direction to whichever side of the cell it is coming in through.
        """
        point = self.grid.cell_point(cell)
        if cell != self.goal_state:
            return point
        step = point - self.grid.cell_point(origin)
        reach = max(abs(step[0]), abs(step[1]))
        if reach == 0:
            return point
        return point - step * (0.5 * CELL / reach)

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
        content.set_z_index(14, family=True)
        backing = RoundedRectangle(
            corner_radius=0.7 * pad,
            width=content.width + 2 * pad,
            height=content.height + 1.7 * pad,
            stroke_width=0, fill_color=PANEL, fill_opacity=0.86,
        ).move_to(content)
        backing.set_z_index(13)
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

    def arrow_swatch(self, size, colour, dashed=False):
        """The legend's mark for a scan: a short arrow, in the same build as the real ones."""
        arrow = self.scan_arrow(LEFT * 1.5 * size, RIGHT * 1.5 * size, colour,
                                width=3.4, dashed=dashed, tip_length=1.4 * size)
        return arrow.set_z_index(14, family=True)

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
        for kind, colour, label in (
            ("cell", START, "start"),
            ("cell", GOAL, "goal"),
            ("cell", WALL, "obstacle"),
            ("arrow", STRAIGHT, "straight scan"),
            ("arrow", DIAGONAL, "diagonal step"),
            ("arrow", PROBE, "scan from a diagonal"),
            # The two outcomes are shown on the straight scan's colour: every kind of scan
            # fades its own colour when it found nothing and darkens it when it did not, so
            # one pair of swatches says what the pattern is without repeating it three times.
            ("dashed", faded(STRAIGHT), "found nothing"),
            ("arrow", HIT[STRAIGHT], "found a jump point"),
            ("ramp", JUMP, "jump point: low f to high f" if COLOUR_JUMP_BY_F else "jump point"),
            ("cell", FORCED, "forced neighbour"),
            ("cell", BLOCKER, "the obstacle forcing it"),
        ):
            text = Text(label, font_size=self.font(13), color=INK)
            if kind == "ramp":
                swatch = self.jump_swatch(text.height)
            elif kind == "cell":
                swatch = Square(side_length=text.height * 1.2, stroke_width=0,
                                fill_color=colour, fill_opacity=0.9)
            else:
                swatch = self.arrow_swatch(text.height, colour, dashed=(kind == "dashed"))
            rows.add(VGroup(swatch, text.next_to(swatch, RIGHT, buff=0.5 * text.height)))
        # Swatches are different widths now, so the labels are aligned to a shared column
        # rather than each one hung off its own swatch.
        column = max(row[0].width for row in rows)
        for row in rows:
            row[1].next_to(row[0].get_left() + RIGHT * column, RIGHT,
                           buff=0.5 * row[1].height)
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

    def arrow_head(self, start_point, end_point, colour, width, tip_length=None):
        """The head alone, pointing start -> end, and the point the shaft has to stop at.

        Manim only grows an arrowhead as part of a Line, and a dashed shaft is not a Line
        underneath, so the head is built on a throwaway line and then lifted off it. The
        line shortens itself to make room for the head when it is added, which is where the
        second return value comes from.
        """
        length = float(np.linalg.norm(np.asarray(end_point) - np.asarray(start_point)))
        if length == 0:
            return None, end_point
        tip_length = min(ARROW_TIP if tip_length is None else tip_length, 0.62 * length)

        helper = Line(start_point, end_point)
        helper.set_stroke(width=stroke_at(width, self.view_width))
        helper.add_tip(tip_length=tip_length, tip_width=0.9 * tip_length)
        head = helper.tip
        helper.remove(head)
        head.set_fill(colour, opacity=1).set_stroke(colour, width=0, opacity=0)
        return head, helper.get_end()

    def scan_arrow(self, start_point, end_point, colour, width, dashed=False,
                   z_index=5, tip_length=None):
        """A scan as an arrow: (shaft, head), kept as a pair so the shaft can be dashed."""
        head, shaft_end = self.arrow_head(start_point, end_point, colour, width, tip_length)
        if head is None:
            return VGroup()
        if dashed:
            length = float(np.linalg.norm(np.asarray(shaft_end) - np.asarray(start_point)))
            # Capped against the shaft's own length as well as ARROW_DASH, so a one-cell
            # scan still reads as dashed rather than as a solid stub with a gap in it.
            shaft = DashedLine(start_point, shaft_end, dashed_ratio=0.55,
                               dash_length=min(ARROW_DASH, max(length / 4.0, 1e-3)))
        else:
            shaft = Line(start_point, shaft_end)
        shaft.set_stroke(color=colour, width=stroke_at(width, self.view_width))
        return VGroup(shaft, head).set_z_index(z_index)

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

    def grow_beats(self, origin, cells, colour, beat, z_index=5, width=3.4):
        """A scan as beats: the arrow lengthens by STRIDE cells at a time, head leading.

        Rebuilt and transformed rather than extended, because the head has to stay at the
        front: each beat is the same arrow made longer. Every beat is linear, so a scan
        several cells long runs at one steady speed instead of easing in and out of a stop
        at each cell. Returns (arrow, beats) — the caller plays the beats, alone or against
        another scan's — and an empty arrow when the very first cell was an obstacle, which
        is a scan with nothing to point at.
        """
        if not cells:
            return VGroup(), []

        start = self.grid.cell_point(origin)
        tips = [cells[min(index + STRIDE, len(cells)) - 1]
                for index in range(0, len(cells), STRIDE)]

        arrow = self.scan_arrow(start, self.arrow_point(origin, tips[0]), colour, width,
                                z_index=z_index)
        beats = [([GrowFromPoint(arrow, start, rate_func=linear)], beat)]
        for tip in tips[1:]:
            longer = self.scan_arrow(start, self.arrow_point(origin, tip), colour, width,
                                     z_index=z_index)
            beats.append(([Transform(arrow, longer, rate_func=linear)], beat))
        return arrow, beats

    def settle_arrow(self, arrow, origin, last_cell, colour, z_index=5, width=3.4,
                     dashed=False):
        """Swap a live scan arrow for what it turned out to be. Returns (keep, animations).

        A hit is a Transform, because dark-solid is the same shape as the live arrow and
        the head should not so much as flicker. A miss is a crossfade: a dashed shaft is a
        row of separate dashes underneath, and there is nothing sane to morph it out of.
        """
        if not len(arrow):
            return arrow, []
        final = self.scan_arrow(self.grid.cell_point(origin), self.arrow_point(origin, last_cell),
                                colour, width, dashed=dashed, z_index=z_index)
        if dashed:
            return final, [FadeOut(arrow), FadeIn(final)]
        return arrow, [Transform(arrow, final)]

    def kill_beat(self, scan, arrow, colour, beat, z_index=5, width=3.4):
        """A scan that found nothing: fade the arrow out to dashes. Returns (keep, beat).

        `colour` is the scan's live colour, which it keeps a washed-out version of rather
        than going grey, so the map still says which kind of scan died where.

        The obstacle that stopped it is left unmarked — the dashed arrow already ends on it,
        and an outline on every wall the search ever touched only clutters the map.
        """
        kept, settle = self.settle_arrow(arrow, scan.origin, scan.cells[-1] if scan.cells else None,
                                         faded(colour), z_index=z_index, width=width, dashed=True)
        return kept, (settle, beat)

    def forced_beat(self, scan, beat):
        """Outline the obstacle and the neighbour it forces. No beat if the scan found the goal."""
        marks = VGroup()
        for blocker, neighbour in scan.forced:
            if self.grid.in_map(blocker):
                marks.add(self.cell_outline(blocker, BLOCKER, opacity=0.3))
            marks.add(self.cell_outline(neighbour, FORCED, opacity=0.35))
        if not len(marks):
            return marks, None
        return marks, ([LaggedStart(*[FadeIn(m, scale=MARK_POP) for m in marks],
                                    lag_ratio=0.15)], beat)

    def probe_beats(self, probe, pace, slow):
        """One of the two straight scans a diagonal step fires, start to finish, as beats.

        Drawn thinner and with a smaller head than a scan the search fired directly: these
        are the diagonal's own bookkeeping, and there are two of them at every step.
        """
        drawn = VGroup()
        arrow, beats = self.grow_beats(
            probe.origin, probe.cells, PROBE,
            beat=(0.22 if slow else 0.13) * pace, z_index=4, width=2.2,
        )
        drawn.add(arrow)

        if probe.jump_point is None:
            kept, beat = self.kill_beat(
                probe, arrow, PROBE, beat=(0.2 if slow else 0.12) * pace, z_index=4, width=2.2)
            drawn.add(kept)
            beats.append(beat)
        else:
            marks, beat = self.forced_beat(probe, beat=0.3 * pace)
            drawn.add(marks)
            _, settle = self.settle_arrow(arrow, probe.origin, probe.jump_point, HIT[PROBE],
                                          z_index=4, width=2.2)
            if beat:
                beats.append((beat[0] + settle, beat[1]))
            elif settle:
                beats.append((settle, 0.2 * pace))
            marker = self.jump_dot(probe.jump_point, scale=0.65, hollow=True)
            drawn.add(marker)
            beats.append(([GrowFromCenter(marker)], 0.25 * pace))

        return drawn, beats

    def play_diagonal(self, scan, pace, detail):
        """Walk a diagonal one cell at a time, scanning both straight directions at each step.

        One arrow for the whole diagonal, lengthened a cell per step, rather than one arrow
        per step: the head is what says where the walk has got to, and there is only ever
        one of it.
        """
        drawn = VGroup()
        found = VGroup()
        origin_point = self.grid.cell_point(scan.origin)
        arrow = VGroup()

        for index, step in enumerate(scan.steps):
            slow = detail and index < DETAILED_DIAG_STEPS
            longer = self.scan_arrow(origin_point, self.arrow_point(scan.origin, step.cell),
                                     DIAGONAL, 3.4, z_index=5)
            if index == 0:
                arrow = longer
                drawn.add(arrow)
                grow = GrowFromPoint(arrow, origin_point, rate_func=linear)
            else:
                grow = Transform(arrow, longer, rate_func=linear)
            self.play(grow, run_time=(0.32 if slow else 0.14) * pace)

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
                _, settle = self.settle_arrow(arrow, scan.origin, step.cell, HIT[DIAGONAL])
                self.play(*settle, *appear, run_time=0.3 * pace)
                found.add(dot)

        # A diagonal that walked itself out without ever stopping bought nothing, and goes
        # dashed and washed out like any other scan that found nothing.
        if scan.jump_point is None and len(arrow):
            kept, settle = self.settle_arrow(arrow, scan.origin, scan.steps[-1].cell,
                                             faded(DIAGONAL), dashed=True)
            drawn.add(kept)
            self.play(*settle, run_time=0.25 * pace)

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

            arrow, beats = self.grow_beats(scan.origin, scan.cells, STRAIGHT, beat=0.2 * pace)
            drawn.add(arrow)
            self.play_beats(beats)

            if scan.jump_point is None:
                if self.grid.in_map(scan.wall):
                    self.say_once("miss", "nothing was forced along the way, so the scan "
                                          "dies at the obstacle and returns nothing")
                else:
                    self.say_once("edge", "the edge of the map stops a scan the same way an "
                                          "obstacle does, and returns nothing either")
                kept, beat = self.kill_beat(scan, arrow, STRAIGHT, beat=0.25 * pace)
                drawn.add(kept)
                self.play_beats([beat])
            else:
                self.say_once("forced", "an obstacle beside the scan forces a neighbour: "
                                        "the only optimal way in is through this cell")
                marks, beat = self.forced_beat(scan, beat=0.35 * pace)
                drawn.add(marks)
                if beat:
                    self.play_beats([beat])
                dot, appear = self.jump_beat(scan.jump_point, scan.f)
                _, settle = self.settle_arrow(arrow, scan.origin, scan.jump_point, HIT[STRAIGHT])
                self.play(*settle, *appear, run_time=0.3 * pace)
                found.add(dot)

        return drawn, found

    # -- construction ------------------------------------------------------

    def construct(self):
        self.camera.background_color = PAPER
        self.pending = {"forced", "miss", "edge", "diagonal-probes", "diagonal-jump", "pruning"}

        free, path, events = run_search()
        start_state, goal_state = path[0], path[-1]
        self.goal_state = goal_state  # the one cell arrows stop short of: see arrow_point

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
        self.grid.rule(stroke_at(1.6, self.view_width))

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
                # The node coming off the open list, ringed round the dot already sitting
                # there. Created rather than faded in, so the ring traces itself round the
                # dot and the eye is walked over the cell it is about to scan out of.
                cursor = self.node_ring(event.state, DIAGONAL)
                self.play(Create(cursor), run_time=0.45)
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
        flash.lines.set_z_index(12)  # Flash builds its lines at z 0, under everything drawn
        # The square alone pulses, not the marker: Indicate recolours what it is given, and
        # the G going GOAL-coloured on a GOAL fill would blank the letter for the beat. It
        # sits still in the middle while the square swells behind it, which reads the same.
        self.play(flash, Indicate(self.grid.marker_cells[1], color=GOAL, scale_factor=1.4),
                  run_time=1.0)

        # -- the path -------------------------------------------------------

        # A plain unbroken line, no head: nothing about it is a scan, and the goal square it
        # runs into says which end it finished on. Above the gridlines, unlike the scans.
        # Both ends stop short of the letter in the cell rather than on the cell's centre:
        # the line is drawn over the markers, and running it into the middle of the S and
        # the G would strike each glyph through.
        corners = list(self.grid.cell_points(path))
        if len(corners) > 1:
            corners[0] = self.clear_of_label(corners[0], corners[1], self.grid.markers[0][1])
            corners[-1] = self.clear_of_label(corners[-1], corners[-2], self.grid.markers[1][1])
        path_line = VMobject(z_index=11)
        path_line.set_points_as_corners(evenly_sampled(corners))
        path_line.set_stroke(color=PATH, width=stroke_at(4.5, self.view_width))

        self.play(jump_points.animate.set_opacity(0.25),
                  scans.animate.set_opacity(0.12), run_time=0.9)
        self.say(f"{plural(len(path), 'jump point')} on the path, "
                 f"and it turns {plural(len(path) - 2, 'time')}")
        # Eased rather than linear now the speed is even: the line starts and stops instead
        # of snapping into motion, which a sine ease does without the long crawl either side
        # that smooth() puts on a three-second draw.
        self.play(Create(path_line), run_time=3.0, rate_func=rate_functions.ease_in_out_sine)
        self.wait(2)
