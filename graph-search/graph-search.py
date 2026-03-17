from manim import *
from itertools import permutations
from manim.utils.color import interpolate_color

import networkx as nx
import sys, os, string
from random import randrange
from numpy import array, cos, sin
# Get the absolute path to the directory containing the target of the symlink
# os.path.realpath(__file__) gets the path to the resolved target file
symlink_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(symlink_dir + "/pathfinding_link")

from search.nodes import Node
from domains.astar import Astar
from domains.gridmap import Gridmap, GridExpander4C, GridExpander8C
from domains.jps import JumpPointSearch
from search.nodes import Node
from search.tie_breakers import StandardTieBreaker
from search.heuristics import OctileDistanceHeuristic
from utils.scenario_parser import ScenarioParser, Scenario


# Convert moves to grid indices
def add_move(curr, move):
    x, y, z = curr + move
    return x, y, z


# Calculate direction vectors for offsetting
def offset_point(p1, p2, amount):
    direction = (p2 - p1)
    direction[2] = 0
    direction = direction / np.linalg.norm(direction)
    return p1 + direction * amount

def get_all_moves(lgrid_dims):
    # Define paths for all permutations of moves between S and G
    # These paths iterate through every perumtation of adding x, y in range (0, 3)
    moves = []
    for _ in range(lgrid_dims * lgrid_dims):
        moves.append([(0, 0, 0)])  # Each list is a distinct object
    
    move_steps = [(1, 0, 0)] * (lgrid_dims - 1) + [(0, 1, 0)] * (lgrid_dims - 1)
    all_paths = list(permutations(move_steps))
    moves.clear()
    for path in all_paths:
        moves.append([(0, 0, 0)] + list(path))
    for move in moves:
        move.append((0, 0, 0))
    
    # Remove duplicate paths from moves
    unique_moves = []
    seen = set()
    for move in moves:
        move_tuple = tuple(move)
        if move_tuple not in seen:
            unique_moves.append(move)
            seen.add(move_tuple)
    moves = unique_moves
    return moves


def get_unique_color(base_colors, i, total):
            # Interpolate between base colors
            idx = i * (len(base_colors) - 1) / max(1, total - 1)
            low = int(idx)
            high = min(low + 1, len(base_colors) - 1)
            frac = idx - low
            return interpolate_color(base_colors[low], base_colors[high], frac)

def run_search():
    os.chdir("pathfinding_link")
    scenario_parser = ScenarioParser()
    scenarios = scenario_parser.parse("maps/brc101d.map.scen", direct_mapnames=True)
    # scenarios = scenario_parser.parse("maps/AcrossTheCapeTest.map.scen", direct_mapnames=True)
    
    map_file = scenarios[0].map_file
    gridmap = Gridmap()
    gridmap.load(map_file)
    print("running search on map: ", map_file)
    for i, scenario in enumerate(scenarios):
        next_mapfile = scenario.map_file
        if next_mapfile != map_file:
            gridmap.load(next_mapfile)
            map_file = next_mapfile
        
        start_state = gridmap.cnvt_to_padded(scenario.start)
        goal_state = gridmap.cnvt_to_padded(scenario.goal)

        heuristic = OctileDistanceHeuristic()
        expander = JumpPointSearch(gridmap, heuristic, Node(state=goal_state))
        tie_breaker = StandardTieBreaker()

        astar = Astar(gridmap, heuristic, tie_breaker, expander)
        path = astar.search(start_state, goal_state)
        full_path = expander.reconstruct_jps_path(path)
        print("path found: ", path)
        return path, full_path, gridmap

def create_manim_grid_from_map(gridmap, path, slen=1):
    print("creating manim grid from map...")
    # create a cropped grid around the start and goal locations
    start = gridmap.cnvt_to_padded(path[0])
    goal = gridmap.cnvt_to_padded(path[-1])
    min_x = max(0, min(start[0], goal[0]))
    max_x = min(gridmap.width_, max(start[0], goal[0]) + 1)
    min_y = max(0, min(start[1], goal[1]))
    max_y = min(gridmap.height_, max(start[1], goal[1]) + 1)
    # min_x, min_y = 0,0
    # max_x, max_y = gridmap.width_, gridmap.height_
    
    
    grid = VGroup(*[
        Square(side_length=slen).move_to(
            np.array([
                x * slen,
                y * slen,
                0
            ])
        ).set_fill(color=BLUE, opacity=1 if not gridmap.is_free((x, y)) else 0)
        for y in range(min_y-1, max_y-1)
        for x in range(min_x-1, max_x-1)
    ])
    # if height is longer than width, rotate the grid by 90 degrees to make it wider than taller
    # if (max_y - min_y) > (max_x - min_x):
    #     grid.rotate(PI/2)
    
    return grid, min_x, min_y, max_x, max_y

def get_path_moves(grid, path, slen=1):
    moves = []
    for i in range(1, len(path)):
        move = (path[i][0] - path[i-1][0], path[i][1] - path[i-1][1], 0)
        # scale move by gridsize to correct coordiantes for grid starting from (0,0)
        moves.append(move)
    return moves

 
# TypeError: MovingCameraScene.__init__() takes from 1 to 2 positional arguments but 4 were given
class GraphSearchScene(MovingCameraScene):
    SIDE_LENGTH = 0.6
    
    def animate_traversal(self, graph: Graph, path, colour, has_weights: bool=False):
        path = [("A","B"), ("B","D"), ("D","H"), ("D","I"), ("B", "E")]
        to_animate, node_animate = [], []
        node_animate.append(graph.vertices[path[0][0]])
        if has_weights:
            for i in range(len(path)):
                target_angle = graph.edges[path[i]].get_angle()
                offset = (self.SIDE_LENGTH / 2) * array([cos(target_angle), sin(target_angle), 0])
                to_animate.append(Line(graph.vertices[path[i][0]].get_center(),
                                graph.edges[path[i]].get_center() - offset,
                                color=colour,
                                ))
                to_animate.append(Line(graph.edges[path[i]].get_center() + offset,
                                graph.vertices[path[i][1]].get_center(),
                                color=colour,
                                ))
                node_animate.append(graph.vertices[path[i][1]])
            count = 0
            for i, anim in enumerate(to_animate):
                target_circ = Circle(color=YELLOW)
                target_circ.surround(node_animate[count], buffer_factor=0.7)
                if i % 2 == 0:
                    self.play(node_animate[count][0].animate.set_fill(colour),
                            node_animate[count][1].animate.set_fill(BLACK),
                            #   Circumscribe(node_animate[count], shape=Circle, buff=0, color=YELLOW),
                            run_time=0.2,)
                    self.play(Create(target_circ), run_time=0.2)
                    target_circ.reverse_direction()
                    self.play(Uncreate(target_circ), run_time=0.2)
                    self.play(Create(anim, rate_func=rate_functions.rush_into))
                    ## USE AnimationGroup
                    count += 1
                else:
                    self.play(Create(anim, rate_func=rate_functions.rush_from))
                    
        else:
            for i in range(len(path)):
                to_animate.append(Line(graph.vertices[path[i][0]].get_center(),
                                graph.vertices[path[i][1]].get_center(),
                                color=colour,
                                ))
                node_animate.append(graph.vertices[path[i][1]].animate.set_color(colour))
            for i, anim in enumerate(to_animate):
                self.play(node_animate[i][0].animate.set_fill(colour))
                self.play(node_animate[i][1].animate.set_fill(BLACK))
                self.play(Create(anim))

    
    def construct(self):
        G = nx.Graph()
        G.add_node("A")

        letters = string.ascii_uppercase[1:]
        # b = 2
        # for i in range(b):
        #     G.add_node(letters[i])
        #     G.add_edge('A', letters[i], weight=1)
        #     # add children
        #     for j in range(b):
        #         G.add_node(letters[b+b*i+j])
        #         G.add_edge(letters[i], letters[b+b*i+j], weight=1)
        #         for k in range(b):
        #             G.add_node(letters[b+b**2+b**2*i+b*j+k])
        #             G.add_edge(letters[b+b*i+j], letters[b+b**2+b**2*i+b*j+k], weight=1)
        
        b = 2
        idx = 0
        # depth 1 (b nodes)
        level1 = []
        for i in range(b):
            node = letters[idx]
            idx += 1
            G.add_node(node)
            G.add_edge('A', node, weight=randrange(0,10))
            level1.append(node)

        # depth 2 (b^2 nodes)
        level2 = []
        for parent in level1:
            for j in range(b):
                node = letters[idx]
                idx += 1
                G.add_node(node)
                G.add_edge(parent, node, weight=randrange(0,10))
                level2.append(node)

        # depth 3 (b^3 nodes)
        level3 = []
        for parent in level2:
            for k in range(b):
                node = letters[idx]
                idx += 1
                G.add_node(node)
                G.add_edge(parent, node, weight=randrange(0,10))
                level3.append(node)
 
        # reverse the order of edges in each row at a time (branching factor = b)
        edges = list(G.edges)
        edges_reversed = []
        for i in range(0, len(edges), b):
            chunk = edges[i:i+b]
            edges_reversed.extend(chunk[::-1])   # reverse edges within the chunk
            edges = list(G.edges)
        nodes = list(G.nodes)
        nodes_reversed = []
        for i in range(0, len(nodes), b):
            chunk = nodes[i:i+b]
            nodes_reversed.extend(chunk[::-1])   # reverse nodes within the chunk

        
        g = Graph(nodes, edges_reversed, layout='tree', root_vertex='A', labels=True,
                        vertex_config={'radius': 0.40, 'fill_color': WHITE, 'stroke_color': BLUE},
                        layout_config={"vertex_spacing": (1.5, 2)},
                               )
        self.play(Create(g), run_time=2.5)
        for node in g.vertices:
            g.vertices[node].set_z_index(2)
        
        edge_blocks = VGroup()
        edge_labels = VGroup()
        for data in G.edges.data():
            edge = data[:2]
            weight = data[2]['weight']
            print("weight", weight, "location", g.edges[edge].get_center())
            label = Tex(weight).move_to(g.edges[edge].get_center())
            target_angle = g.edges[edge].get_angle() + PI/2
            square = Square(side_length=self.SIDE_LENGTH, stroke_width=1, color="black", fill_opacity=1).move_to(g.edges[edge].get_center()).rotate(target_angle)
            # circ = Circle(radius=0.3, stroke_width=1, color="black", fill_opacity=1).move_to(g.edges[edge].get_center())
            edge_blocks.add(square)
            # edge_labels.add(circ)
            edge_labels.add(label)
        
        self.add(edge_blocks)
        self.add(edge_labels)
        self.wait(1)
        
        self.animate_traversal(g, [], ORANGE, has_weights=True)
        return
        slen = 1   
        jps_path, path, gridmap = run_search()
        grid, minx, miny, maxx, maxy = create_manim_grid_from_map(gridmap, path, slen)
        
        self.add(grid)
        # frame the camera width by the grid width AND the grid height
        self.camera.frame.set(width=(grid.width + 1), height=(grid.height + 1)).move_to(grid.get_center())
        
        # start, goal = path[0], path[-1]
        # start pos in croped grid is 0th index shifted by cropx and cropy
        start_node, goal_node = path[0], path[-1]
        
        # Add Respective start and goal labels for each grid
        start_label = Tex("$S$").scale(slen).move_to(np.array([start_node[0], start_node[1], 0]))
        goal_label = Tex("$G$").scale(slen).move_to(np.array([goal_node[0], goal_node[1], 0]))
        self.add(start_label)
        self.add(goal_label)

        # animate a line following the path from start to goal
        path_line = VMobject().set_stroke(color=YELLOW, width=10)
        # create points from the states in jps_path, no need to shift the points
        points = [np.array([state[0], state[1], 0]) for state in jps_path]
        path_line.set_points_as_corners(points)
        self.play(Create(path_line), run_time=2 + len(jps_path) * 0.2, rate_func=linear)
        self.wait(1)