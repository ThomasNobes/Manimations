from manim import *
from itertools import permutations
from manim.utils.color import interpolate_color

import networkx as nx
import sys, os, string
from random import randrange
from numpy import array, cos, sin, sqrt

from manim import *
import random

blended_blue = interpolate_color(BLACK, BLUE, 0.6)
global_positions = []

directions = [
            (-1, -1), (-1, 0), (-1, 1),
            ( 0, -1),          ( 0, 1),
            ( 1, -1), ( 1, 0), ( 1, 1)
        ]
dir_names = [
    'nw', 'n', 'ne',
    'w',       'e',
    'sw', 's', 'se'
]
slide_dir = {
    'n':  LEFT,
    's':  LEFT,
    'w':  UP,
    'e':  UP,
    'nw': LEFT,
    'ne': RIGHT,
    'sw': LEFT,
    'se': RIGHT,
}

class TableRLE(VGroup):
    def __init__(self, data, stretch_width, stretch_height, row_labels=None, col_labels=None, include_outer_lines=True, include_inner_lines=True, **kwargs):
        super().__init__()

        # main table
        table = Table(
            data,
            include_outer_lines=include_outer_lines,
            element_to_mobject=lambda s: MathTex(s),
            arrange_in_grid_config={"cell_alignment": LEFT},
            **kwargs
        )
        if not include_inner_lines:
            for i, line in enumerate(table.get_horizontal_lines()):
                if include_outer_lines and i == 0:
                    continue
                line.set_stroke(width=0)
            for i, line in enumerate(table.get_vertical_lines()):
                if include_outer_lines and i == 0:
                    continue
                line.set_stroke(width=0)
        table.stretch_to_fit_width(stretch_width)
        table.stretch_to_fit_height(stretch_height)

        # Get left edge of first column
        first_col_left = table.get_columns()[0].get_left()[0]

        # Row labels
        row_labels_group = VGroup()
        for i in range(len(data)):
            label = MathTex(row_labels[i])
            # Position the label so its right edge is to the left of the first column
            label.next_to(table.get_rows()[i], LEFT, buff=0.6)
            # Optional: ensure precise alignment
            label.shift((first_col_left - label.get_right()[0] - 0.9) * RIGHT)
            row_labels_group.add(label)
        
        # Column labels
        col_labels_group = VGroup(*[
            MathTex(col_labels[j]).next_to(table.get_columns()[j], UP, buff=0.6)
            for j in range(len(data[0]))
        ])
        
        start_lab = Tex('start', color=BLUE).next_to(row_labels_group, LEFT).rotate(PI/2)
        target_lab = Tex('target', color=BLUE).next_to(col_labels_group, UP)
        
        # Group everything
        full_table = VGroup(table, row_labels_group, col_labels_group, start_lab, target_lab)
        self.add(full_table)
        self.table = table
        self.row_labels = row_labels_group
        self.col_labels = col_labels_group
        self.start_lab = start_lab
        self.target_lab = target_lab
    
    def hide_none_chars(self):
        rows = self.table.get_rows()
        for i, row in enumerate(rows):
            for j, val in enumerate(row):
                if val.tex_string == '-/-':
                    self.table.get_entries((i + 1, j + 1)).set_opacity(0)


class CornerTable(VGroup):
    def __init__(self, data, row_labels=None, col_labels=None, include_outer_lines=True, include_inner_lines=True, **kwargs):
        super().__init__()

        # main table
        table = Table(
            data,
            include_outer_lines=include_outer_lines,
            element_to_mobject=lambda s: MathTex(s),
            **kwargs
        )
        if not include_inner_lines:
            for i, line in enumerate(table.get_horizontal_lines()):
                if include_outer_lines and i == 0:
                    continue
                line.set_stroke(width=0.5)
            for i, line in enumerate(table.get_vertical_lines()):
                if include_outer_lines and i == 0:
                    continue
                line.set_stroke(width=0.5)

        # Get left edge of first column
        first_col_left = table.get_columns()[0].get_left()[0]

        # Row labels
        row_labels_group = VGroup()
        for i in range(len(data)):
            label = MathTex(row_labels[i])
            # Position the label so its right edge is to the left of the first column
            label.next_to(table.get_rows()[i], LEFT, buff=0.6)
            # Optional: ensure precise alignment
            label.shift((first_col_left - label.get_right()[0] - 0.9) * RIGHT)
            row_labels_group.add(label)

        # Column labels
        col_labels_group = VGroup(*[
            MathTex(col_labels[j]).next_to(table.get_columns()[j], UP, buff=0.6)
            for j in range(len(data[0]))
        ])
        
        start_lab = Tex('start', color=BLUE).next_to(row_labels_group, LEFT).rotate(PI/2)
        target_lab = Tex('target', color=BLUE).next_to(col_labels_group, UP)
            
        # Group everything
        full_table = VGroup(table, row_labels_group, col_labels_group, start_lab, target_lab)
        self.add(full_table)
        self.table = table
        self.row_labels = row_labels_group
        self.col_labels = col_labels_group
        self.start_lab = start_lab
        self.target_lab = target_lab


class CompressedPathDatabasesScene(MovingCameraScene):
    def SlidingTilePuzzle(self):
        # Grid size
        n = 3

        # Create tiles (numbered 1 to 15) and a blank
        numbers = list(range(1, n*n))
        numbers.append(None)  # empty tile
        random.shuffle(numbers)

        # Tile size
        tile_size = 1.5

        # Create tile objects
        tiles = {}
        tile_group = VGroup()
        for i, num in enumerate(numbers):
            row = i // n
            col = i % n
            pos = np.array([col*tile_size, -row*tile_size, 0])

            if num is not None:
                tile = Square(side_length=tile_size, color=BLUE, fill_opacity=0.7)
                label = MathTex(str(num)).move_to(tile.get_center())
                tile.add(label)
                tile.move_to(pos)
                tile_group.add(tile)
                tiles[num] = tile
            else:
                empty_pos = pos  # track empty space

        # Store positions in a dict (row, col) -> position
        positions = {}
        for row in range(n):
            for col in range(n):
                positions[(row, col)] = np.array([col*tile_size, -row*tile_size, 0])

        # Add all tiles
        self.add(tile_group)

        # Function to find empty neighbor tiles
        def neighbors(pos):
            row, col = pos
            result = []
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = row+dr, col+dc
                if 0 <= nr < n and 0 <= nc < n:
                    result.append((nr,nc))
            return result

        # Map positions to row,col
        tile_positions = {}
        idx = 0
        for row in range(n):
            for col in range(n):
                num = numbers[idx]
                tile_positions[(row,col)] = num
                idx += 1

        # Simple shuffle animation (slide 5 random tiles)
        for _ in range(5):
            er, ec = next((r, c) for (r, c), v in tile_positions.items() if v is None)
            neighbors_list = neighbors((er, ec))
            nr, nc = random.choice(neighbors_list)
            num = tile_positions[(nr, nc)]
            tile = tiles[num]

            # animate sliding tile into empty
            self.play(tile.animate.move_to(positions[(er,ec)]), run_time=0.5)

            # update positions
            tile_positions[(er,ec)] = num
            tile_positions[(nr,nc)] = None


    def GridExample(self):
        # Grid size
        n = 3

        # Create tiles (numbered 1 to 15) and a blank
        numbers = list(range(1,5))
        numbers.append(None)  # empty tile
        numbers.append(None)  # empty tile
        numbers += list(range(5,8))
        # random.shuffle(numbers)

        # Tile size
        tile_size = 2

        # Create tile objects
        tiles = {}
        tile_group = VGroup()
        for i, num in enumerate(numbers):
            row = i // n
            col = i % n
            pos = np.array([col*tile_size, -row*tile_size, 0])

            if num is not None:
                tile = Square(side_length=tile_size, color=BLUE, fill_opacity=0.6)
                label = MathTex(str(num)).move_to(tile.get_center())
                tile.add(label)
                tile.move_to(pos)
                tile_group.add(tile)
                tiles[num] = tile
            if num is None:
                tile = Square(side_length=tile_size, color=RED, fill_opacity=0.7)
                tile.move_to(pos)
                tile_group.add(tile)
                tiles[num] = tile

        # Store positions in a dict (row, col) -> position
        positions = {}
        for row in range(n):
            for col in range(n):
                positions[(row, col)] = np.array([col*tile_size, -row*tile_size, 0])
        
        # draw arrows from row 1 col 2 to its neighbours (including diagonals)
        # Define the cell to draw arrows from (row=1, col=2)
        source_row, source_col = 1, 0  # zero-indexed

        dir_dict = dict(zip(directions, dir_names))

        arrows = VGroup()
        labels = VGroup()
        for dr, dc in directions:
            nr, nc = min(source_row + dr,n-1), min(source_col + dc,n-1)
            if 0 <= nr < n and 0 <= nc < n:
                if len(tile_group[nr * n + nc]) != 2:
                    continue # skip blank tiles
                neighbor_center = positions[(nr, nc)]
                source_center = positions[(source_row, source_col)]
                
                # arrow = Arrow(start=positions[(source_row, source_col)], end=neighbor_center, buff=0.4, stroke_width=6, max_stroke_width_to_length_ratio=5, color=YELLOW)
                # arrows.add(arrow)
                # Vector from source to target
                vec = neighbor_center - source_center
                vec_unit = vec / np.linalg.norm(vec)

                # Shorten both ends by half the arrow_length so arrow is centered
                if abs(dr) > 0 and abs(dc) > 0:
                    start = source_center + vec_unit * 0.5
                    end = neighbor_center - vec_unit * 0.5
                    arrow = Arrow(start=start, end=end, buff=0, stroke_width=5, color=YELLOW)
                else:
                    start = source_center + vec_unit * 0.48
                    end = neighbor_center - vec_unit * 0.48
                    arrow = Arrow(start=start, end=end, buff=0, stroke_width=8, color=YELLOW,
                                  max_tip_length_to_length_ratio=0.5)
                arrows.add(arrow)
                label = MathTex(dir_dict[(dr,dc)], color=YELLOW).next_to(arrow, slide_dir[dir_dict[(dr,dc)]], buff=0.1).scale(1.2)
                labels.add(label)

        return tile_group, arrows, labels


    def NodeExample(self):
        # Grid size
        n = 3

        # Create tiles (numbered 1 to 15) and a blank
        numbers = list(range(1,5))
        numbers.append(None)  # empty tile
        numbers.append(None)  # empty tile
        numbers += list(range(5,8))
        # random.shuffle(numbers)

        # Tile size
        tile_size = 2

        # Create tile objects
        nodes = {}
        nodes_group = VGroup()
        labels = VGroup()
        for i, num in enumerate(numbers):
            row = i // n
            col = i % n
            pos = np.array([col*tile_size, -row*tile_size, 0])

            if num is not None:
                node = Circle(radius=tile_size/4, color=WHITE, fill_opacity=1).set_fill(blended_blue)
                label = MathTex(str(num)).move_to(node.get_center())
                node.add(label)
                node.move_to(pos)
                nodes_group.add(node)
                labels.add(label.copy().set_z_index(1))
                nodes[num] = node
            if num is None:
                node = Square(side_length=tile_size, fill_opacity=0).set_stroke(opacity=0)
                node.move_to(pos)
                nodes_group.add(node)
                nodes[num] = node

        # Store positions in a dict (row, col) -> position
        positions = {}
        for row in range(n):
            for col in range(n):
                positions[(row, col)] = np.array([col*tile_size, -row*tile_size, 0])
        global global_positions
        global_positions = positions

        # draw arrows from row 1 col 2 to its neighbours (including diagonals)
        # Define the cell to draw arrows from (row=1, col=2)
        source_row, source_col = 1, 0  # zero-indexed
        lines = VGroup()
        for source_row in range(n):
            for source_col in range(n):
                if len(nodes_group[source_row * n + source_col]) != 2:
                            continue # skip blank nodes
                for dr, dc in directions:
                    nr, nc = source_row + dr, source_col + dc
                    if 0 <= nr < n and 0 <= nc < n:
                        if len(nodes_group[nr * n + nc]) != 2:
                            continue # skip blank nodes
                        neighbor_center = positions[(nr, nc)]
                        source_center = positions[(source_row, source_col)]
                        
                        # Vector from source to target
                        vec = neighbor_center - source_center
                        vec_unit = vec / np.linalg.norm(vec)

                        # Shorten both ends by half the arrow_length so arrow is centered
                        if abs(dr) > 0 and abs(dc) > 0:
                            start = source_center + vec_unit * 0.5
                            end = neighbor_center - vec_unit * 0.5
                            line = Line(start=start, end=end, buff=0, stroke_width=5, color=WHITE)
                        else:
                            start = source_center + vec_unit * 0.5
                            end = neighbor_center - vec_unit * 0.5
                            line = Line(start=start, end=end, buff=0, stroke_width=6, color=WHITE)
                        lines.add(line)

        return nodes_group, lines, labels
    
    
    def draw_arrow(self, start_loc, dir_name):
        dir_dict = dict(zip(dir_names, directions))
        source_row, source_col = start_loc
        dr, dc = dir_dict[dir_name]
        nr, nc = source_row + dr, source_col + dc
        
        source_center = global_positions[(source_row, source_col)]
        neighbor_center = global_positions[(nr, nc)]
        
        # Vector from source to target
        vec = neighbor_center - source_center
        vec_unit = vec / np.linalg.norm(vec)

        # Shorten both ends by half the arrow_length so arrow is centered
        if abs(dr) > 0 and abs(dc) > 0:
            start = source_center + vec_unit * 0.5
            end = neighbor_center - vec_unit * 0.5
            arrow = Arrow(start=start, end=end, buff=0, stroke_width=5, color=YELLOW)
        else:
            start = source_center + vec_unit * 0.48
            end = neighbor_center - vec_unit * 0.48
            arrow = Arrow(start=start, end=end, buff=0, stroke_width=8, color=YELLOW,
                            max_tip_length_to_length_ratio=0.5)
        label = MathTex(dir_name, color=YELLOW).next_to(arrow, slide_dir[dir_name], buff=0.1).scale(1.2)
        return arrow, label
    
    
    def draw_line_path(self, start_id, end_col, dir_name, table, nodes, node_row_len):
        curr_dir_name = dir_name.tex_string
        lines = VGroup()
        curr_loc = self.get_loc_from_node_id(start_id, nodes, node_row_len)
        next_loc = (None, None)
        backup_count = 0
        while curr_dir_name != "*" and backup_count < 100:
            dir_dict = dict(zip(dir_names, directions))
            source_row, source_col = curr_loc
            dr, dc = dir_dict[curr_dir_name]
            nr, nc = source_row + dr, source_col + dc
            
            source_center = global_positions[(source_row, source_col)]
            neighbour_center = global_positions[(nr, nc)]
            
            # Vector from source to target
            vec = neighbour_center - source_center
            vec_unit = vec / np.linalg.norm(vec)

            # Shorten both ends by half the arrow_length so arrow is centered
            if abs(dr) > 0 and abs(dc) > 0:
                start = source_center + vec_unit * 0.5
                end = neighbour_center - vec_unit * 0.5
                line = Line(start=start, end=end, buff=0, stroke_width=5, color=ORANGE)
            else:
                start = source_center + vec_unit * 0.5
                end = neighbour_center - vec_unit * 0.5
                line = Line(start=start, end=end, buff=0, stroke_width=6, color=ORANGE)
            lines.add(line)
            next_id = self.get_node_id(nr, nc, nodes, grid_len=len(nodes), row_len=node_row_len)
            prev_dir_name = curr_dir_name
            curr_dir_name = table.get_rows()[next_id][end_col].tex_string
            curr_loc = (nr, nc)
            backup_count += 1
        return lines
    
    
    def get_node_id(self, row, col, nodes, grid_len, row_len):
        node_id = row * row_len + col
        skip_count = 0
        for i in range(node_id):
            if len(nodes[i]) != 2:
                skip_count += 1
        id_to_node_num = node_id - skip_count
        return id_to_node_num
    
    
    def get_loc_from_node_id(self, node_id, nodes, row_len):
        skip_count = 0
        for i in range(node_id):
            if len(nodes[i]) != 2:
                skip_count += 1
        node_id += skip_count
        nr, nc = node_id // row_len, node_id % row_len
        return nr, nc
    
    
    def get_runs_in_rows(self, table):
        rows = table.table.get_rows()
        runs = [[] for _ in range(len(rows))]
        for i, row in enumerate(rows):
            curr_letter = row[0].get_tex_string()
            col_start = 1
            run = 0
            for j, tex in enumerate(row):
                next_letter = tex.get_tex_string()
                if curr_letter == next_letter:
                    run += 1
                elif next_letter == '*':
                    run += 1
                else:
                    if j != len(row)-1:
                        runs[i].append(f'{col_start}/{curr_letter}')
                        run = 0
                        col_start = j+1
                if j == len(row)-1:
                    if curr_letter == next_letter or next_letter == '*':
                        runs[i].append(f'{col_start}/{curr_letter}')
                    else:
                        runs[i].append(f'{col_start}/{curr_letter}')
                        runs[i].append(f'{j+1}/{next_letter}')
                elif next_letter == '*':
                    if j == 0:
                        curr_letter = row[j+1].get_tex_string()
                    pass
                else:
                    curr_letter = next_letter
        # convert runs into table format (all rows are same length)
        n = len(table.table.get_columns())
        num_runs = 0
        tab_runs = []
        for i, row in enumerate(runs):
            num_runs += len(row)
            if len(row) != n:
                tab_runs.append(row + ['-/-' for _ in range(n - len(row))])
        return tab_runs, num_runs
    
    
    def move_column_to_end(self, data, col_index):
        for row in data:
            row.append(row.pop(col_index))
        return data
    
    
    def highlight(self, node):
        highlight = node.copy().set_color(YELLOW)
        self.play(Create(highlight))
        return highlight


    def remove_highlight(self, mobject):
        self.play(Uncreate(mobject), reverse_rate_function=True)
    

    def construct(self):
        n = 7

        # Create an outer border
        top_line = Line((0,0,0),(n,0,0))
        left_line = Line((0,0,0),(0,-n,0))
        # top_line = Line((0,-1,0),(n,-1,0))
        # left_line = Line((1,0,0),(1,-n,0))
        
        # Create table data
        data = [[str(i * n + j + 1) for j in range(n)] for i in range(n)]
        data = [['*','e','e','s','s','s','s'],
                ['w','*','e','sw','sw','sw','sw'],
                ['w','w','*','w','w','w','w'],
                ['n','ne','ne','*','s','se','se'],
                ['n','n','n','n','*','e','e'],
                ['nw','nw','nw','nw','w','*','e'],
                ['w','w','w','w','w','w','*'],
                ]
        table = CornerTable(data,
                            row_labels=[str(i+1) for i in range(n)],
                            col_labels=[str(i+1) for i in range(n)],
                            include_outer_lines=True,
                            include_inner_lines=False,
                            )
        
        self.camera.frame.set(width=self.camera.frame.width*1.6)
        tiles, arrows, labels = self.GridExample()
        # Center camera on group of objects
        group = VGroup(tiles, table)
        table.next_to(tiles, RIGHT*5)
        table_frame = table.copy()
        table_frame.table.get_entries().set_opacity(0)
        self.camera.frame.move_to(group)
        tiles.set_z_index(-2)
        
        # Add all tiles
        self.add(tiles)
        self.wait()
        
        nodes, lines, labels = self.NodeExample()
        lines.set_z_index(-1)
        
        # Add all nodes
        self.play(Create(nodes),
        )
        self.wait(0.5)
        self.add(labels)
        self.play(*[Create(line) for line in lines])
        self.wait()
        self.play(FadeOut(tiles))
        
        self.play(Write(table_frame))
        self.wait()
        start_node = 0
        self.play(Write(table.table.get_rows()[start_node]))
        self.wait()
        
        # save mobject states so that we can retreive after 'Unwrite'
        for i in range(1, len(table.table.get_rows()[start_node])):
            table.table.get_rows()[start_node][i].save_state()
        self.play(Unwrite(table.table.get_rows()[start_node][1:]))
        self.wait(0.2)
        self.play(nodes[start_node][0].animate.set_fill(ORANGE),
                  table.table.get_rows()[start_node][0].animate.set_color(ORANGE),
                  table.row_labels[0].animate.set_color(ORANGE),
        )
        self.wait()
        # highlight second node 2, 3, 4, etc.
        table_ind = 1
        for i in range(1, len(nodes)):
            if len(nodes[i]) != 2:
                continue # skip blank nodes/tiles
            # self.play(FocusOn(nodes[i].get_arc_center()))
            # self.play(nodes[i][0].animate.set_fill(YELLOW),
            path = self.draw_line_path(start_node, table_ind, table.table.get_rows()[0][table_ind],
                                       table.table, nodes, int(sqrt(len(nodes))))
            self.play(Circumscribe(nodes[i], color=YELLOW, shape=Circle, buff=0), Create(path))
            self.wait()
            arrow, arrow_label = self.draw_arrow((0,0), table.table.get_rows()[0][table_ind].tex_string)
            arr_group = VGroup(arrow, arrow_label)
            self.play(Restore(table.table.get_rows()[0][table_ind]),
                      Create(arr_group))
            # self.play(Indicate(nodes[i][0], color=YELLOW, scale_factor=1.3),
            self.play(Indicate(nodes[i][0], color=YELLOW, scale_factor=1.3),
                      Indicate(table.table.get_rows()[0][table_ind], color=YELLOW, scale_factor=5)
            )
            # self.play(nodes[i][0].animate.set_fill(blended_blue))
            
            # new_node = self.highlight(nodes[i])
            # self.play(Restore(table.table.get_rows()[0][i]))
            # self.remove_highlight(new_node)
            table_ind += 1
            self.wait(0.5 - 0.05 * i)
            self.play(Uncreate(path))
            self.play(Uncreate(arrow_label), Uncreate(arrow))
        self.wait()
        
        # draw the corresponding element between each step
        self.play(nodes[start_node][0].animate.set_fill(blended_blue),
                  table.table.get_rows()[start_node][0].animate.set_color(WHITE),
                  table.row_labels[0].animate.set_color(WHITE),
        )
        self.wait()
        self.play(Write(table.table.get_rows()[1:]))
        self.wait()
        diagonal_axes = VGroup()
        for i in range(len(table.table.get_rows()[0])):
            print("i", i)
            diagonal_axes.add(table.table.get_rows()[i][i])
        self.play(*[Indicate(mob) for mob in diagonal_axes], color=YELLOW, scale_factor=2)
        self.add(table)
        self.remove(table_frame)
        self.wait()
                
        row_to_color = 3
        table.row_labels[row_to_color].set_color(YELLOW)
        row = table.table.get_rows()[row_to_color]
        rowbox = SurroundingRectangle(VGroup(row, table.row_labels[row_to_color]), color=YELLOW)
        self.play(row.animate.set_color(YELLOW), Create(rowbox))
        
        self.play(*[Create(arrow) for arrow in arrows])
        self.play(AnimationGroup(*[Write(mob) for mob in labels]))
        self.wait(1)
         
        self.play(Uncreate(rowbox), reverse_rate_function=True)
        self.wait(0.3)
        self.play(row.animate.set_color(WHITE))
        table.row_labels[row_to_color].set_color(WHITE)
        
        runs, num_runs = self.get_runs_in_rows(table)
        max_run = max([len(runs[i]) for i in range(len(runs))])

        hidden_rle_table = TableRLE(runs,
                            stretch_width=table.table.width,
                            stretch_height=table.table.height,
                            row_labels=[str(i+1) for i in range(n)],
                            col_labels=['1' for _ in range(max_run)],
                            include_outer_lines=True,
                            include_inner_lines=False,
                            )
        hidden_rle_table.hide_none_chars()
        # hidden_rle_table.col_labels.set_opacity(0)
        hidden_rle_table.next_to(table, RIGHT*3.5)
        
        center_point = (table.get_center() + hidden_rle_table.get_center()) / 2
        self.play(self.camera.frame.animate.set_width(self.camera.frame.width*1.4))
        self.play(self.camera.frame.animate.move_to(center_point))
        
        rle_table = table.copy()
        # self.add(hidden_rle_table)
        
        # group1 = VGroup(table1.table, *table1.get_col_labels(), *table1.get_row_labels())
        # group2 = VGroup(table2.table, *table2.get_col_labels(), *table2.get_row_labels())

        # # Align both groups
        # group2.move_to(group1.get_center())
        # self.play(Transform(group1, group2))

        self.wait()
        self.play(rle_table.animate.move_to(hidden_rle_table))
        self.wait()
        
        self.play(
            rle_table.col_labels.animate.set_opacity(0),
            Transform(rle_table.table, hidden_rle_table.table),
            # TransformMatchingTex(rle_table.table.get_entries(), hidden_rle_table.table.get_entries()),
            Transform(rle_table.target_lab, Tex('target (RLE)', color=BLUE).move_to(rle_table.target_lab.get_center())),
        )
        self.wait()
        rle_num = MathTex(r"\rightarrow").next_to(rle_table.target_lab, RIGHT)
        rle_num2 = Tex(f"{num_runs} runs").next_to(rle_num, RIGHT)
        self.play(Write(rle_num), Write(rle_num2))
        self.wait(0.5)
        shift_amount = 4.2
        self.play(table.animate.shift(UP*shift_amount),
                  rle_table.animate.shift(UP*shift_amount),
                  rle_num.animate.shift(UP*shift_amount),
                  rle_num2.animate.shift(UP*shift_amount),
        )
        
        swapped_table = table.copy().set_z_index(-2)
        swapped_rle = rle_table.copy().set_z_index(-2)
        save_loc = swapped_rle.copy().set_z_index(-2)
        swapped_table.start_lab.set_color(GREEN),
        swapped_table.target_lab.set_color(GREEN),
        swapped_rle.start_lab.set_color(GREEN),
        swapped_rle.target_lab.set_color(GREEN),
                
        self.play(
                swapped_table.animate.shift(DOWN*2.1*shift_amount),
                swapped_rle.animate.shift(DOWN*2.1*shift_amount),
                FadeOut(swapped_rle.table.get_entries()),
        )
        
        columns = swapped_table.table.get_columns()
        pop_index = 2
        popped_col = columns[pop_index]
        popped_label = swapped_table.col_labels[pop_index]
        shift_amount = popped_col.width * 3
        later_cols = columns[pop_index + 1:]
        later_col_labels = swapped_table.col_labels[pop_index + 1:]
        last_col = later_cols[-1]
        
        swapped_data = self.move_column_to_end(data, pop_index)
        better_table = CornerTable(swapped_data,
                            row_labels=[str(i+1) for i in range(n)],
                            col_labels=[str(i+1) for i in range(n)],
                            include_outer_lines=True,
                            include_inner_lines=False,
                            )  

        self.play(swapped_table.table.get_columns()[pop_index].animate.set_color(RED),
                  popped_label.animate.set_color(RED))
        self.wait(0.5)
        self.play(
            # FadeOut(popped_col),
            *[col.animate.shift(LEFT * shift_amount) for col in later_cols],
            *[col_lab.animate.shift(LEFT * shift_amount) for col_lab in later_col_labels],
            popped_col.animate.shift(RIGHT * shift_amount * len(later_cols)),
            popped_label.animate.shift(RIGHT * shift_amount * len(later_cols))
            # popped_col.animate.next_to(last_col, RIGHT, buff=swapped_table.table.h_buff)
        )
        self.wait(1)
        # self.play(popped_col.animate.next_to(last_col, RIGHT, buff=swapped_table.table.h_buff))
        
        better_runs, better_num_runs = self.get_runs_in_rows(better_table)
        better_rle_table = TableRLE(better_runs,
                            stretch_width=table.table.width,
                            stretch_height=table.table.height,
                            row_labels=[str(i+1) for i in range(n)],
                            col_labels=['1' for _ in range(max_run)],
                            include_outer_lines=False,
                            include_inner_lines=False,
                            ).move_to(save_loc)
                            # ).move_to(swapped_rle).shift(DOWN*2.1*shift_amount)
        better_rle_table.shift(DOWN*4.7*shift_amount)
        # better_rle_table.table.shift(DOWN*4.7*shift_amount)
        better_rle_table.start_lab.set_opacity(0)
        better_rle_table.target_lab.set_opacity(0)
        better_rle_table.row_labels.set_opacity(0)
        better_rle_table.col_labels.set_opacity(0)
        better_rle_table.hide_none_chars()
        better_rle_num = MathTex(r"\rightarrow").next_to(better_rle_table.target_lab, RIGHT*4)
        better_rle_num2 = Tex(f"{better_num_runs} runs").next_to(better_rle_num, RIGHT)
        # for row in better_rle_table.table.get_rows():
        #     self.play(Create(row))
        self.play(Create(better_rle_table))
        self.wait()
        self.play(Write(better_rle_num), Write(better_rle_num2))
        self.wait()
        
        # self.SlidingTilePuzzle()
        