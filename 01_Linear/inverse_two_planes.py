"""在仓库根目录运行 uv run manim -pqm 01_Linear/inverse_two_planes.py InverseTwoPlanes。

与同目录的 inverse_by_row_operations.py 共用有理数消元和矩阵排版。
"""

import numpy as np
from manim import (
    DOWN, LEFT, ORIGIN, RIGHT, Arrow, DashedVMobject, DecimalNumber, Dot,
    FadeIn, FadeOut, Line, NumberPlane, Polygon, Rectangle, Scene, Text,
    Transform, UpdateFromAlphaFunc, VGroup, smooth,
)

from inverse_by_row_operations import (
    CORAL, GOLD, GRID, IDENTITY, INK, MUTED, SUBSCRIPT, TEAL,
    entry, label, linear_form, matrix_picture, place, row_reduce, scalar,
)


# 与单图版采用同一个例子。可在这里单独更换矩阵，分数可写成 "1/3"。
MATRIX = ((1, 2), (-1, 1))
COLUMN_COLORS = (TEAL, CORAL)
MATRIX_SCALE = 0.77


class BlockView:
    """一块 2×2 矩阵及其在固定坐标系中的两个列向量。"""

    def __init__(self, states, center_x, heading, start_symbol, tint):
        self.tint = tint
        # 网格包含单位正方形以及这块矩阵在各步的四个顶点。
        corners = [np.zeros(2)]
        for block in [IDENTITY, *states]:
            corners.extend((*block.T, block.sum(axis=1)))
        points = np.asarray(corners, dtype=float)
        low, high = points.min(axis=0), points.max(axis=0)
        margin = 0.16 * max(high - low) + 0.10
        box = np.array([6.00, 3.77])
        unit = min(box / (high - low + 2 * margin))
        midpoint = (low + high) / 2
        low, high = midpoint - box / (2 * unit), midpoint + box / (2 * unit)
        tick = max(1, int(np.ceil(max(high - low) / 10)))
        self.plane = NumberPlane(
            x_range=[low[0], high[0], tick], y_range=[low[1], high[1], tick],
            x_length=box[0], y_length=box[1], faded_line_ratio=1, tips=False,
            axis_config={"color": MUTED, "stroke_width": 1.65},
            background_line_style={"stroke_color": GRID, "stroke_width": 1.10},
            faded_line_style={"stroke_color": GRID, "stroke_width": 0.65, "stroke_opacity": 0.38},
        ).move_to([center_x, -0.42, 0])
        origin = self.plane.c2p(0, 0)
        axes = VGroup(label("O", 15, MUTED).next_to(origin, DOWN + LEFT, buff=0.05))
        for axis, direction, name in ((0, DOWN, "x"), (1, LEFT, "y")):
            for k in range(int(np.ceil((low[axis] + 0.07 * tick) / tick)),
                           int(np.floor((high[axis] - 0.07 * tick) / tick)) + 1):
                if k:
                    xy = np.zeros(2)
                    xy[axis] = k * tick
                    axes.add(label(scalar(k * tick), 16, MUTED).next_to(self.plane.c2p(*xy), direction, buff=0.06))
            xy = np.zeros(2)
            xy[axis] = high[axis]
            axes.add(label(name, 18, MUTED).next_to(self.plane.c2p(*xy), RIGHT, buff=0.06))

        self.table = matrix_picture(states[0]).scale(MATRIX_SCALE).move_to([center_x + 1.52, 2.25, 0])
        for index, cell in enumerate(self.table[1]):
            cell.set_color(COLUMN_COLORS[index % 2])
        left = center_x - 2.99
        name = place(label(heading, 32, tint), left, 2.51, 2.6)
        self.status = place(label(f"从 {start_symbol} 出发", 23, MUTED), left, 1.96, 2.75)
        standard = Polygon(*(self.plane.c2p(*xy) for xy in ((0, 0), (1, 0), (1, 1), (0, 1))),
                           stroke_color=MUTED, stroke_width=2.1, fill_opacity=0)
        dashed = DashedVMobject(standard, num_dashes=25, dashed_ratio=0.55).set_z_index(2)
        self.static = VGroup(self.plane, axes, dashed, name, self.status, self.table)
        self.plot = self.footprint(states[0])

    def footprint(self, block):
        v, w = np.asarray(block, dtype=float).T
        origin = self.plane.c2p(0, 0)
        p, q, far = (self.plane.c2p(*xy) for xy in (v, w, v + w))
        face = Polygon(origin, p, far, q, color=self.tint, stroke_width=2.2,
                       fill_opacity=0.16).set_z_index(1)
        arrows = []
        for tip, color in zip((p, q), COLUMN_COLORS):
            if np.linalg.norm(tip - origin) < 1e-8:
                arrow = Dot(origin, radius=0.042, color=color)
            else:
                arrow = Arrow(origin, tip, buff=0, color=color, stroke_width=5.1, tip_length=0.17)
            arrows.append(arrow.set_z_index(3))
        return VGroup(face, *arrows, Dot(far, radius=0.048, color=INK).set_z_index(4))

    def highlight(self, rows):
        return VGroup(*[
            Rectangle(width=self.table.width + 0.14, height=0.83 * MATRIX_SCALE, color=GOLD,
                      stroke_width=1.4, fill_opacity=0.075).move_to(
                [self.table.get_x(), self.table[1][2 * r].get_y(), 0]).set_z_index(-1)
            for r in rows
        ])


def caption(heading, content, x, width):
    return VGroup(place(label(heading, 22, MUTED), x, -2.68, width),
                  place(content, x, -3.41, width))


class InverseTwoPlanes(Scene):
    def construct(self):
        self.camera.background_color = "#101722"
        start, steps, missing = row_reduce(MATRIX)
        states = [start, *(s.augmented for s in steps)]
        views = (
            BlockView([s[:, :2] for s in states], -3.49, "左块", "A", "#9BD4D2"),
            BlockView([s[:, 2:] for s in states], 3.49, "右块", "I", "#E1C287"),
        )
        left, right = views
        title = place(label("用初等行变换求逆矩阵", 37), -6.52, 3.48, 6.45)
        legend = VGroup(label("第 1 列", 21, TEAL), label("第 2 列", 21, CORAL)).arrange(RIGHT, buff=0.27)
        place(legend, -0.83, 3.48, 2.35)
        goal = "[ A | I ]  →  [ I | A⁻¹ ]" if missing is None else "[ A | I ]"
        target = label(goal, 27).move_to([4.58, 3.48, 0])
        lines = VGroup(Line([-6.52, 3.04, 0], [6.52, 3.04, 0]),
                       Line([0, 2.87, 0], [0, -2.27, 0]),
                       Line([-6.52, -2.40, 0], [6.52, -2.40, 0])).set_stroke(GRID, width=1.2)
        info = VGroup(caption("同一步行变换", label("同时作用于左右四个列向量", 28), -6.47, 6.04),
                      caption("虚线标出各图的单位正方形", label("白点是两列向量之和的终点", 28), 0.50, 6.01))
        # 两幅图及运动时显示的小数放在同一动画中，共用一个进度 alpha。
        live_numbers = VGroup()
        motion = VGroup(left.plot, right.plot, live_numbers)
        self.play(FadeIn(VGroup(title, legend, target, lines, left.static, right.static)),
                  FadeIn(motion), run_time=1.0)
        self.play(FadeIn(info), run_time=0.50)
        self.wait(1.90)

        for n, step in enumerate(steps, start=1):
            sub = str(n).translate(SUBSCRIPT)
            rule_x, rule_y = (linear_form(row, ("x", "y")) for row in step.elementary)
            elementary = VGroup(label(f"E{sub} =", 26), matrix_picture(step.elementary).scale(0.62))
            elementary.arrange(RIGHT, buff=0.18)
            next_info = VGroup(
                caption(f"第 {n} 步  ·  {step.purpose}", label(step.operation, 32, GOLD), -6.47, 4.36),
                caption("两图共用的坐标规则", label(f"(x, y) → ({rule_x}, {rule_y})", 28), -1.68, 4.45),
                place(elementary, 3.49, -3.08, 3.0),
            )
            bands = VGroup(*(view.highlight(step.rows) for view in views))
            self.play(FadeOut(info), run_time=0.28)
            self.play(FadeIn(next_info), FadeIn(bands), run_time=0.50)
            self.wait(0.95)
            self.move_blocks(views, motion, live_numbers, states[n - 1], step.augmented)
            self.wait(1.45)
            self.play(FadeOut(bands), run_time=0.28)
            info = next_info

        self.play(FadeOut(info), run_time=0.30)
        if missing is None:
            inverse = states[-1][:, 2:]
            coords = [f"({scalar(x)}, {scalar(y)})" for x, y in inverse.T]
            tips = VGroup(*(label(xy, 30, color) for xy, color in zip(coords, COLUMN_COLORS)))
            tips.arrange(RIGHT, buff=0.36)
            ending = VGroup(caption("把这些行变换合起来，记为 E", label("EA = I，所以 E = A⁻¹", 29),
                                    -6.47, 6.04),
                            caption("右图的两个箭尖给出 A⁻¹ 的两列", tips, 0.50, 6.04))
            left_status, right_status = "得到单位矩阵 I", "得到逆矩阵 A⁻¹"
        else:
            ending = VGroup(caption(f"第 {missing + 1} 列没有可用主元", label("矩阵 A 不可逆", 30, GOLD),
                                    -6.47, 6.04),
                            caption("右图的两列", label("记录已经完成的行变换", 28), 0.50, 6.04))
            left_status, right_status = "无法化为 I", "当前累计变换"
        self.play(Transform(left.status, label(left_status, 23, INK).move_to(left.status, aligned_edge=LEFT)),
                  Transform(right.status, label(right_status, 23, GOLD).move_to(right.status, aligned_edge=LEFT)),
                  FadeIn(ending), run_time=0.70)
        self.wait(3.30)

    def move_blocks(self, views, motion, live_numbers, before, after):
        # 停稳时以分数显示精确值，只有正在变化的格子临时改为两位小数。
        changes = []
        for r, c in np.ndindex(before.shape):
            if before[r, c] == after[r, c]:
                continue
            cell = views[c // 2].table[1][2 * r + c % 2]
            color = COLUMN_COLORS[c % 2]
            number = DecimalNumber(-max(abs(float(before[r, c])), abs(float(after[r, c]))),
                                   mob_class=Text, num_decimal_places=2, edge_to_fix=ORIGIN,
                                   font_size=22, color=color)
            if number.width > 0.74:
                number.scale_to_fit_width(0.74)
            number.set_value(float(before[r, c])).move_to(cell)
            live_numbers.add(number)
            exact = entry(after[r, c], color).scale(MATRIX_SCALE).move_to(cell)
            cell.set_opacity(0)
            changes.append((r, c, number, cell, exact))

        base = before.astype(float)
        delta = (after - before).astype(float)

        def update(_, alpha):
            # after = E @ before，所以同一插值等价于对两块同时左乘 (1−α)I + αE。
            current = base + alpha * delta
            for view, block in zip(views, (current[:, :2], current[:, 2:])):
                view.plot.become(view.footprint(block))
            for r, c, number, _, _exact in changes:
                value = round(current[r, c], 2) or 0.0
                if value != number.get_value():
                    number.set_value(value)

        self.play(UpdateFromAlphaFunc(motion, update), rate_func=smooth, run_time=2.65)
        for _, _, number, cell, exact in changes:
            live_numbers.remove(number)
            cell.become(exact)
