"""在仓库根目录运行 uv run manim -pqm 01_Linear/augmented_system_2d.py AugmentedSystem2D。

与同目录的 inverse_by_row_operations.py 共用精确的有理数行变换。
"""

import numpy as np
from manim import (
    DOWN, LEFT, ORIGIN, RIGHT, UP, Arrow, DecimalNumber, Dot, FadeIn, FadeOut,
    Line, NumberPlane, Rectangle, Scene, Text, Transform, UpdateFromAlphaFunc,
    VGroup, smooth,
)

from inverse_by_row_operations import (
    CORAL, GOLD, GRID, INK, MUTED, TEAL, entry, label, linear_form,
    matrix_picture, place, row_reduce, scalar,
)


# 每行依次填写 x 的系数、y 的系数和等号右边的常数。分数可写成 "1/3"。
AUGMENTED = ((1, 2, 4), (-1, 1, -1))
ROW_COLORS = (TEAL, CORAL)
COLUMN_COLORS = ("#91BAFF", "#C7ACFF", GOLD)
TABLE_SCALE = 0.77


def pair(values):
    return "(" + ", ".join(scalar(v) for v in values) + ")"


def equation(row):
    return f"{linear_form(row[:2], ('x', 'y'))} = {scalar(row[2])}"


def rhs_calculation(step, before):
    """把与系数同步执行的常数运算单独写出来。"""
    after = step.augmented
    if len(step.rows) == 2:
        return f"{pair(before[:, 2])} → {pair(after[:, 2])}"
    r = step.rows[0]
    other = 1 - r
    old, source = before[r, 2], before[other, 2]
    factor = step.elementary[r, other]
    if factor:
        sign = "+" if factor > 0 else "−"
        term = scalar(source) if source >= 0 else f"({scalar(source)})"
        if abs(factor) != 1:
            term = f"{scalar(abs(factor))} × {term}"
        left = f"{scalar(old)} {sign} {term}"
    else:
        scale = step.elementary[r, r]
        if scale.numerator == 1:
            left = f"{scalar(old)} ÷ {scale.denominator}"
        else:
            term = scalar(old) if old >= 0 else f"({scalar(old)})"
            left = f"{scalar(scale)} × {term}"
    return f"{left} = {scalar(after[r, 2])}"


def clip_equation(row, low, high):
    """将 ax + by = c 裁到视窗，竖线和水平线也用同一套参数方程。"""
    a, b, c = np.asarray(row, dtype=float)
    length = np.hypot(a, b)
    if length == 0:
        return None  # 0 = c 表示整个平面或空集，不是一条直线。
    normal = np.array([a, b]) / length
    anchor = (c / length) * normal
    direction = np.array([-normal[1], normal[0]])
    enter, leave = -np.inf, np.inf
    for d, center, lo, hi in zip(direction, anchor, low, high):
        if abs(d) < 1e-12:
            if not lo - 1e-9 <= center <= hi + 1e-9:
                return None
        else:
            t1, t2 = (lo - center) / d, (hi - center) / d
            enter, leave = max(enter, min(t1, t2)), min(leave, max(t1, t2))
    if leave <= enter:
        return None
    return anchor + enter * direction, anchor + leave * direction


class Plot:
    def __init__(self, anchors, center_x, axis_names):
        points = np.asarray(anchors, dtype=float)
        lo, hi = points.min(axis=0), points.max(axis=0)
        margin = max(0.70, 0.17 * max(hi - lo))
        size = np.array([5.92, 3.46])
        span = size / min(size / (hi - lo + 2 * margin))
        self.low, self.high = (lo + hi - span) / 2, (lo + hi + span) / 2
        tick = max(1, int(np.ceil(max(span) / 11)))
        self.plane = NumberPlane(
            x_range=[self.low[0], self.high[0], tick], y_range=[self.low[1], self.high[1], tick],
            x_length=size[0], y_length=size[1], faded_line_ratio=0, tips=False,
            axis_config={"color": MUTED, "stroke_width": 1.6, "include_numbers": True,
                         "font_size": 17, "label_constructor": Text, "numbers_to_exclude": [0]},
            x_axis_config={"label_direction": DOWN}, y_axis_config={"label_direction": LEFT},
            background_line_style={"stroke_color": GRID, "stroke_width": 1.1},
        )
        self.plane.shift([center_x, -0.65, 0] - self.plane.c2p(*((lo + hi) / 2)))
        top_x = self.plane.c2p(self.high[0], 0)
        top_y = self.plane.c2p(0, self.high[1])
        self.grid = VGroup(self.plane, label("O", 15, MUTED).next_to(self.plane.c2p(0, 0), DOWN + LEFT, 0.05),
                           label(axis_names[0], 19, MUTED).next_to(top_x, RIGHT, 0.04),
                           label(axis_names[1], 19, MUTED).next_to(top_y, RIGHT, 0.04))

    def lines_for(self, state):
        lines = VGroup()
        for i, row in enumerate(state):
            ends = clip_equation(row, self.low, self.high)
            if ends is None:
                origin = self.plane.c2p(0, 0)
                line = Line(origin, origin + RIGHT * 0.01, stroke_opacity=0)
            else:
                line = Line(*(self.plane.c2p(*p) for p in ends), color=ROW_COLORS[i],
                            stroke_width=(5.1, 3.3)[i])
            lines.add(line)
        return lines

    def columns_for(self, state):
        origin = self.plane.c2p(0, 0)
        arrows = VGroup()
        for i, col in enumerate(state.T):
            end = self.plane.c2p(*np.asarray(col, dtype=float))
            if np.linalg.norm(end - origin) < 1e-9:
                arrow = Dot(origin, radius=0.045, color=COLUMN_COLORS[i])
            else:
                arrow = Arrow(origin, end, buff=0, color=COLUMN_COLORS[i],
                              stroke_width=(3.8, 3.8, 5.5)[i], tip_length=0.17)
            arrows.add(arrow)
        return arrows

    def point_label(self, position, color):
        tip = self.plane.c2p(*np.asarray(position, dtype=float))
        mob = label(pair(position), 22, color).next_to(tip, UP + RIGHT, buff=0.07)
        low, high = self.plane.c2p(*self.low), self.plane.c2p(*self.high)
        shift = np.maximum(low[:2] + 0.04 - mob.get_corner(DOWN + LEFT)[:2], 0)
        shift += np.minimum(high[:2] - 0.04 - mob.get_corner(UP + RIGHT)[:2], 0)
        return mob.shift([*shift, 0]).set_z_index(6)


def caption(heading, content, x, width, color=INK):
    return VGroup(place(label(heading, 22, MUTED), x, -2.85, width),
                  place(label(content, 29, color), x, -3.48, width))


class AugmentedSystem2D(Scene):
    def construct(self):
        self.camera.background_color = "#101722"
        data = np.asarray(AUGMENTED, dtype=object)
        if data.shape != (2, 3):
            raise ValueError("增广矩阵需要两行三列，每行填写两个系数和一个常数。")
        start, steps, missing = row_reduce(data[:, :2], rhs=data[:, 2])
        states = [start, *(step.augmented for step in steps)]
        final = states[-1]
        solution = final[:, 2] if missing is None else None

        # 左图用未知数坐标，右图用两个方程的输出坐标。两套网格在整段动画中固定。
        left_points = [(-1, -1), (1, 1)]
        if solution is not None:
            left_points.append(solution)
        for state in states:
            for a, b, c in state.astype(float):
                norm = np.hypot(a, b)
                if norm:
                    left_points.append((c / norm) * np.array([a, b]) / norm)
        left = Plot(left_points, -3.48, ("x", "y"))
        right = Plot([(0, 0), (1, 1), *(col for s in states for col in s.T)], 3.48, ("u", "v"))
        table = matrix_picture(start, augmented=True).scale(TABLE_SCALE).move_to([4.80, 2.24, 0])
        for i, cell in enumerate(table[1]):
            cell.set_color(COLUMN_COLORS[i % 3])

        def equation_label(r, content):
            return place(label(content, 28, ROW_COLORS[r]), -5.72, 2.59 - 0.7 * r, 5.32)

        equations = VGroup(*(equation_label(r, equation(row)) for r, row in enumerate(start)))
        row_names = VGroup(*(place(label(name, 22, ROW_COLORS[r]), -6.46, 2.59 - 0.7 * r, 0.57)
                             for r, name in enumerate(("R₁", "R₂"))))
        title = place(label("用增广矩阵解二元一次方程组", 36), -6.52, 3.46, 8.55)
        notation = label("A (x, y)ᵀ = b", 28).move_to([4.73, 3.46, 0])
        board_name = VGroup(place(label("增广矩阵", 27), 0.50, 2.49, 2.4),
                            place(label("[ A | b ]", 24, MUTED), 0.50, 1.96, 2.4))
        common = "白点是两条直线的交点" if solution is not None else "公共点构成解集"
        left_key = VGroup(label("未知数平面", 22), label(common, 20, MUTED)).arrange(RIGHT, buff=0.25)
        place(left_key, -6.47, 1.31, 6.04)
        right_key = VGroup(label("方程值平面", 22), *(label(s, 20, color) for s, color in zip(
            ("x 系数列", "y 系数列", "常数列 b"), COLUMN_COLORS))).arrange(RIGHT, buff=0.21)
        place(right_key, 0.50, 1.31, 6.04)
        rules = VGroup(Line([-6.52, 3.05, 0], [6.52, 3.05, 0]),
                       Line([0, 2.94, 0], [0, -2.42, 0]),
                       Line([-6.52, -2.54, 0], [6.52, -2.54, 0])).set_stroke(GRID, width=1.25)
        numbers = VGroup()
        motion = VGroup(left.lines_for(start), right.columns_for(start), numbers)
        marker = VGroup()
        if solution is not None:
            marker.add(Dot(left.plane.c2p(*np.asarray(solution, dtype=float)), radius=0.065, color=INK)
                       .set_stroke("#101722", width=1.25).set_z_index(5))
        info = VGroup(caption("每个方程给出一个条件", "公共解要同时满足两个方程", -6.47, 6.02),
                      caption("每个箭头对应增广矩阵的一列", "金色箭头跟踪常数列 b", 0.50, 6.05))
        self.play(FadeIn(VGroup(title, notation, rules, left.grid, right.grid, row_names, board_name,
                               left_key, right_key, table, equations)), FadeIn(motion), run_time=1.0)
        self.play(FadeIn(info), FadeIn(marker), run_time=0.65)
        self.wait(1.85)

        for n, step in enumerate(steps, start=1):
            before, after = states[n - 1], step.augmented
            rows = step.rows
            operation_hint = step.purpose
            if len(rows) == 1 and step.elementary[rows[0], rows[0]] != 1:
                operation_hint = f"第 {rows[0] + 1} 行缩放前后表示同一条直线"
            next_info = VGroup(caption(f"第 {n} 步  ·  {operation_hint}", step.operation,
                                       -6.47, 6.05, ROW_COLORS[rows[0]]),
                               caption("等号右边做相同运算", rhs_calculation(step, before), 0.50, 6.05, GOLD))
            bands = VGroup(*[Rectangle(width=table.width + 0.12, height=0.82 * TABLE_SCALE,
                                       color=ROW_COLORS[r], stroke_width=1.3, fill_opacity=0.065)
                             .move_to([table.get_x(), table[1][3 * r].get_y(), 0]).set_z_index(-1) for r in rows])
            self.play(FadeOut(info), run_time=0.28)
            updates = [Transform(equations[r], equation_label(
                r, f"{equation(before[r])}   →   {equation(after[r])}")) for r in rows]
            if n == 1:
                updates.append(Transform(board_name[1], place(label("系数 | 常数", 23, MUTED), 0.50, 1.96, 2.4)))
            self.play(FadeIn(next_info), FadeIn(bands), *updates, run_time=0.58)
            self.wait(1.10)
            self.move_system(left, right, table, motion, before, step)
            self.play(*(Transform(equations[r], equation_label(r, equation(after[r]))) for r in rows),
                      run_time=0.32)
            self.wait(1.25)
            self.play(FadeOut(bands), run_time=0.27)
            info = next_info

        self.play(FadeOut(info), run_time=0.32)
        marks = VGroup()
        if solution is not None:
            marks.add(left.point_label(solution, INK), right.point_label(solution, GOLD))
            check = start[:, :2] @ solution
            end = VGroup(caption("左图交点的坐标也是最后一列", f"(x, y) = {pair(solution)}", -6.47, 6.05),
                         caption("代回原方程组验算", f"A {pair(solution)}ᵀ = {pair(check)}ᵀ", 0.50, 6.05, GOLD))
        else:
            contradiction = next((row for row in final if not any(row[:2]) and row[2]), None)
            constraint = next((row for row in final if any(row[:2])), None)
            if contradiction is not None:
                end = VGroup(caption("出现矛盾行", equation(contradiction), -6.47, 6.05),
                             caption("两个条件没有公共点", "方程组无解", 0.50, 6.05, GOLD))
            else:
                condition = equation(constraint) if constraint is not None else "任意 (x, y)"
                end = VGroup(caption("解集", condition, -6.47, 6.05),
                             caption("至少有一个自由变量", "方程组有无穷多解", 0.50, 6.05, GOLD))
        ending = [FadeIn(end), FadeIn(marks)]
        if solution is not None:
            ending.append(Transform(board_name[1], place(label("[ I | 解 ]", 24, GOLD), 0.50, 1.96, 2.4)))
        self.play(*ending, run_time=0.65)
        self.wait(3.10)

    def move_system(self, left, right, table, motion, before, step):
        after = step.augmented
        base, delta = before.astype(float), (after - before).astype(float)
        changes = []
        for r, c in np.argwhere(before != after):
            cell, color = table[1][3 * r + c], COLUMN_COLORS[c]
            number = DecimalNumber(-max(abs(float(before[r, c])), abs(float(after[r, c]))),
                                   mob_class=Text, font_size=22, num_decimal_places=2,
                                   edge_to_fix=ORIGIN, color=color)
            if number.width > 0.75:
                number.scale_to_fit_width(0.75)
            number.set_value(float(before[r, c])).move_to(cell)
            exact = entry(after[r, c], color).scale(TABLE_SCALE).move_to(cell)
            cell.set_opacity(0)
            motion[2].add(number)
            changes.append((r, c, number, cell, exact))

        e = step.elementary
        flip = e[0, 0] * e[1, 1] - e[0, 1] * e[1, 0] < 0

        def update(group, alpha):
            # 加行与正倍缩放沿等价方程组连续移动。换行或负倍缩放则淡入对应的精确状态，
            # 避免线性插值经过奇异矩阵时显示一个多出解的中间方程组。
            t = float(alpha >= 0.5) if flip else alpha
            current = base + t * delta
            lines, arrows = left.lines_for(current), right.columns_for(current)
            if flip:
                fade = 1 - abs(2 * alpha - 1)
                lines.fade(fade)
                arrows.fade(fade)
            group[0].become(lines)
            group[1].become(arrows)
            for r, c, number, _, _exact in changes:
                value = round(current[r, c], 2) or 0.0
                if value != number.get_value():
                    number.set_value(value)
                if flip:
                    number.set_opacity(abs(2 * alpha - 1))

        self.play(UpdateFromAlphaFunc(motion, update), rate_func=smooth, run_time=2.55)
        for _, _, number, cell, exact in changes:
            motion[2].remove(number)
            cell.become(exact)
