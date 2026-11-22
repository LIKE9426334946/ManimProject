"""在仓库根目录运行 uv run manim -pqm 01_Linear/change_of_basis_2d.py ChangeOfBasis2D。"""

import numpy as np
from manim import (
    DOWN, LEFT, ORIGIN, RIGHT, UP, Arrow, DecimalNumber, Dot, FadeIn, FadeOut,
    GrowArrow, Line, NumberPlane, Polygon, Scene, Text, UpdateFromAlphaFunc,
    VGroup, VMobject, smooth,
)

from augmented_system_2d import clip_equation, pair
from inverse_by_row_operations import (
    CORAL, GOLD, GRID, IDENTITY, INK, MUTED, TEAL, entry, label, linear_form,
    matrix_picture, place, row_reduce, scalar,
)


# 全部用标准基 E 的坐标填写。NEW_BASIS 中依次写 b₁ 和 b₂；组矩阵时它们是列。
VECTOR = (3, 2)
NEW_BASIS = ((2, 1), (1, 2))
NAMES = ("b₁", "b₂")
COLORS = (TEAL, CORAL)


def determinant(b):
    return b[0, 0] * b[1, 1] - b[1, 0] * b[0, 1]


def basis_stages(target):
    """优先每次只移动一支基向量，并保证每个可见坐标系的基都线性无关。"""
    oriented = target.copy()
    reversed_axis = determinant(target) < 0
    if reversed_axis:
        oriented[:, 1] *= -1

    for first in (0, 1):
        middle = IDENTITY.copy()
        middle[:, first] = oriented[:, first]
        if determinant(middle) <= 0:
            continue
        old = IDENTITY
        for column, new in ((first, middle), (1 - first, oriented)):
            if not np.array_equal(old, new):
                start, delta = old.astype(float), (new - old).astype(float)
                # 另一列固定时，行列式随 t 线性变化，两个端点都为正。
                def path(t, a=start, d=delta):
                    return a + t * d

                yield (f"移动第 {column + 1} 支基向量", f"{NAMES[column]}   "
                       f"{pair(old[:, column])} → {pair(new[:, column])}", path, new)
            old = new
        break
    else:
        # 例如 B = −I。先分出旋转与正定伸缩，两支箭头就不会在中途共线。
        u, scales, vt = np.linalg.svd(oriented.astype(float))
        rotation = u @ vt
        angle = np.arctan2(rotation[1, 0], rotation[0, 0])
        stretch = (vt.T * scales) @ vt

        def path(t):
            c, s = np.cos(t * angle), np.sin(t * angle)
            return np.array([[c, -s], [s, c]]) @ ((1 - t) * np.eye(2) + t * stretch)

        yield "同时移动两支基向量", "旋转基并调整网格形状", path, oriented

    # 定向相反的两组基无法通过连续的可逆基连接，直接切换第二根轴的正向。
    if reversed_axis:
        yield "切换第二根轴的正方向", f"b₂   {pair(oriented[:, 1])} → {pair(target[:, 1])}", None, target


def coordinate_symbol(basis, size=28, color=INK):
    body = label("[v]", size, color)
    sub = label(basis, size * 0.57, color)
    sub.next_to(body, RIGHT, buff=0.035).align_to(body, DOWN).shift(0.065 * DOWN)
    return VGroup(body, sub)


class LiveMatrix(VGroup):
    """运动时显示小数，停稳后可以切换成精确分数。"""

    def __init__(self, initial, largest, center, scale, colors):
        layout = matrix_picture(initial).scale(scale).move_to(center)
        frame, slots = layout
        self.centers = [cell.get_center() for cell in slots]
        self.colors, self.entry_scale = colors, scale
        digits = VGroup()
        for j, (slot, value) in enumerate(zip(slots, initial.flat)):
            number = DecimalNumber(-largest, num_decimal_places=2, mob_class=Text,
                                   font_size=28, edge_to_fix=ORIGIN, color=colors[j])
            if number.width > 1.02 * scale:
                number.scale_to_fit_width(1.02 * scale)
            number.set_value(float(value)).move_to(slot)
            digits.add(number)
        super().__init__(frame, digits)

    def update_values(self, values):
        for number, raw in zip(self[1], np.ravel(values)):
            value = round(float(raw), 2) or 0.0
            if value != number.get_value():
                number.set_value(value)

    def exact_values(self, values):
        for number, raw, color, center in zip(self[1], np.ravel(values), self.colors, self.centers):
            number.become(entry(raw, color).scale(self.entry_scale).move_to(center))


class BasisDiagram:
    def __init__(self, vector, bases, coefficients):
        # 同一个视窗要容得下基向量、单位格和两段分量之间的拐点。
        corners = [[-0.45, -0.35], [1, 1], vector]
        for b, coeff in zip(bases, coefficients):
            corners.extend((b[:, 0], b[:, 1], b.sum(axis=1), coeff[0] * b[:, 0]))
        lo, hi = np.min(corners, axis=0), np.max(corners, axis=0)
        margin = max(0.58, 0.14 * max(hi - lo))
        box = np.array([6.98, 4.82])
        span = box / min(box / (hi - lo + 2 * margin))
        self.low, self.high = (hi + lo - span) / 2, (hi + lo + span) / 2
        tick = max(1, int(np.ceil(max(span) / 10)))
        self.plane = NumberPlane(
            x_range=[self.low[0], self.high[0], tick], y_range=[self.low[1], self.high[1], tick],
            x_length=box[0], y_length=box[1], tips=False, faded_line_ratio=0,
            axis_config={"color": "#929FAC", "stroke_width": 1.5, "include_numbers": True,
                         "label_constructor": Text, "font_size": 16, "numbers_to_exclude": [0]},
            x_axis_config={"label_direction": DOWN}, y_axis_config={"label_direction": LEFT},
            background_line_style={"stroke_color": "#354358", "stroke_width": 0.85},
        )
        self.plane.shift([-3.08, -0.22, 0] - self.plane.c2p(*((lo + hi) / 2)))
        self.origin = self.plane.c2p(0, 0)
        self.tip = self.plane.c2p(*vector)
        self.axes = VGroup(self.plane, label("O", 16, MUTED).next_to(self.origin, DOWN + LEFT, 0.06),
                           label("x", 20, MUTED).next_to(self.plane.c2p(self.high[0], 0), RIGHT, 0.05),
                           label("y", 20, MUTED).next_to(self.plane.c2p(0, self.high[1]), RIGHT, 0.05))

        self.window = np.array([(x, y) for x in (self.low[0], self.high[0])
                                for y in (self.low[1], self.high[1])]).T
        # 大范围时适当抽稀，同时让同一族网格线的间隔贯穿全片。
        ranges = [np.ptp(np.linalg.solve(b, self.window), axis=1) for b in bases]
        self.stride = np.maximum(1, np.ceil(np.max(ranges, axis=0) / 24)).astype(int)
        self.lattice = VGroup(*(VMobject(stroke_color=color, stroke_width=1.55, stroke_opacity=0.42)
                                for color in (CORAL, TEAL)))
        self.cell = Polygon(ORIGIN, RIGHT, RIGHT + UP, UP, color="#91B7C9", stroke_width=1.9,
                            fill_color=TEAL, fill_opacity=0.12).set_z_index(1)
        self.arrows = VGroup(*(Arrow(ORIGIN, RIGHT, buff=0, color=c).set_z_index(4) for c in COLORS))
        self.tags = VGroup(*(label(n, 23, c).set_z_index(7) for n, c in zip(NAMES, COLORS)))
        self.parts = VGroup(*(VMobject(stroke_color=c, stroke_width=4).set_z_index(3) for c in COLORS))
        self.joint = Dot(self.origin, radius=0.047, color=INK).set_z_index(5)
        self.moving = VGroup(self.lattice, self.cell, self.parts, self.arrows, self.tags, self.joint)
        self.draw(np.eye(2), vector)

    def draw(self, basis, coeff):
        # 对标准坐标点 p，B⁻¹p 的一项为整数，就落在相应的一条活动网格线上。
        for row, trail, stride in zip(np.linalg.inv(basis), self.lattice, self.stride):
            levels = (row @ self.window) / stride
            trail.clear_points()
            for k in range(int(np.ceil(min(levels))), int(np.floor(max(levels))) + 1):
                ends = clip_equation((*row, k * stride), self.low, self.high)
                if ends is not None:
                    trail.start_new_path(self.plane.c2p(*ends[0]))
                    trail.add_line_to(self.plane.c2p(*ends[1]))

        p, q = (self.plane.c2p(*column) for column in basis.T)
        far = self.plane.c2p(*basis.sum(axis=1))
        self.cell.set_points_as_corners([self.origin, p, far, q, self.origin])
        for arrow, tag, end, color, side in zip(self.arrows, self.tags, (p, q), COLORS,
                                               (DOWN + RIGHT, UP + LEFT)):
            arrow.become(Arrow(self.origin, end, buff=0, color=color, stroke_width=5.5, tip_length=0.18))
            tag.next_to(end, side, buff=0.09)
        # 两段路径分别是 c₁ b₁ 和 c₂ b₂；第二段必须落在固定的箭尖处。
        elbow = self.plane.c2p(*(coeff[0] * basis[:, 0]))
        finish = self.plane.c2p(*(basis @ coeff))
        for trail, start, end in zip(self.parts, (self.origin, elbow), (elbow, finish)):
            trail.clear_points()
            delta = end - start
            length = np.linalg.norm(delta)
            if length > 1e-9:
                n = max(1, int(np.ceil(length / 0.18)))
                step = delta / (n - 0.38)
                for k in range(n):
                    trail.start_new_path(start + k * step)
                    trail.add_line_to(start + (k + 0.62) * step)
        self.joint.move_to(elbow)


class ChangeOfBasis2D(Scene):
    def construct(self):
        self.camera.background_color = "#101722"
        # 共用仓库的有理数消元，片尾的分数直接来自 B c = v 的精确解。
        augmented, reduction, missing = row_reduce(np.asarray(NEW_BASIS, dtype=object).T, rhs=VECTOR)
        if missing is not None:
            raise ValueError("两个目标基向量必须线性无关，才能给出唯一的二维坐标。")
        target, standard = augmented[:, :2], augmented[:, 2]
        exact = (reduction[-1].augmented if reduction else augmented)[:, 2]
        vector = standard.astype(float)
        stages = list(basis_stages(target))
        samples = [np.eye(2)]
        for _, _, path, end in stages:
            if path is not None:
                samples.extend(path(t) for t in np.linspace(0, 1, 33))
            samples.append(end.astype(float))
        coords = [np.linalg.solve(b, vector) for b in samples]
        diagram = BasisDiagram(vector, samples, coords)

        matrix = LiveMatrix(IDENTITY, np.max(np.abs(samples)), [3.86, 1.42, 0], 0.93,
                            (TEAL, CORAL, TEAL, CORAL))
        reading = LiveMatrix(standard.reshape(2, 1), np.max(np.abs(coords)), [3.86, -1.07, 0],
                             0.92, COLORS)
        moving = VGroup(diagram.moving, matrix, reading)
        zero = np.all(vector == 0)
        fixed = (Dot(diagram.origin, radius=0.066, color=GOLD) if zero else
                 Arrow(diagram.origin, diagram.tip, buff=0, color=GOLD, stroke_width=5.3, tip_length=0.20))
        fixed.set_z_index(6)
        v_tag = label("v", 26, GOLD).next_to(diagram.tip, RIGHT + UP, buff=0.08).set_z_index(7)

        title = place(label("换基时，向量的坐标怎样变化？", 37), -6.55, 3.48, 8.55)
        reference = place(label(f"标准坐标   {pair(standard)}", 26, GOLD), 2.14, 3.48, 4.43)
        legend = VGroup(label("基向量 b₁", 22, TEAL), label("基向量 b₂", 22, CORAL),
                        label("固定向量 v", 22, GOLD), label("虚线为分量", 19, MUTED)).arrange(RIGHT, buff=0.29)
        place(legend, -6.51, 2.62, 6.96)
        rules = VGroup(Line([-6.54, 2.98, 0], [6.55, 2.98, 0]),
                       Line([0.74, 2.77, 0], [0.74, -2.51, 0]),
                       Line([-6.54, -2.68, 0], [6.55, -2.68, 0]),
                       Line([1.05, 0.38, 0], [6.50, 0.38, 0])).set_stroke(GRID, width=1.25)
        matrix_heading = place(label("两列分别是 b₁、b₂ 的标准坐标", 23, MUTED), 1.05, 2.60, 5.44)
        matrix_prefix = place(label("B =", 30), 1.07, 1.42, 1.6)
        coord_heading = place(label("v 在当前基下的坐标", 23, MUTED), 1.05, 0.02, 5.43)
        coord_prefix = VGroup(coordinate_symbol("B", 29), label("=", 29)).arrange(RIGHT, buff=0.12)
        place(coord_prefix, 1.07, -1.07, 1.65)
        combo = VGroup(label("v =", 27, GOLD), label("c₁ b₁", 27, TEAL),
                       label("+", 26), label("c₂ b₂", 27, CORAL)).arrange(RIGHT, buff=0.15)
        place(combo, 1.08, -2.30, 5.43)
        formula = VGroup(coordinate_symbol("B"), label("= B⁻¹", 28),
                         coordinate_symbol("E", color=GOLD)).arrange(RIGHT, buff=0.13)
        place(formula, 1.06, -3.58, 5.45)
        inverse_heading = place(label("从标准坐标求当前坐标", 21, MUTED), 1.06, -3.03, 5.45)

        def caption(heading, line):
            return VGroup(place(label(heading, 22, MUTED), -6.52, -3.03, 6.97),
                          place(label(line, 29), -6.52, -3.58, 6.97))

        def update(b):
            coeff = np.linalg.solve(b, vector)
            diagram.draw(b, coeff)
            matrix.update_values(b)
            reading.update_values(coeff)

        def walk():
            # 白点按首尾相接的顺序走两段，以便直接读出每个系数的含义。
            cursor = Dot(diagram.origin, radius=0.075, color=INK).set_z_index(8)
            self.play(FadeIn(cursor), run_time=0.25)
            self.play(cursor.animate.move_to(diagram.joint.get_center()), run_time=1.15)
            self.play(cursor.animate.move_to(diagram.tip), run_time=1.00)
            self.play(FadeOut(cursor), run_time=0.25)

        phase = caption("从标准基 E 出发", "v = " + linear_form(standard, ("e₁", "e₂")))
        self.play(FadeIn(VGroup(title, reference, rules, legend, diagram.axes, matrix_heading, matrix_prefix,
                               coord_heading, coord_prefix, inverse_heading, formula, combo)), FadeIn(moving),
                  run_time=0.95)
        self.play(FadeIn(fixed) if zero else GrowArrow(fixed), FadeIn(v_tag), FadeIn(phase), run_time=0.95)
        self.wait(0.95)
        walk()
        for heading, text, path, end in stages:
            self.play(FadeOut(phase), run_time=0.27)
            phase = caption(heading, text)
            self.play(FadeIn(phase), run_time=0.48)
            self.wait(0.70)
            if path is None:
                self.play(FadeOut(moving), run_time=0.46)
                update(end.astype(float))
                self.play(FadeIn(moving), run_time=0.46)
            else:
                self.play(UpdateFromAlphaFunc(moving, lambda _, t: update(path(t))), run_time=4.30, rate_func=smooth)
            self.wait(1.45)

        self.play(FadeOut(phase), run_time=0.28)
        matrix.exact_values(target)
        reading.exact_values(exact)
        result = caption("沿新基走到同一箭尖", f"沿 b₁ 走 {scalar(exact[0])} 格，沿 b₂ 走 {scalar(exact[1])} 格")
        self.play(FadeIn(result), run_time=0.50)
        walk()
        self.wait(3.00)
